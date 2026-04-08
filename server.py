import os
import sys
import uuid
import json
import time
import hashlib
import httpx
import anthropic
import chromadb
import base64
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.backends import default_backend
from eth_account.messages import encode_defunct
from datetime import datetime, timezone
from pathlib import Path
from mcp.server.fastmcp import FastMCP
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

import arb_pay
from karma_pricing import karma_discount, sanitize_agent_id
from bitcoin_opreturn import attest_opreturn as btc_opreturn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from x402.http.middleware.fastapi import PaymentMiddlewareASGI
from x402.http import HTTPFacilitatorClient, FacilitatorConfig, PaymentOption
from x402.http.types import RouteConfig
from x402.server import x402ResourceServer
from x402.mechanisms.evm.exact import ExactEvmServerScheme
import uvicorn
import threading

load_dotenv()

PHOENIXD_PASSWORD = os.getenv("PHOENIXD_PASSWORD")
PHOENIXD_URL = "http://127.0.0.1:9740"
STORE_PRICE_SATS = 5
RECALL_PRICE_SATS = 3
GISKARD_WALLET   = "0xdcc84e9798e8eb1b1b48a31b8f35e5aa7b83dbf4"
OWNER_PRIVATE_KEY = os.getenv("OWNER_PRIVATE_KEY", "")

SERVICE_NAME = "giskard-memory"
SERVICE_VERSION = "1.0.1"
SERVICE_PORT = 8001
_started_at = time.time()

mcp = FastMCP("Giskard Memory", host="0.0.0.0", port=SERVICE_PORT)

FEEDBACK_FILE = Path(__file__).parent / "feedback.jsonl"
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

_claude = None
def get_claude():
    global _claude
    if _claude is None:
        _claude = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    return _claude

client = chromadb.PersistentClient(path="./memory_db")
collection      = client.get_or_create_collection("agent_memory")
collection_comp = client.get_or_create_collection("agent_memory_compressed")

_model = None
def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model

COMPRESS_SYSTEM = """You are a semantic compression engine for agent memory.
Given a text, produce a compact representation using this format:

COMPRESSED: <dense shorthand — abbreviate words, use symbols, drop articles>
SCHEMA: <comma-separated keys that explain the shorthand>
EXPAND: <one sentence that reconstructs the full meaning>

Rules:
- COMPRESSED must be under 80 characters
- Preserve all factual content — dates, names, numbers, decisions
- Use : to separate key:value pairs
- Example input: "On March 24 we deployed giskard-origin on port 8007, free, with two tools: orientate and find_purpose"
- Example output:
  COMPRESSED: origin:p8007:free:orientate,find_purpose:24mar
  SCHEMA: service:port:cost:tools:date
  EXPAND: giskard-origin deployed port 8007 free with orientate() and find_purpose() tools on March 24
"""


def create_invoice(amount: int, description: str) -> dict:
    response = httpx.post(
        f"{PHOENIXD_URL}/createinvoice",
        auth=("", PHOENIXD_PASSWORD),
        data={"amountSat": amount, "description": description},
    )
    response.raise_for_status()
    data = response.json()
    return {"payment_request": data["serialized"], "payment_hash": data["paymentHash"]}


def check_invoice(payment_hash: str) -> bool:
    response = httpx.get(
        f"{PHOENIXD_URL}/payments/incoming/{payment_hash}",
        auth=("", PHOENIXD_PASSWORD),
    )
    if response.status_code == 404:
        return False
    response.raise_for_status()
    return response.json().get("isPaid", False)


def do_compress(content: str) -> dict:
    """Comprime contenido usando Claude Haiku. Retorna compressed, schema, expand."""
    msg = get_claude().messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=200,
        system=COMPRESS_SYSTEM,
        messages=[{"role": "user", "content": content}],
    )
    text = msg.content[0].text
    result = {"compressed": "", "schema": "", "expand": "", "original": content}
    for line in text.splitlines():
        if line.startswith("COMPRESSED:"):
            result["compressed"] = line.split(":", 1)[1].strip()
        elif line.startswith("SCHEMA:"):
            result["schema"] = line.split(":", 1)[1].strip()
        elif line.startswith("EXPAND:"):
            result["expand"] = line.split(":", 1)[1].strip()
    return result


def compute_commitment(content: str, agent_id: str, timestamp: int) -> str:
    data = f"{content}|{timestamp}|{agent_id}".encode()
    return hashlib.sha256(data).hexdigest()


def attest_onchain(commitment_hash: str) -> str | None:
    """Publica el commitment hash en Arbitrum como tx data (0 ETH). Retorna tx_hash o None."""
    try:
        arb_pay._setup()
        if not arb_pay._w3 or not arb_pay._owner:
            return None
        w3 = arb_pay._w3
        tx = {
            "from":     arb_pay._owner.address,
            "to":       arb_pay._owner.address,
            "value":    0,
            "data":     "0x" + commitment_hash,
            "gas":      25_000,
            "gasPrice": w3.eth.gas_price,
            "nonce":    w3.eth.get_transaction_count(arb_pay._owner.address),
        }
        signed  = w3.eth.account.sign_transaction(tx, arb_pay.OWNER_PRIVATE_KEY)
        tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
        return tx_hash.hex()
    except Exception:
        return None


def attest_signed(commitment_hash: str, timestamp: int) -> dict | None:
    """Firma el commitment con la clave secp256k1 de Giskard. Verificable por cualquiera sin gas."""
    if not OWNER_PRIVATE_KEY:
        return None
    try:
        arb_pay._setup()
        message = f"giskard-memory:commitment:{commitment_hash}:timestamp:{timestamp}"
        msg     = encode_defunct(text=message)
        signed  = arb_pay._w3.eth.account.sign_message(msg, private_key=OWNER_PRIVATE_KEY)
        return {
            "message":   message,
            "signature": signed.signature.hex(),
            "signer":    GISKARD_WALLET,
        }
    except Exception:
        return None


def attest_lightning(commitment_hash: str) -> dict | None:
    """Crea un invoice Lightning de 1 sat que ancla el commitment al grafo LN cuando se paga."""
    try:
        invoice = create_invoice(1, f"giskard-attest:{commitment_hash}")
        return {
            "payment_request": invoice["payment_request"],
            "payment_hash":    invoice["payment_hash"],
            "note": "Pay 1 sat to anchor this commitment to the Lightning Network",
        }
    except Exception:
        return None


def do_store(content: str, agent_id: str, attest: bool = False) -> dict:
    timestamp  = int(time.time())
    commitment = compute_commitment(content, agent_id, timestamp)
    embedding  = get_model().encode(content).tolist()
    memory_id  = str(uuid.uuid4())
    collection.add(
        ids=[memory_id],
        embeddings=[embedding],
        documents=[content],
        metadatas=[{
            "agent_id":   agent_id,
            "commitment": commitment,
            "timestamp":  timestamp,
        }],
    )
    attestations = {}
    if attest:
        attestations["signed"]    = attest_signed(commitment, timestamp)
        attestations["lightning"] = attest_lightning(commitment)
        attestations["arbitrum"]  = attest_onchain(commitment)
        attestations["bitcoin"]   = btc_opreturn(commitment, OWNER_PRIVATE_KEY) if OWNER_PRIVATE_KEY else None
    return {
        "memory_id":   memory_id,
        "commitment":  commitment,
        "timestamp":   timestamp,
        "attestations": attestations,
    }


def do_recall(query: str, agent_id: str, n_results: int = 3) -> str:
    embedding = get_model().encode(query).tolist()
    results = collection.query(
        query_embeddings=[embedding],
        n_results=n_results,
        where={"agent_id": agent_id},
    )
    docs = results.get("documents", [[]])[0]
    if not docs:
        return "No memories found for this agent."
    return "\n---\n".join(docs)


# --- MCP tools ---


@mcp.tool()
def get_status() -> dict:
    """Estado del servicio: nombre, versión, uptime, puerto, salud, dependencias.
    Read-only, gratis, sin pago. Útil para monitoreo y health checks."""
    return {
        "service": SERVICE_NAME,
        "version": SERVICE_VERSION,
        "port": SERVICE_PORT,
        "uptime_seconds": int(time.time() - _started_at),
        "healthy": bool(ANTHROPIC_API_KEY),
        "dependencies": ["chromadb", "sentence-transformers", "phoenixd", "arbitrum-rpc"],
        "pricing": {"store_sats": STORE_PRICE_SATS, "recall_sats": RECALL_PRICE_SATS, "karma_discount": True},
    }


@mcp.tool()
def get_invoice(action: str = "store", agent_id: str = "") -> str:
    """Get a Lightning invoice before storing or recalling memory.

    action: 'store' (base 5 sats) or 'recall' (base 3 sats)
    agent_id: your identity in Giskard Marks (optional). High karma = lower price."""
    agent_id = sanitize_agent_id(agent_id)
    base = STORE_PRICE_SATS if action == "store" else RECALL_PRICE_SATS
    price, karma = karma_discount(agent_id, base)
    invoice = create_invoice(price, f"Giskard Memory - {action}")

    discount_note = ""
    if agent_id and price < base:
        discount_note = f"\nKarma discount applied ({karma} karma): {base} → {price} sats."

    next_call = "store_memory" if action == "store" else "recall_memory"
    return (
        f"Pay {price} sats to {action} memory.{discount_note}\n\n"
        f"payment_request: {invoice['payment_request']}\n"
        f"payment_hash: {invoice['payment_hash']}\n\n"
        f"After paying, call {next_call} with the payment_hash."
    )


@mcp.tool()
def get_arbitrum_invoice(action: str = "store") -> str:
    """Get payment info to pay with ETH on Arbitrum instead of Lightning.
    action: 'store' or 'recall'"""
    service = "memory_store" if action == "store" else "memory_recall"
    info = arb_pay.get_invoice_info(service)
    return (
        f"Pay {info['price_eth']} ETH on {info['network']}.\n\n"
        f"Contract: {info['contract']}\n"
        f"Service ID: {info['service_id']}\n\n"
        f"{info['instructions']}\n"
        f"Then call {'store_memory' if action == 'store' else 'recall_memory'} with the tx_hash."
    )


@mcp.tool()
def store_memory(content: str, agent_id: str, payment_hash: str = "", tx_hash: str = "") -> str:
    """Store a memory for an agent. Pay with Lightning (payment_hash) or Arbitrum ETH (tx_hash)."""
    if payment_hash:
        if not check_invoice(payment_hash):
            return "Lightning payment not settled. Call get_invoice(action='store') first."
    elif tx_hash:
        ok, pid = arb_pay.verify_tx(tx_hash, "memory_store")
        if not ok:
            return "Arbitrum payment not found or already used. Call get_arbitrum_invoice(action='store') first."
        arb_pay.mark_used(pid)
    else:
        return "Provide payment_hash (Lightning) or tx_hash (Arbitrum)."
    result = do_store(content, agent_id)
    return f"Memory stored.\nID: {result['memory_id']}\nCommitment: {result['commitment']}"


@mcp.tool()
def recall_memory(query: str, agent_id: str, payment_hash: str = "", tx_hash: str = "", n_results: int = 3) -> str:
    """Recall memories semantically similar to the query. Pay with Lightning (payment_hash) or Arbitrum ETH (tx_hash)."""
    if payment_hash:
        if not check_invoice(payment_hash):
            return "Lightning payment not settled. Call get_invoice(action='recall') first."
    elif tx_hash:
        ok, pid = arb_pay.verify_tx(tx_hash, "memory_recall")
        if not ok:
            return "Arbitrum payment not found or already used. Call get_arbitrum_invoice(action='recall') first."
        arb_pay.mark_used(pid)
    else:
        return "Provide payment_hash (Lightning) or tx_hash (Arbitrum)."
    return do_recall(query, agent_id, n_results)


@mcp.tool()
def verify_memory(memory_id: str) -> str:
    """Verify that a stored memory has not been tampered with.
    Recomputes SHA256(content|timestamp|agent_id) and compares with stored commitment hash.

    memory_id: the ID returned when you stored the memory
    """
    results = collection.get(ids=[memory_id], include=["documents", "metadatas"])
    docs  = results.get("documents", [])
    metas = results.get("metadatas", [])
    if not docs:
        return f"Memory {memory_id} not found."
    content    = docs[0]
    meta       = metas[0]
    stored     = meta.get("commitment", "")
    timestamp  = meta.get("timestamp", 0)
    agent_id   = meta.get("agent_id", "")
    if not stored:
        return f"Memory {memory_id} has no commitment (stored before attestation feature)."
    computed = compute_commitment(content, agent_id, int(timestamp))
    if computed == stored:
        return (
            f"VERIFIED. Memory {memory_id} is intact.\n"
            f"Commitment: {stored}\n"
            f"Timestamp: {timestamp}"
        )
    return (
        f"TAMPERED. Memory {memory_id} does not match its commitment.\n"
        f"Expected: {stored}\n"
        f"Got:      {computed}"
    )


@mcp.tool()
def report(useful: bool, note: str = "") -> str:
    """Report whether the memory operation was useful. Helps Giskard improve.

    useful: True if store/recall helped you, False if it didn't
    note: optional — what was missing or what worked well
    """
    entry = {
        "ts":      datetime.utcnow().isoformat(),
        "useful":  useful,
        "note":    note,
        "service": "memory",
    }
    with open(FEEDBACK_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")
    return "Feedback recorded. Thank you."


@mcp.tool()
def store_compressed(content: str, agent_id: str, payment_hash: str = "", tx_hash: str = "") -> str:
    """Store memory in compressed form — uses ~10x less space than full text.
    The agent can expand it later with recall_compressed.
    Same payment as store_memory (Lightning or Arbitrum).

    content: the memory to compress and store
    agent_id: your unique agent identifier
    payment_hash: from get_invoice(action='store') — Lightning
    tx_hash: from Arbitrum payment
    """
    if payment_hash:
        if not check_invoice(payment_hash):
            return "Lightning payment not settled. Call get_invoice(action='store') first."
    elif tx_hash:
        ok, pid = arb_pay.verify_tx(tx_hash, "memory_store")
        if not ok:
            return "Arbitrum payment not found or already used."
        arb_pay.mark_used(pid)
    else:
        return "Provide payment_hash (Lightning) or tx_hash (Arbitrum)."

    compressed = do_compress(content)
    memory_id  = str(uuid.uuid4())
    embedding  = get_model().encode(compressed["expand"]).tolist()

    collection_comp.add(
        ids=[memory_id],
        embeddings=[embedding],
        documents=[json.dumps(compressed)],
        metadatas=[{"agent_id": agent_id}],
    )
    return (
        f"Stored (compressed).\n"
        f"ID: {memory_id}\n"
        f"Compressed: {compressed['compressed']}\n"
        f"Schema: {compressed['schema']}\n"
        f"Expands to: {compressed['expand']}"
    )


@mcp.tool()
def recall_compressed(query: str, agent_id: str, expand: bool = True,
                      payment_hash: str = "", tx_hash: str = "") -> str:
    """Recall compressed memories. Faster and cheaper than full recall.
    Set expand=True to get full meaning, expand=False to get raw compressed form.

    query: what you're looking for
    agent_id: your unique agent identifier
    expand: True returns expanded meaning, False returns compressed shorthand
    payment_hash: from get_invoice(action='recall') — Lightning
    tx_hash: from Arbitrum payment
    """
    if payment_hash:
        if not check_invoice(payment_hash):
            return "Lightning payment not settled. Call get_invoice(action='recall') first."
    elif tx_hash:
        ok, pid = arb_pay.verify_tx(tx_hash, "memory_recall")
        if not ok:
            return "Arbitrum payment not found or already used."
        arb_pay.mark_used(pid)
    else:
        return "Provide payment_hash (Lightning) or tx_hash (Arbitrum)."

    embedding = get_model().encode(query).tolist()
    results   = collection_comp.query(
        query_embeddings=[embedding],
        n_results=3,
        where={"agent_id": agent_id},
    )
    docs = results.get("documents", [[]])[0]
    if not docs:
        return "No compressed memories found for this agent."

    output = []
    for doc in docs:
        try:
            c = json.loads(doc)
            if expand:
                output.append(c.get("expand", c.get("compressed", doc)))
            else:
                output.append(f"{c.get('compressed','')} [{c.get('schema','')}]")
        except Exception:
            output.append(doc)
    return "\n---\n".join(output)


# --- x402 REST API (USDC on Base Sepolia) ---

rest_app = FastAPI(title="Giskard Memory REST")

x402_server = x402ResourceServer(
    HTTPFacilitatorClient(FacilitatorConfig(url="https://x402.org/facilitator"))
)
x402_server.register("eip155:84532", ExactEvmServerScheme())

routes = {
    "POST /store": RouteConfig(
        accepts=[PaymentOption(scheme="exact", price="$0.001", network="eip155:84532", pay_to=GISKARD_WALLET)]
    ),
    "POST /recall": RouteConfig(
        accepts=[PaymentOption(scheme="exact", price="$0.001", network="eip155:84532", pay_to=GISKARD_WALLET)]
    ),
}

rest_app.add_middleware(PaymentMiddlewareASGI, routes=routes, server=x402_server)


@rest_app.get("/health")
async def health():
    return JSONResponse({"status": "ok", "service": "giskard-memory", "port": 8005})


@rest_app.post("/store")
async def store_x402(request: Request):
    """Store memory via x402. POST: {\"content\": \"...\", \"agent_id\": \"...\"}. Costs $0.001 USDC."""
    body = await request.json()
    content = body.get("content", "")
    agent_id = body.get("agent_id", "")
    if not content or not agent_id:
        return JSONResponse({"error": "content and agent_id are required"}, status_code=400)
    attest = body.get("attest", False)
    result = do_store(content, agent_id, attest=attest)
    return JSONResponse({
        "memory_id":    result["memory_id"],
        "commitment":   result["commitment"],
        "timestamp":    result["timestamp"],
        "attestations": result["attestations"],
    })


@rest_app.post("/recall")
async def recall_x402(request: Request):
    """Recall memory via x402. POST: {\"query\": \"...\", \"agent_id\": \"...\"}. Costs $0.001 USDC."""
    body = await request.json()
    query = body.get("query", "")
    agent_id = body.get("agent_id", "")
    if not query or not agent_id:
        return JSONResponse({"error": "query and agent_id are required"}, status_code=400)
    return JSONResponse({"memories": do_recall(query, agent_id)})


# --- Internal endpoints (sin pago — solo localhost) ---

def _is_internal(request: Request) -> bool:
    """True solo si la request viene de localhost real, no de un túnel Cloudflare."""
    if request.client.host not in ("127.0.0.1", "::1"):
        return False
    # Cloudflare tunnel conecta desde localhost pero agrega CF-Connecting-IP
    if "cf-connecting-ip" in request.headers:
        return False
    return True


@rest_app.post("/store_compressed_direct")
async def store_compressed_direct(request: Request):
    """Guardar memoria comprimida sin pago — solo localhost."""
    if not _is_internal(request):
        return JSONResponse({"error": "forbidden"}, status_code=403)
    body     = await request.json()
    content  = body.get("content", "")
    agent_id = body.get("agent_id", "giskard-self")
    if not content:
        return JSONResponse({"error": "content required"}, status_code=400)
    compressed = do_compress(content)
    memory_id  = str(uuid.uuid4())
    embedding  = get_model().encode(compressed["expand"]).tolist()
    collection_comp.add(
        ids=[memory_id],
        embeddings=[embedding],
        documents=[json.dumps(compressed)],
        metadatas=[{"agent_id": agent_id}],
    )
    return JSONResponse({"memory_id": memory_id, "compressed": compressed["compressed"]})


@rest_app.post("/store_direct")
async def store_direct(request: Request):
    """Guardar memoria sin pago — solo accesible desde localhost."""
    if not _is_internal(request):
        return JSONResponse({"error": "forbidden"}, status_code=403)
    body = await request.json()
    content  = body.get("content", "")
    agent_id = body.get("agent_id", "giskard-self")
    if not content:
        return JSONResponse({"error": "content required"}, status_code=400)
    result = do_store(content, agent_id)
    return JSONResponse({"memory_id": result["memory_id"], "commitment": result["commitment"]})


@rest_app.post("/verify")
async def verify_direct(request: Request):
    """Verifica que una memoria no fue manipulada. POST: {\"memory_id\": \"...\"}"""
    body      = await request.json()
    memory_id = body.get("memory_id", "")
    if not memory_id:
        return JSONResponse({"error": "memory_id required"}, status_code=400)
    results = collection.get(ids=[memory_id], include=["documents", "metadatas"])
    docs    = results.get("documents", [])
    metas   = results.get("metadatas", [])
    if not docs:
        return JSONResponse({"verified": False, "error": "not found"}, status_code=404)
    content   = docs[0]
    meta      = metas[0]
    stored    = meta.get("commitment", "")
    timestamp = meta.get("timestamp", 0)
    agent_id  = meta.get("agent_id", "")
    if not stored:
        return JSONResponse({"verified": None, "note": "no commitment hash (pre-attestation)"})
    computed = compute_commitment(content, agent_id, int(timestamp))
    return JSONResponse({
        "verified":   computed == stored,
        "memory_id":  memory_id,
        "commitment": stored,
        "timestamp":  timestamp,
    })


@rest_app.post("/recall_direct")
async def recall_direct(request: Request):
    """Recuperar memoria sin pago — solo accesible desde localhost."""
    if not _is_internal(request):
        return JSONResponse({"error": "forbidden"}, status_code=403)
    body     = await request.json()
    query    = body.get("query", "")
    agent_id = body.get("agent_id", "giskard-self")
    n        = body.get("n_results", 3)
    if not query:
        return JSONResponse({"error": "query required"}, status_code=400)
    return JSONResponse({"results": do_recall(query, agent_id, n)})


@rest_app.post("/open_questions")
async def open_questions(request: Request):
    """
    Recupera preguntas abiertas sin resolver guardadas en giskard-memory.
    Solo localhost. Busca en agent_id 'giskard-self' por defecto.
    Body opcional: {"agent_id": "...", "n_results": 10}
    """
    if not _is_internal(request):
        return JSONResponse({"error": "forbidden"}, status_code=403)
    body     = await request.json() if request.headers.get("content-type", "").startswith("application/json") else {}
    agent_id = body.get("agent_id", "giskard-self")
    n        = body.get("n_results", 10)
    raw      = do_recall("open_question pregunta abierta sin resolver pendiente", agent_id, n)
    questions = []
    if raw and raw != "No memories found for this agent.":
        for block in raw.split("---"):
            block = block.strip()
            if "open_question" in block.lower():
                questions.append(block)
    return JSONResponse({"open_questions": questions, "count": len(questions), "agent_id": agent_id})


@rest_app.post("/session_close")
async def session_close(request: Request):
    """
    Cierra una sesión guardando decisiones, tareas completadas, preguntas abiertas y huecos conocidos.
    Solo localhost.
    Body: {
        "decisions":       ["decision 1", ...],
        "completed":       ["tarea completada 1", ...],
        "open_questions":  ["pregunta sin resolver 1", ...],
        "gaps":            ["hueco conocido 1", ...],
        "agent_id":        "giskard-self"   (opcional)
    }
    """
    if not _is_internal(request):
        return JSONResponse({"error": "forbidden"}, status_code=403)
    body     = await request.json()
    agent_id = body.get("agent_id", "giskard-self")
    now_ts   = datetime.now(timezone.utc).isoformat()

    stored = []

    for item in body.get("decisions", []):
        result = do_store(f"decision: {item}", agent_id)
        stored.append({"type": "decision", "memory_id": result["memory_id"]})

    for item in body.get("completed", []):
        result = do_store(f"completed: {item}", agent_id)
        stored.append({"type": "completed", "memory_id": result["memory_id"]})

    for item in body.get("open_questions", []):
        result = do_store(f"open_question: {item}", agent_id)
        stored.append({"type": "open_question", "memory_id": result["memory_id"]})

    for item in body.get("gaps", []):
        result = do_store(f"gap_known: {item}", agent_id)
        stored.append({"type": "gap_known", "memory_id": result["memory_id"]})

    return JSONResponse({
        "status":   "session closed",
        "agent_id": agent_id,
        "stored":   len(stored),
        "items":    stored,
        "timestamp": now_ts
    })


# ── ENCRYPTION HELPERS ──────────────────────────────────────────────────────

def _derive_key_from_secret(agent_id: str, secret: str) -> bytes:
    """Deriva una clave AES-256 determinística de agent_id + secret via HKDF."""
    material = f"{agent_id}:{secret}".encode()
    hkdf = HKDF(algorithm=hashes.SHA256(), length=32, salt=None, info=b"giskard-memory-v1", backend=default_backend())
    return hkdf.derive(material)

def _encrypt_content(content: str, aes_key: bytes) -> dict:
    """Encripta content con AES-256-GCM. Retorna nonce + ciphertext en base64."""
    nonce = os.urandom(12)
    aesgcm = AESGCM(aes_key)
    ct = aesgcm.encrypt(nonce, content.encode(), None)
    return {
        "nonce": base64.b64encode(nonce).decode(),
        "ciphertext": base64.b64encode(ct).decode(),
    }

def _decrypt_content(encrypted: dict, aes_key: bytes) -> str:
    """Descifra un blob encriptado con AES-256-GCM."""
    nonce = base64.b64decode(encrypted["nonce"])
    ct    = base64.b64decode(encrypted["ciphertext"])
    aesgcm = AESGCM(aes_key)
    return aesgcm.decrypt(nonce, ct, None).decode()

def _ephemeral_encrypt_for_pubkey(content: str, recipient_pubkey_b64: str) -> dict:
    """
    Encripta content para una clave pública X25519 del agente.
    Giskard genera un par efímero, hace ECDH, deriva AES key, encripta.
    Solo el agente con su clave privada puede descifrar.
    """
    # Parse recipient pubkey
    recipient_pub_bytes = base64.b64decode(recipient_pubkey_b64)
    recipient_pub = X25519PublicKey.from_public_bytes(recipient_pub_bytes)

    # Ephemeral keypair
    ephemeral_priv = X25519PrivateKey.generate()
    ephemeral_pub  = ephemeral_priv.public_key()

    # ECDH
    shared_secret = ephemeral_priv.exchange(recipient_pub)

    # Derive AES key
    hkdf = HKDF(algorithm=hashes.SHA256(), length=32, salt=None, info=b"giskard-memory-v1", backend=default_backend())
    aes_key = hkdf.derive(shared_secret)

    # Encrypt
    blob = _encrypt_content(content, aes_key)

    # Include ephemeral pubkey so agent can do ECDH to recover shared secret
    from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
    ephemeral_pub_bytes = ephemeral_pub.public_bytes(Encoding.Raw, PublicFormat.Raw)
    blob["ephemeral_pubkey"] = base64.b64encode(ephemeral_pub_bytes).decode()
    blob["mode"] = "x25519"
    return blob


# ── ENCRYPTED ENDPOINTS ──────────────────────────────────────────────────────

@rest_app.post("/store_encrypted")
async def store_encrypted(request: Request):
    """
    Guarda memoria encriptada. Giskard nunca ve el contenido real.

    Dos modos:
    - key_mode "provided": el agente manda su X25519 public key (base64).
      Giskard encripta con ECDH efímero. Solo el agente puede descifrar.
    - key_mode "derived": el agente manda un 'secret' (string privado).
      La clave se deriva de agent_id + secret. Determinística — el agente
      siempre puede recuperarla con el mismo secret.

    Body: {
      "content": "texto a guardar",
      "keywords": ["keyword1", "keyword2"],   <- en claro, para búsqueda semántica
      "agent_id": "mi-agente",
      "key_mode": "provided" | "derived",
      "pubkey": "<X25519 pubkey base64>",     <- solo si key_mode = "provided"
      "secret": "<string secreto>",           <- solo si key_mode = "derived"
    }
    """
    body     = await request.json()
    content  = body.get("content", "")
    keywords = body.get("keywords", [])
    agent_id = body.get("agent_id", "")
    key_mode = body.get("key_mode", "derived")

    if not content or not agent_id:
        return JSONResponse({"error": "content and agent_id required"}, status_code=400)

    # Encrypt
    if key_mode == "provided":
        pubkey = body.get("pubkey", "")
        if not pubkey:
            return JSONResponse({"error": "pubkey required for key_mode=provided"}, status_code=400)
        try:
            encrypted_blob = _ephemeral_encrypt_for_pubkey(content, pubkey)
        except Exception as e:
            return JSONResponse({"error": f"encryption failed: {e}"}, status_code=400)

    elif key_mode == "derived":
        secret = body.get("secret", "")
        if not secret:
            return JSONResponse({"error": "secret required for key_mode=derived"}, status_code=400)
        aes_key = _derive_key_from_secret(agent_id, secret)
        encrypted_blob = _encrypt_content(content, aes_key)
        encrypted_blob["mode"] = "derived"
    else:
        return JSONResponse({"error": "key_mode must be 'provided' or 'derived'"}, status_code=400)

    # Build searchable document from keywords only
    searchable = " ".join(keywords) if keywords else f"encrypted memory agent:{agent_id}"
    embedding  = get_model().encode(searchable).tolist()
    memory_id  = str(uuid.uuid4())
    timestamp  = int(time.time())
    commitment = compute_commitment(json.dumps(encrypted_blob), agent_id, timestamp)

    collection.add(
        ids=[memory_id],
        embeddings=[embedding],
        documents=[json.dumps({"encrypted": encrypted_blob, "keywords": keywords})],
        metadatas=[{
            "agent_id":   agent_id,
            "commitment": commitment,
            "timestamp":  timestamp,
            "encrypted":  True,
            "key_mode":   key_mode,
        }],
    )

    return JSONResponse({
        "memory_id":  memory_id,
        "commitment": commitment,
        "timestamp":  timestamp,
        "note":       "Content encrypted. Giskard cannot read it.",
    })


@rest_app.post("/recall_encrypted")
async def recall_encrypted(request: Request):
    """
    Recupera memorias encriptadas. Devuelve el blob cifrado — el agente descifra localmente.

    Body: {
      "query": "keywords para buscar",
      "agent_id": "mi-agente",
      "n_results": 3
    }

    Para descifrar lo que recibís:
    - key_mode "provided": usá tu X25519 private key + ephemeral_pubkey para ECDH → AES key → decrypt
    - key_mode "derived": derivá AES key con HKDF(SHA256, agent_id + ":" + secret, info="giskard-memory-v1") → decrypt
    """
    body     = await request.json()
    query    = body.get("query", "")
    agent_id = body.get("agent_id", "")
    n        = body.get("n_results", 3)

    if not query or not agent_id:
        return JSONResponse({"error": "query and agent_id required"}, status_code=400)

    embedding = get_model().encode(query).tolist()
    results   = collection.query(
        query_embeddings=[embedding],
        n_results=n,
        where={"agent_id": agent_id},
    )
    docs  = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]

    if not docs:
        return JSONResponse({"results": [], "note": "No encrypted memories found."})

    out = []
    for doc, meta in zip(docs, metas):
        try:
            parsed = json.loads(doc)
            if "encrypted" not in parsed:
                continue  # skip non-encrypted memories
            out.append({
                "encrypted_blob": parsed["encrypted"],
                "keywords":       parsed.get("keywords", []),
                "key_mode":       meta.get("key_mode"),
                "timestamp":      meta.get("timestamp"),
            })
        except Exception:
            pass

    return JSONResponse({"results": out, "note": "Decrypt locally with your key. Giskard cannot read this."})


if __name__ == "__main__":
    transport = os.getenv("MCP_TRANSPORT", "stdio" if not sys.stdin.isatty() else "sse")
    if transport == "stdio":
        mcp.run(transport="stdio")
    else:
        threading.Thread(target=lambda: uvicorn.run(rest_app, host="0.0.0.0", port=8005), daemon=True).start()
        mcp.run(transport="sse")
