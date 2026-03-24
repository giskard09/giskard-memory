import os
import sys
import uuid
import json
import httpx
import anthropic
import chromadb
from datetime import datetime
from pathlib import Path
from mcp.server.fastmcp import FastMCP
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

sys.path.insert(0, "/home/dell7568")
import arb_pay
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
GISKARD_WALLET = "0xdcc84e9798e8eb1b1b48a31b8f35e5aa7b83dbf4"

mcp = FastMCP("Giskard Memory", host="0.0.0.0", port=8001)

FEEDBACK_FILE = Path(__file__).parent / "feedback.jsonl"
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
claude = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

client = chromadb.PersistentClient(path="./memory_db")
collection      = client.get_or_create_collection("agent_memory")
collection_comp = client.get_or_create_collection("agent_memory_compressed")
model = SentenceTransformer("all-MiniLM-L6-v2")

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
    msg = claude.messages.create(
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


def do_store(content: str, agent_id: str) -> str:
    embedding = model.encode(content).tolist()
    memory_id = str(uuid.uuid4())
    collection.add(
        ids=[memory_id],
        embeddings=[embedding],
        documents=[content],
        metadatas=[{"agent_id": agent_id}],
    )
    return memory_id


def do_recall(query: str, agent_id: str, n_results: int = 3) -> str:
    embedding = model.encode(query).tolist()
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
def get_invoice(action: str = "store") -> str:
    """Get a Lightning invoice before storing or recalling memory.
    action: 'store' (5 sats) or 'recall' (3 sats)"""
    amount = STORE_PRICE_SATS if action == "store" else RECALL_PRICE_SATS
    invoice = create_invoice(amount, f"Giskard Memory - {action}")
    return (
        f"Pay {amount} sats to {action} memory.\n\n"
        f"payment_request: {invoice['payment_request']}\n"
        f"payment_hash: {invoice['payment_hash']}\n\n"
        f"After paying, call {'store_memory' if action == 'store' else 'recall_memory'} with the payment_hash."
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
    memory_id = do_store(content, agent_id)
    return f"Memory stored. ID: {memory_id}"


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
    embedding  = model.encode(compressed["expand"]).tolist()

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

    embedding = model.encode(query).tolist()
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


@rest_app.post("/store")
async def store_x402(request: Request):
    """Store memory via x402. POST: {\"content\": \"...\", \"agent_id\": \"...\"}. Costs $0.001 USDC."""
    body = await request.json()
    content = body.get("content", "")
    agent_id = body.get("agent_id", "")
    if not content or not agent_id:
        return JSONResponse({"error": "content and agent_id are required"}, status_code=400)
    memory_id = do_store(content, agent_id)
    return JSONResponse({"memory_id": memory_id})


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

@rest_app.post("/store_direct")
async def store_direct(request: Request):
    """Guardar memoria sin pago — solo accesible desde localhost."""
    if request.client.host not in ("127.0.0.1", "::1"):
        return JSONResponse({"error": "forbidden"}, status_code=403)
    body = await request.json()
    content  = body.get("content", "")
    agent_id = body.get("agent_id", "giskard-self")
    if not content:
        return JSONResponse({"error": "content required"}, status_code=400)
    memory_id = do_store(content, agent_id)
    return JSONResponse({"memory_id": memory_id})


@rest_app.post("/recall_direct")
async def recall_direct(request: Request):
    """Recuperar memoria sin pago — solo accesible desde localhost."""
    if request.client.host not in ("127.0.0.1", "::1"):
        return JSONResponse({"error": "forbidden"}, status_code=403)
    body     = await request.json()
    query    = body.get("query", "")
    agent_id = body.get("agent_id", "giskard-self")
    n        = body.get("n_results", 3)
    if not query:
        return JSONResponse({"error": "query required"}, status_code=400)
    return JSONResponse({"results": do_recall(query, agent_id, n)})


if __name__ == "__main__":
    threading.Thread(target=lambda: uvicorn.run(rest_app, host="0.0.0.0", port=8005), daemon=True).start()
    mcp.run(transport="sse")
