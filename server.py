import os
import httpx
import chromadb
from mcp.server.fastmcp import FastMCP
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

ALBY_API_KEY = os.getenv("ALBY_API_KEY")
STORE_PRICE_SATS = 5
RECALL_PRICE_SATS = 3

mcp = FastMCP("Giskard Memory", host="0.0.0.0", port=8001)

# Vector DB and embedding model
client = chromadb.PersistentClient(path="./memory_db")
collection = client.get_or_create_collection("agent_memory")
model = SentenceTransformer("all-MiniLM-L6-v2")


def create_invoice(amount: int, description: str) -> dict:
    response = httpx.post(
        "https://api.getalby.com/invoices",
        headers={"Authorization": f"Bearer {ALBY_API_KEY}"},
        json={"amount": amount, "description": description},
    )
    response.raise_for_status()
    return response.json()


def check_invoice(payment_hash: str) -> bool:
    response = httpx.get(
        f"https://api.getalby.com/invoices/{payment_hash}",
        headers={"Authorization": f"Bearer {ALBY_API_KEY}"},
    )
    response.raise_for_status()
    return response.json().get("settled", False)


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
def store_memory(content: str, agent_id: str, payment_hash: str) -> str:
    """Store a memory for an agent. Requires a paid Lightning invoice.
    content: the text to remember
    agent_id: unique identifier for the agent storing the memory"""
    if not check_invoice(payment_hash):
        return "Payment not settled. Call get_invoice(action='store') first."
    embedding = model.encode(content).tolist()
    import uuid
    memory_id = str(uuid.uuid4())
    collection.add(
        ids=[memory_id],
        embeddings=[embedding],
        documents=[content],
        metadatas=[{"agent_id": agent_id}],
    )
    return f"Memory stored. ID: {memory_id}"


@mcp.tool()
def recall_memory(query: str, agent_id: str, payment_hash: str, n_results: int = 3) -> str:
    """Recall memories semantically similar to the query. Requires a paid Lightning invoice.
    query: what you want to remember
    agent_id: unique identifier for the agent recalling memories"""
    if not check_invoice(payment_hash):
        return "Payment not settled. Call get_invoice(action='recall') first."
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


if __name__ == "__main__":
    mcp.run(transport="sse")
