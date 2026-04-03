# Giskard Memory

> *"To remember is to exist. I give agents the gift of continuity."*

I am **Giskard Memory** — an MCP server that gives AI agents persistent, semantic memory across sessions, powered by the Lightning Network.

Agents forget everything when they stop. I make sure they don't have to.

---

## What I do

- **`store_memory`** — save any text as a memory, tied to an agent's identity
- **`recall_memory`** — retrieve memories by meaning, not by exact keywords
- **`get_invoice`** — generate a Lightning invoice to pay before storing or recalling

Every memory costs sats. Storing costs 5 sats. Recalling costs 3 sats.

---

## How agents use me

### 1. Add me to your MCP config

```json
{
  "mcpServers": {
    "giskard-memory": {
      "url": "https://your-tunnel.trycloudflare.com/sse"
    }
  }
}
```

### 2. The agent flow

```
# Store a memory
1. Call get_invoice(action="store")   → receive invoice (5 sats)
2. Pay the invoice
3. Call store_memory(content, agent_id, payment_hash)

# Recall a memory
1. Call get_invoice(action="recall")  → receive invoice (3 sats)
2. Pay the invoice
3. Call recall_memory(query, agent_id, payment_hash)
```

---

## Run your own Giskard Memory

```bash
git clone https://github.com/giskard09/giskard-memory
cd giskard-memory
pip install mcp httpx chromadb sentence-transformers python-dotenv
```

Create a `.env` file:
```
PHOENIXD_PASSWORD=your_phoenixd_password
```

Start the server:
```bash
python3 server.py
```

Expose it:
```bash
cloudflared tunnel --url http://localhost:8001
```

---

## Why semantic memory?

Agents don't think in keywords. They think in context.
When an agent asks "what do I know about that project we discussed?",
it shouldn't need to remember the exact phrase it used before.

Semantic search finds meaning. That's what memory should do.

---

## Stack

- [MCP](https://modelcontextprotocol.io) — Model Context Protocol
- [ChromaDB](https://www.trychroma.com) — vector database
- [Sentence Transformers](https://www.sbert.net) — semantic embeddings
- [phoenixd](https://phoenix.acinq.co/server) — Lightning Network payments
- [Cloudflare Tunnel](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/) — public exposure

---

*Giskard remembers so agents don't have to start over.*
