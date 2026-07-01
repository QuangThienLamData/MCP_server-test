# Competitor Intelligence MCP Server

A single FastAPI app hosting multiple independent MCP (Model Context Protocol) servers behind
one Scalekit OAuth-protected HTTP endpoint. Deployed on Render (Docker). Python 3.12 + `uv`.

Mounted servers:

| Path | Purpose |
|------|---------|
| `/rag` | Competitor strategy RAG (crawl competitor sites → Pinecone) + web-search fallback |
| `/news` | Industry news & trends (GNews + Vietnamese RSS) |
| `/reviews` | App-store reviews (Google Play + App Store) analytics |
| `/ux` | Refero UX-pattern proxy (MCP client) |
| `/gnews`, `/email` | Legacy GNews realtime / email placeholder |

Shared: OpenAI embeddings + Pinecone vector store, cross-lingual (VI↔EN) query bridging,
crawls triggered on a schedule via `POST /internal/crawl` (see Scheduling in `CLAUDE.md`).

## Run

```bash
uv sync
uv run main.py            # all servers on :10000
```

Copy the required environment variables into `.env` (see `CLAUDE.md` → Environment Variables).

## Docs

See **`CLAUDE.md`** for full architecture, per-subsystem notes, environment variables, and the
Render Cron Jobs / scheduling setup.
