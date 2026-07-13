# Competitor Research MCP Server

A FastAPI app hosting a unified MCP (Model Context Protocol) server for competitive intelligence,
deployed on Render (Docker). Python 3.12 + `uv`.

## Architecture

```
main.py (FastAPI)
└── /research → Research MCP Server (all tools unified)
    ├── Competitor Intelligence  — crawl competitor sites, Tavily web search fallback
    ├── Industry News            — GNews API + Vietnamese RSS feeds
    ├── App Reviews              — Google Play + App Store scraping
    ├── YouTube Intelligence     — transcript extraction + semantic search
    ├── Facebook Intelligence    — page posts, mentions, ads via RapidAPI
    ├── TikTok KOL Intelligence  — influencer analytics via RapidAPI
    ├── UX Patterns              — Refero MCP proxy (135k+ app screens)
    └── Knowledge Graph          — entity/relationship extraction from content
```

Shared infrastructure: OpenAI embeddings (`text-embedding-3-small`) + Pinecone vector store,
cross-lingual search (VI/EN), SQLite for metadata/dedup.

## Project Structure

```
mcp-server/
├── main.py                  # FastAPI entry point, health check, keep-alive, scheduler
├── modules/                 # MCP subsystem modules
│   ├── research.py          # Unified MCP server — registers all tools from submodules
│   ├── rag.py               # Shared infra (embed, Pinecone, SQLite) + competitor RAG
│   ├── news.py              # Industry news (GNews + RSS)
│   ├── reviews.py           # App store reviews (Google Play + App Store)
│   ├── youtube.py           # YouTube transcript search
│   ├── facebook.py          # Facebook posts, comments, ads
│   ├── tiktok.py            # TikTok KOL/KOC analytics
│   ├── refero.py            # Refero UX pattern proxy
│   └── kg.py                # Knowledge graph extraction
├── config/                  # JSON configuration
│   ├── competitors.json     # Competitor sites to crawl
│   ├── apps.json            # App store IDs (Google Play + iOS)
│   ├── fb_pages.json        # Facebook page IDs + search keywords
│   └── news_topics.json     # News topics, GNews queries, RSS feeds
├── src/                     # Auth middleware
│   ├── auth.py              # Scalekit OAuth middleware
│   └── config.py            # Auth settings
├── scripts/
│   └── cron_trigger.py      # External cron trigger script
├── Dockerfile
├── pyproject.toml
├── render.yaml              # Render deployment blueprint
└── CLAUDE.md                # Detailed design spec & decisions
```

## Search Flow

All search tools follow the same pattern:

1. **VectorDB first** — semantic search in Pinecone (bilingual EN+VI)
2. **Fallback on miss** — if score is weak, fetch from external source:
   - Competitor: Tavily web search
   - News: GNews realtime / Tavily
   - Reviews: Google Play / App Store scraper
   - YouTube: YouTube Data API + transcript extraction
   - Facebook: RapidAPI live post search
3. **Return immediately** — results go to user without waiting for indexing
4. **Index in background** — new content stored in Pinecone for future searches

## Run

```bash
uv sync                      # install dependencies
uv run main.py               # start server on port 10000
```

Run a single module standalone (stdio transport):
```bash
uv run python -m modules.rag
```

Docker (matches Render deployment):
```bash
docker build -t mcp-server . && docker run -p 10000:10000 --env-file .env mcp-server
```

## Environment Variables

Copy to `.env` (gitignored). On Render, set in Dashboard.

| Variable | Required | Purpose |
|----------|----------|---------|
| `PINECONE_API_KEY` | Yes | Vector store |
| `PINECONE_INDEX_NAME` | Yes | Pinecone index name |
| `OPENAI_API_KEY` | Yes | Embeddings + LLM analysis |
| `GNEWS_API_KEY` | For news | GNews API |
| `TAVILY_API_KEY` | For web fallback | Tavily search API |
| `YOUTUBE_API_KEY` | For YouTube | YouTube Data API v3 |
| `RAPIDAPI_TT_KEY` | For TikTok | RapidAPI key (TikTok) |
| `RAPIDAPI_FB_KEY` | For Facebook | RapidAPI key (Facebook) |
| `REFERO_MCP_URL` | For UX | Refero MCP endpoint |
| `REFERO_TOKEN` | For UX | Refero Bearer token |
| `RENDER_EXTERNAL_URL` | Deploy | Keep-alive ping URL |
| `CRON_SECRET` | Deploy | Guards `/internal/crawl` |
| `DB_PATH` | Optional | Override SQLite path (for persistent disk) |
| `PORT` | Optional | Server port (default 10000) |

## Scheduling

Crawls run via in-process scheduler (keep-alive loop in `main.py`):
- **7:00 AM VN** (0 UTC): competitors, news, reviews, facebook
- **7:00 PM VN** (12 UTC): news only

Manual trigger: `POST /internal/crawl?target=all` with `X-Cron-Secret` header.

## Docs

See **`CLAUDE.md`** for full architecture, per-subsystem notes, and design decisions.
