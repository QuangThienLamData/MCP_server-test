import hashlib
import json
import logging
import os
import sqlite3
import threading
from datetime import datetime, timedelta, timezone

import feedparser
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

# Reuse shared infra from the RAG module: same Pinecone index (different namespace),
# same OpenAI embedding helper, same SQLite DB file.
from modules.rag import DB_PATH, OPENAI_API_KEY, PINECONE_API_KEY, _bilingual_queries, _detect_lang, _embed, _get_index

load_dotenv()

logger = logging.getLogger(__name__)

GNEWS_API_KEY = os.getenv("GNEWS_API_KEY", "")
GNEWS_BASE = "https://gnews.io/api/v4"
NEWS_NAMESPACE = "news_trends"
NEWS_TOPICS_CONFIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "news_topics.json")

mcp = FastMCP(
    name="Industry News MCP Server",
    transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
)

_news_crawl_status = {
    "running": False, "total": 0, "done": 0, "current": "", "errors": [], "new_articles": 0,
}


# --- SQLite (article listing / digest; Pinecone holds the vectors) ---
def _init_news_db():
    conn = sqlite3.connect(DB_PATH)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS news_articles (
            id TEXT PRIMARY KEY,
            topic TEXT,
            source TEXT,
            article_url TEXT NOT NULL UNIQUE,
            title TEXT,
            description TEXT,
            published_at TEXT,
            language TEXT,
            crawled_at TEXT NOT NULL
        );
    """)
    conn.commit()
    conn.close()


def _load_topics() -> dict:
    if not os.path.exists(NEWS_TOPICS_CONFIG):
        logger.warning(f"News topics config not found: {NEWS_TOPICS_CONFIG}")
        return {}
    with open(NEWS_TOPICS_CONFIG, encoding="utf-8") as f:
        return json.load(f)


# --- GNews client ---
def _gnews_search(keyword: str, lang: str = "en", country: str | None = None,
                  from_date: str | None = None, max_results: int = 10) -> list[dict]:
    params = {"q": keyword, "lang": lang, "max": max_results, "sortby": "publishedAt", "apikey": GNEWS_API_KEY}
    if country:
        params["country"] = country
    if from_date:
        params["from"] = from_date
    r = requests.get(f"{GNEWS_BASE}/search", params=params, timeout=30)
    r.raise_for_status()
    return r.json().get("articles", [])


def _gnews_top_headlines(category: str = "technology", lang: str = "en",
                         country: str | None = None, max_results: int = 10) -> list[dict]:
    params = {"category": category, "lang": lang, "max": max_results, "apikey": GNEWS_API_KEY}
    if country:
        params["country"] = country
    r = requests.get(f"{GNEWS_BASE}/top-headlines", params=params, timeout=30)
    r.raise_for_status()
    return r.json().get("articles", [])


def _fetch_feed_articles(feed_url: str, source_name: str, filters: list[str]) -> list[dict]:
    """Parse an RSS feed and return GNews-shaped article dicts, keeping only entries
    whose title/summary mention a topic keyword (feeds are per-section, not per-topic)."""
    out: list[dict] = []
    try:
        feed = feedparser.parse(feed_url)
    except Exception as e:
        logger.error(f"RSS parse failed {feed_url}: {e}")
        return out
    filters_l = [f.lower() for f in filters]
    for e in feed.entries:
        title = getattr(e, "title", "") or ""
        raw = getattr(e, "summary", "") or getattr(e, "description", "") or ""
        summary = BeautifulSoup(raw, "html.parser").get_text(" ", strip=True) if raw else ""
        if filters_l and not any(f in f"{title} {summary}".lower() for f in filters_l):
            continue  # not relevant to this topic
        pub = ""
        if getattr(e, "published_parsed", None):
            pub = datetime(*e.published_parsed[:6]).strftime("%Y-%m-%dT%H:%M:%SZ")
        out.append({
            "url": getattr(e, "link", ""),
            "title": title,
            "description": summary,
            "content": "",
            "publishedAt": pub,
            "source": {"name": source_name, "url": feed_url},
        })
    return out


# --- Indexing ---
def _index_articles(articles: list[dict], topic: str) -> int:
    """Embed new articles into Pinecone (namespace news_trends) + record in SQLite. Skips dupes by URL."""
    if not articles:
        return 0
    conn = sqlite3.connect(DB_PATH)
    pending = []
    for a in articles:
        url = a.get("url") or ""
        if not url:
            continue
        if conn.execute("SELECT 1 FROM news_articles WHERE article_url = ?", (url,)).fetchone():
            continue  # already indexed
        title = a.get("title") or ""
        desc = a.get("description") or ""
        content = a.get("content") or ""
        text = f"{title}. {desc} {content}".strip()
        if len(text) < 20:
            continue
        pub_date = (a.get("publishedAt") or "")[:10]
        date_compact = pub_date.replace("-", "") if pub_date else "unknown"
        vid = f"news_{topic}_{date_compact}_{hashlib.md5(url.encode()).hexdigest()[:8]}"
        src = a.get("source") or {}
        pending.append({
            "id": vid, "text": text, "url": url, "title": title,
            "description": desc[:500], "published_at": pub_date,
            "source": src.get("name", ""), "source_url": src.get("url", ""),
        })

    if not pending:
        conn.close()
        return 0

    try:
        embeddings = _embed([p["text"] for p in pending])
    except Exception as e:
        logger.error(f"News embed failed [{topic}]: {e}")
        _news_crawl_status["errors"].append(f"Embed {topic}: {e}")
        conn.close()
        return 0

    now = datetime.now(timezone.utc).isoformat()
    vectors = []
    for p, emb in zip(pending, embeddings):
        alang = _detect_lang(p["text"])  # actual article language, not the query's lang
        vectors.append({
            "id": p["id"], "values": emb,
            "metadata": {
                "topic": topic, "source": p["source"], "source_url": p["source_url"],
                "article_url": p["url"], "title": p["title"], "description": p["description"],
                "published_at": p["published_at"], "language": alang, "crawled_at": now,
            },
        })
        conn.execute(
            "INSERT OR IGNORE INTO news_articles (id,topic,source,article_url,title,description,published_at,language,crawled_at) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (p["id"], topic, p["source"], p["url"], p["title"], p["description"], p["published_at"], alang, now),
        )

    index = _get_index()
    for b in range(0, len(vectors), 100):
        index.upsert(vectors=vectors[b:b + 100], namespace=NEWS_NAMESPACE)
    conn.commit()
    conn.close()
    return len(vectors)


def _crawl_news_background(wanted_topics: list[str]):
    try:
        topics_cfg = _load_topics()
        if wanted_topics:
            topics_cfg = {k: v for k, v in topics_cfg.items() if k in wanted_topics}
        _news_crawl_status["total"] = len(topics_cfg)
        for i, (name, cfg) in enumerate(topics_cfg.items()):
            _news_crawl_status["done"] = i
            _news_crawl_status["current"] = name
            # Each topic has a list of queries; mixing lang=en and lang=vi builds a
            # bilingual corpus so Vietnamese and English searches both have content.
            for qobj in cfg.get("queries", []):
                q = qobj.get("q", "")
                if not q:
                    continue
                lang = qobj.get("lang", "en")
                country = qobj.get("country")
                try:
                    # No date filter: GNews returns newest-first (max 10) and free-tier
                    # coverage is sparse, so a narrow window yields nothing. Dedup by URL
                    # (in _index_articles) prevents re-embedding already-seen articles.
                    articles = _gnews_search(q, lang=lang, country=country, max_results=10)
                    _news_crawl_status["new_articles"] += _index_articles(articles, name)
                except Exception as e:
                    logger.error(f"News search failed [{name}/{q}]: {e}")
                    _news_crawl_status["errors"].append(f"{name}/{q[:40]}: {e}")
            # Vietnamese RSS feeds (VnExpress/CafeF) — GNews free has ~no VN coverage.
            # feed_filters keep only topic-relevant entries from these section-wide feeds.
            for feed in cfg.get("feeds", []):
                try:
                    entries = _fetch_feed_articles(feed["url"], feed.get("source", ""), cfg.get("feed_filters", []))
                    _news_crawl_status["new_articles"] += _index_articles(entries, name)
                except Exception as e:
                    logger.error(f"RSS feed failed [{name}/{feed.get('url')}]: {e}")
                    _news_crawl_status["errors"].append(f"{name}/rss {feed.get('url', '')[:40]}: {e}")
            # NOTE: category top-headlines are intentionally NOT indexed per topic —
            # they are generic (not keyword-filtered) and pollute topic relevance.
            # They remain available in real time via the get_top_headlines tool.
        _news_crawl_status["done"] = len(topics_cfg)
        _news_crawl_status["current"] = ""
        logger.info(f"News crawl done: {_news_crawl_status['new_articles']} new, {len(_news_crawl_status['errors'])} errors")
    except Exception as e:
        logger.exception("News crawl failed")
        _news_crawl_status["errors"].append(str(e))
    finally:
        _news_crawl_status["running"] = False


# --- Formatting helpers ---
def _format_articles(articles: list[dict], header: str) -> str:
    if not articles:
        return "No articles found."
    lines = [f"{header} ({len(articles)} articles):"]
    for i, a in enumerate(articles):
        src = (a.get("source") or {}).get("name", "?")
        pub = (a.get("publishedAt") or "")[:10]
        lines.append(
            f"\n[{i + 1}] {a.get('title', '?')} — {src} ({pub})\n"
            f"{a.get('description', '')}\n{a.get('url', '')}"
        )
    return "\n".join(lines)


# --- MCP Tools ---

@mcp.tool()
def search_industry_trends(query: str, topic: str = "", from_date: str = "", to_date: str = "", top_k: int = 5) -> str:
    """
    Search indexed industry news & trends (fintech, AI, product, growth) by meaning.
    Sources may be in any language; ALWAYS answer the user in Vietnamese or English.

    Args:
        query: Natural-language question
        topic: Filter by topic: fintech, ai_product, growth_marketing, regulatory (optional)
        from_date: From date YYYY-MM-DD (optional)
        to_date: To date YYYY-MM-DD (optional)
        top_k: Number of results (default 5)
    """
    if not PINECONE_API_KEY or not OPENAI_API_KEY:
        return "Error: PINECONE_API_KEY or OPENAI_API_KEY not set."

    # Search with the query in both EN and VI so cross-lingual content is retrieved.
    variants = _bilingual_queries(query)
    try:
        qvecs = _embed(variants)
    except Exception as e:
        return f"Embedding error: {e}"

    filters: dict = {}
    if topic:
        filters["topic"] = {"$eq": topic}
    if from_date and to_date:
        filters["published_at"] = {"$gte": from_date, "$lte": to_date}
    elif from_date:
        filters["published_at"] = {"$gte": from_date}
    elif to_date:
        filters["published_at"] = {"$lte": to_date}

    index = _get_index()
    merged: dict = {}
    for qvec in qvecs:
        res = index.query(
            vector=qvec, top_k=top_k, include_metadata=True,
            namespace=NEWS_NAMESPACE, filter=filters or None,
        )
        for m in res.matches:
            if m.id not in merged or m.score > merged[m.id].score:
                merged[m.id] = m
    matches = sorted(merged.values(), key=lambda m: m.score, reverse=True)[:top_k]
    if not matches:
        return "No indexed news found. Run trigger_crawl first, or use search_news_realtime."

    out = [f"Found {len(matches)} results for: '{query}' (searched EN+VI)\n"]
    for i, m in enumerate(matches):
        md = m.metadata
        out.append(
            f"\n--- Result {i + 1} (score: {m.score:.3f}) ---\n"
            f"Topic: {md.get('topic', '?')}\n"
            f"Title: {md.get('title', '?')}\n"
            f"Source: {md.get('source', '?')}\n"
            f"URL: {md.get('article_url', '')}\n"
            f"Published: {md.get('published_at', '?')}\n"
            f"{md.get('description', '')}\n"
        )
    return "".join(out)


@mcp.tool()
def get_top_headlines(category: str = "technology", lang: str = "en", country: str = "", max_results: int = 10) -> str:
    """
    Fetch current top headlines from GNews in real time (not from the vector store).
    Content may be in any language; ALWAYS answer the user in Vietnamese or English.

    Args:
        category: general, world, nation, business, technology, entertainment, sports, science, health
        lang: 2-letter language code (default 'en')
        country: 2-letter country code (optional)
        max_results: Number of articles, 1-100 (default 10)
    """
    if not GNEWS_API_KEY:
        return "Error: GNEWS_API_KEY not set."
    try:
        articles = _gnews_top_headlines(category=category, lang=lang, country=country or None, max_results=max_results)
    except Exception as e:
        return f"GNews error: {e}"
    return _format_articles(articles, f"Top headlines · {category}")


@mcp.tool()
def search_news_realtime(keyword: str, lang: str = "en", from_date: str = "", max_results: int = 10) -> str:
    """
    Search GNews in real time by keyword. Results are NOT stored in the vector DB.
    Supports boolean operators (AND, OR, NOT) and "exact phrases".
    Content may be in any language; ALWAYS answer the user in Vietnamese or English.

    Args:
        keyword: Search query, e.g. 'fintech AND Vietnam'
        lang: 2-letter language code (default 'en')
        from_date: Published on/after this ISO 8601 date, e.g. '2024-01-15T00:00:00Z' (optional)
        max_results: Number of articles, 1-100 (default 10)
    """
    if not GNEWS_API_KEY:
        return "Error: GNEWS_API_KEY not set."
    try:
        articles = _gnews_search(keyword, lang=lang, from_date=from_date or None, max_results=max_results)
    except Exception as e:
        return f"GNews error: {e}"
    return _format_articles(articles, f"News · '{keyword}'")


@mcp.tool()
def get_weekly_trend_digest(topics: list[str] | None = None, days_back: int = 7) -> str:
    """
    Summarize notable news per topic over the last N days, from the indexed store.
    Sources may be in any language; ALWAYS answer the user in Vietnamese or English.

    Args:
        topics: Topics to include, e.g. ['fintech', 'ai_product'] (optional, all if empty)
        days_back: How many days back to cover (default 7)
    """
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days_back)).strftime("%Y-%m-%d")
    conn = sqlite3.connect(DB_PATH)
    q = "SELECT topic, title, source, article_url, published_at FROM news_articles WHERE COALESCE(published_at, crawled_at) >= ?"
    params: list = [cutoff]
    if topics:
        q += " AND topic IN (%s)" % ",".join("?" * len(topics))
        params += topics
    q += " ORDER BY topic, COALESCE(published_at, crawled_at) DESC"
    try:
        rows = conn.execute(q, params).fetchall()
    except sqlite3.OperationalError:
        rows = []
    conn.close()

    if not rows:
        return f"No indexed news in the last {days_back} days. Run trigger_crawl first."

    groups: dict = {}
    for topic, title, source, url, pub in rows:
        groups.setdefault(topic, []).append((title, source, url, pub))

    lines = [f"Weekly trend digest (last {days_back} days):"]
    for topic, items in groups.items():
        lines.append(f"\n## {topic} ({len(items)} articles)")
        for title, source, url, pub in items[:15]:
            d = (pub or "?")[:10]
            lines.append(f"- [{d}] {title} — {source or '?'}\n  {url}")
    return "\n".join(lines)


@mcp.tool()
def trigger_crawl(topics: list[str] | None = None) -> str:
    """
    Index the latest news into the vector DB. Meant to be called on a schedule by an
    external cron (e.g. a Render Cron Job), or manually. Runs in the background.

    Args:
        topics: Only crawl these topics (optional, all configured topics if empty)
    """
    if not GNEWS_API_KEY:
        return "Error: GNEWS_API_KEY not set."
    if not PINECONE_API_KEY or not OPENAI_API_KEY:
        return "Error: PINECONE_API_KEY or OPENAI_API_KEY not set."
    if _news_crawl_status["running"]:
        return f"News crawl already running: {_news_crawl_status['done']}/{_news_crawl_status['total']} topics."

    _init_news_db()
    _news_crawl_status.update({
        "running": True, "total": 0, "done": 0, "current": "starting...", "errors": [], "new_articles": 0,
    })
    threading.Thread(target=_crawl_news_background, args=(topics or [],), daemon=True).start()
    target = ", ".join(topics) if topics else "all topics"
    return f"News crawl started for {target}. Use get_crawl_status to check progress."


@mcp.tool()
def get_crawl_status() -> str:
    """Check the news crawler status and indexed article counts per topic."""
    lines = []
    if _news_crawl_status["running"]:
        lines.append(f"News crawl: IN PROGRESS ({_news_crawl_status['done']}/{_news_crawl_status['total']} topics)")
        lines.append(f"Current: {_news_crawl_status['current']}")
    elif _news_crawl_status["total"] > 0:
        lines.append(f"News crawl: COMPLETE ({_news_crawl_status['done']}/{_news_crawl_status['total']} topics)")
    else:
        lines.append("No news crawl has run yet.")
    lines.append(f"New articles this run: {_news_crawl_status['new_articles']}")

    if _news_crawl_status["errors"]:
        lines.append(f"Errors ({len(_news_crawl_status['errors'])}):")
        for e in _news_crawl_status["errors"][-5:]:
            lines.append(f"  - {e}")

    conn = sqlite3.connect(DB_PATH)
    try:
        rows = conn.execute("SELECT topic, COUNT(*) FROM news_articles GROUP BY topic ORDER BY topic").fetchall()
    except sqlite3.OperationalError:
        rows = []
    conn.close()
    if rows:
        lines.append("\nIndexed by topic:")
        for topic, count in rows:
            lines.append(f"  {topic}: {count}")
    return "\n".join(lines)


if __name__ == "__main__":
    _init_news_db()
    mcp.run(transport="stdio")
