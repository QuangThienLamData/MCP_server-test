"""Facebook Intelligence MCP — crawl page posts, user comments, ad creatives,
and public post mentions for competitor brands via RapidAPI Facebook Scraper3.

Endpoints used (facebook-scraper3.p.rapidapi.com):
  GET /search/posts    — public posts mentioning a brand (user sentiment)
  GET /profile/posts   — official page posts (brand content)
  GET /post/comments   — comments on a specific post
  GET /post            — single post details
  GET /page/details    — page metadata (name, id, image)
  GET /page/reviews    — page reviews
  GET /ads/search      — Meta Ad Library search
"""

import hashlib
import json
import logging
import os
import re
import sqlite3
import threading
import time
from datetime import datetime, timezone

import httpx
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

from rag_mcp import (
    DB_PATH, OPENAI_API_KEY, PINECONE_API_KEY,
    _bilingual_queries, _embed, _get_index, _get_openai,
)

load_dotenv()

logger = logging.getLogger(__name__)

RAPIDAPI_KEY = os.getenv("RAPIDAPI_FB_KEY", "")
RAPIDAPI_HOST = "facebook-scraper3.p.rapidapi.com"
RAPIDAPI_BASE = f"https://{RAPIDAPI_HOST}"
FB_NAMESPACE = "facebook"
FB_PAGES_FILE = os.path.join(os.path.dirname(__file__), "fb_pages.json")

mcp = FastMCP(
    name="Facebook Intelligence MCP Server",
    transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
)

_fb_crawl_status = {
    "running": False, "total": 0, "done": 0, "current": "",
    "errors": [], "new_posts": 0, "new_comments": 0,
}


# ---------------------------------------------------------------------------
# SQLite
# ---------------------------------------------------------------------------

def _init_fb_db():
    conn = sqlite3.connect(DB_PATH)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS fb_posts (
            post_id         TEXT PRIMARY KEY,
            brand           TEXT NOT NULL,
            source_type     TEXT NOT NULL DEFAULT 'page',
            url             TEXT,
            message         TEXT,
            timestamp       INTEGER,
            comments_count  INTEGER DEFAULT 0,
            reactions_count INTEGER DEFAULT 0,
            reshare_count   INTEGER DEFAULT 0,
            reactions_json  TEXT DEFAULT '{}',
            author_name     TEXT,
            author_id       TEXT,
            group_name      TEXT,
            group_id        TEXT,
            indexed_at      TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_fb_posts_brand ON fb_posts(brand);
        CREATE INDEX IF NOT EXISTS idx_fb_posts_ts ON fb_posts(timestamp);

        CREATE TABLE IF NOT EXISTS fb_comments (
            comment_id      TEXT PRIMARY KEY,
            post_id         TEXT NOT NULL,
            brand           TEXT NOT NULL,
            message         TEXT,
            author_name     TEXT,
            author_id       TEXT,
            created_time    INTEGER,
            reactions_count INTEGER DEFAULT 0,
            replies_count   INTEGER DEFAULT 0,
            depth           INTEGER DEFAULT 0,
            indexed_at      TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_fb_comments_post ON fb_comments(post_id);
        CREATE INDEX IF NOT EXISTS idx_fb_comments_brand ON fb_comments(brand);
    """)
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def _api_headers() -> dict:
    return {
        "x-rapidapi-key": RAPIDAPI_KEY,
        "x-rapidapi-host": RAPIDAPI_HOST,
    }


def _api_get(path: str, params: dict, timeout: int = 30, retries: int = 3) -> dict:
    """Make a GET request to the RapidAPI Facebook Scraper with retry on 429."""
    with httpx.Client(timeout=timeout) as c:
        for attempt in range(retries):
            r = c.get(f"{RAPIDAPI_BASE}{path}", params=params, headers=_api_headers())
            if r.status_code == 429:
                wait = 2 ** (attempt + 1)
                logger.info(f"Rate limited on {path}, waiting {wait}s...")
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()
        return {"results": [], "ads": []}  # exhausted retries


def _load_fb_pages() -> dict:
    try:
        with open(FB_PAGES_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning(f"fb_pages.json not found at {FB_PAGES_FILE}")
        return {}


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def _search_public_posts(query: str, limit: int = 20) -> list[dict]:
    """Search public Facebook posts mentioning a keyword."""
    all_posts: list[dict] = []
    cursor = None
    fetched = 0
    while fetched < limit:
        params: dict = {"query": query, "limit": str(min(limit - fetched, 20))}
        if cursor:
            params["cursor"] = cursor
        data = _api_get("/search/posts", params)
        posts = data.get("results", [])
        if not posts:
            break
        all_posts.extend(posts)
        fetched += len(posts)
        cursor = data.get("cursor")
        if not cursor:
            break
        time.sleep(0.5)
    return all_posts


def _get_page_posts(page_id: str, limit: int = 10) -> list[dict]:
    """Get posts from an official brand page."""
    all_posts: list[dict] = []
    cursor = None
    fetched = 0
    while fetched < limit:
        params: dict = {"profile_id": page_id, "limit": str(min(limit - fetched, 10))}
        if cursor:
            params["cursor"] = cursor
        data = _api_get("/profile/posts", params)
        posts = data.get("results", [])
        if not posts:
            break
        all_posts.extend(posts)
        fetched += len(posts)
        cursor = data.get("cursor")
        if not cursor:
            break
        time.sleep(0.5)
    return all_posts


def _get_post_comments(post_id: str, limit: int = 50) -> list[dict]:
    """Get comments on a specific post."""
    all_comments: list[dict] = []
    cursor = None
    fetched = 0
    while fetched < limit:
        params: dict = {"post_id": post_id, "limit": str(min(limit - fetched, 25))}
        if cursor:
            params["cursor"] = cursor
        data = _api_get("/post/comments", params)
        comments = data.get("results", [])
        if not comments:
            break
        all_comments.extend(comments)
        fetched += len(comments)
        cursor = data.get("cursor")
        if not cursor:
            break
        time.sleep(0.5)
    return all_comments


def _search_ads(query: str, country: str = "VN", active_status: str = "ALL") -> list[dict]:
    """Search Meta Ad Library."""
    data = _api_get("/ads/search", {
        "query": query, "country": country, "active_status": active_status,
    })
    return data.get("ads", [])


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------

def _post_vector_id(post_id: str) -> str:
    return f"fb_{hashlib.md5(post_id.encode()).hexdigest()}"


def _comment_vector_id(comment_id: str) -> str:
    return f"fbc_{hashlib.md5(comment_id.encode()).hexdigest()}"


def _ts_to_iso(ts: int | None) -> str:
    if not ts:
        return ""
    try:
        return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
    except Exception:
        return ""


def _index_posts(posts: list[dict], brand: str, source_type: str = "search") -> int:
    """Store posts in SQLite + embed & upsert into Pinecone. Returns count indexed."""
    if not posts or not PINECONE_API_KEY or not OPENAI_API_KEY:
        return 0

    _init_fb_db()
    conn = sqlite3.connect(DB_PATH)
    now = datetime.now(timezone.utc).isoformat()

    texts_to_embed: list[str] = []
    metas: list[dict] = []
    vec_ids: list[str] = []
    new_count = 0

    for p in posts:
        pid = str(p.get("post_id", ""))
        msg = (p.get("message") or "").strip()
        if not pid or len(msg) < 10:
            continue

        # Dedup in SQLite
        existing = conn.execute("SELECT post_id FROM fb_posts WHERE post_id = ?", (pid,)).fetchone()
        if existing:
            continue

        ts = p.get("timestamp")
        author = p.get("author") or {}
        group = p.get("associated_group") or {}
        reactions = p.get("reactions") or {}

        conn.execute(
            "INSERT OR IGNORE INTO fb_posts "
            "(post_id,brand,source_type,url,message,timestamp,comments_count,"
            "reactions_count,reshare_count,reactions_json,author_name,author_id,"
            "group_name,group_id,indexed_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (pid, brand, source_type, p.get("url", ""), msg, ts,
             p.get("comments_count", 0), p.get("reactions_count", 0),
             p.get("reshare_count", 0), json.dumps(reactions),
             author.get("name", ""), author.get("id", ""),
             group.get("name", ""), group.get("group_id", ""), now),
        )

        texts_to_embed.append(msg[:2000])
        metas.append({
            "source_type": f"facebook_{source_type}",
            "brand": brand,
            "post_id": pid,
            "url": p.get("url", ""),
            "author_name": author.get("name", ""),
            "group_name": group.get("name", ""),
            "reactions_count": p.get("reactions_count", 0),
            "comments_count": p.get("comments_count", 0),
            "published_at": _ts_to_iso(ts),
            "text": msg[:1500],
        })
        vec_ids.append(_post_vector_id(pid))
        new_count += 1

    conn.commit()
    conn.close()

    if texts_to_embed:
        try:
            embeddings = _embed(texts_to_embed)
            vectors = [
                {"id": vid, "values": emb, "metadata": meta}
                for vid, emb, meta in zip(vec_ids, embeddings, metas)
            ]
            index = _get_index()
            for b in range(0, len(vectors), 100):
                index.upsert(vectors=vectors[b:b + 100], namespace=FB_NAMESPACE)
        except Exception as e:
            logger.error(f"Facebook index embed/upsert failed: {e}")

    return new_count


def _index_comments(comments: list[dict], brand: str, post_id: str) -> int:
    """Store comments in SQLite + embed & upsert into Pinecone."""
    if not comments or not PINECONE_API_KEY or not OPENAI_API_KEY:
        return 0

    _init_fb_db()
    conn = sqlite3.connect(DB_PATH)
    now = datetime.now(timezone.utc).isoformat()

    texts: list[str] = []
    metas: list[dict] = []
    vec_ids: list[str] = []
    new_count = 0

    for c in comments:
        cid = c.get("comment_id") or c.get("legacy_comment_id", "")
        msg = (c.get("message") or "").strip()
        if not cid or len(msg) < 5:
            continue

        existing = conn.execute("SELECT comment_id FROM fb_comments WHERE comment_id = ?", (cid,)).fetchone()
        if existing:
            continue

        author = c.get("author", {})
        ct = c.get("created_time")

        conn.execute(
            "INSERT OR IGNORE INTO fb_comments "
            "(comment_id,post_id,brand,message,author_name,author_id,"
            "created_time,reactions_count,replies_count,depth,indexed_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (cid, post_id, brand, msg, author.get("name", ""), author.get("id", ""),
             ct, int(c.get("reactions_count") or 0), c.get("replies_count", 0),
             c.get("depth", 0), now),
        )

        texts.append(msg[:2000])
        metas.append({
            "source_type": "facebook_comment",
            "brand": brand,
            "post_id": post_id,
            "comment_id": cid,
            "author_name": author.get("name", ""),
            "reactions_count": int(c.get("reactions_count") or 0),
            "published_at": _ts_to_iso(ct),
            "text": msg[:1500],
        })
        vec_ids.append(_comment_vector_id(cid))
        new_count += 1

    conn.commit()
    conn.close()

    if texts:
        try:
            embeddings = _embed(texts)
            vectors = [
                {"id": vid, "values": emb, "metadata": meta}
                for vid, emb, meta in zip(vec_ids, embeddings, metas)
            ]
            index = _get_index()
            for b in range(0, len(vectors), 100):
                index.upsert(vectors=vectors[b:b + 100], namespace=FB_NAMESPACE)
        except Exception as e:
            logger.error(f"Facebook comment embed/upsert failed: {e}")

    return new_count


# ---------------------------------------------------------------------------
# Background crawl
# ---------------------------------------------------------------------------

def _crawl_background(brands: list[str] | None = None):
    """Background worker: crawl page posts + search mentions + fetch comments."""
    try:
        pages = _load_fb_pages()
        if not pages:
            _fb_crawl_status["errors"].append("fb_pages.json not found or empty")
            return

        targets = brands if brands else list(pages.keys())
        _fb_crawl_status["total"] = len(targets)

        for i, brand in enumerate(targets):
            _fb_crawl_status["done"] = i
            _fb_crawl_status["current"] = brand
            cfg = pages.get(brand, {})
            page_id = cfg.get("page_id", "")
            keywords = cfg.get("search_keywords", [brand])

            logger.info(f"Facebook crawl ({i + 1}/{len(targets)}): {brand}")

            # 1. Page posts
            if page_id:
                try:
                    posts = _get_page_posts(page_id, limit=10)
                    n = _index_posts(posts, brand, source_type="page")
                    _fb_crawl_status["new_posts"] += n
                    # Fetch comments on top-engagement page posts (limit to avoid rate limits)
                    top_posts = sorted(
                        [p for p in posts if p.get("comments_count", 0) > 0],
                        key=lambda p: p.get("comments_count", 0), reverse=True,
                    )[:3]
                    for p in top_posts:
                        try:
                            comments = _get_post_comments(str(p["post_id"]), limit=30)
                            nc = _index_comments(comments, brand, str(p["post_id"]))
                            _fb_crawl_status["new_comments"] += nc
                        except Exception as e:
                            logger.warning(f"Comments failed for {p['post_id']}: {e}")
                        time.sleep(2)
                except Exception as e:
                    _fb_crawl_status["errors"].append(f"{brand} page: {e}")

            # 2. Public search mentions
            for kw in keywords[:3]:
                try:
                    posts = _search_public_posts(kw, limit=20)
                    n = _index_posts(posts, brand, source_type="search")
                    _fb_crawl_status["new_posts"] += n
                    # Comments on top-engagement search posts only
                    top_search = sorted(
                        [p for p in posts if p.get("comments_count", 0) >= 5],
                        key=lambda p: p.get("comments_count", 0), reverse=True,
                    )[:3]
                    for p in top_search:
                        try:
                            comments = _get_post_comments(str(p["post_id"]), limit=20)
                            nc = _index_comments(comments, brand, str(p["post_id"]))
                            _fb_crawl_status["new_comments"] += nc
                        except Exception:
                            pass
                        time.sleep(2)
                except Exception as e:
                    _fb_crawl_status["errors"].append(f"{brand} search '{kw}': {e}")
                time.sleep(1)

        _fb_crawl_status["done"] = len(targets)
        _fb_crawl_status["current"] = ""
        logger.info(
            f"Facebook crawl done: {_fb_crawl_status['new_posts']} posts, "
            f"{_fb_crawl_status['new_comments']} comments"
        )
    except Exception as e:
        logger.exception("Facebook crawl failed")
        _fb_crawl_status["errors"].append(str(e))
    finally:
        _fb_crawl_status["running"] = False


# ---------------------------------------------------------------------------
# LLM analysis
# ---------------------------------------------------------------------------

def _sentiment_analysis(brand: str, comments: list[dict]) -> str:
    """GPT-4o-mini sentiment summary of Facebook comments in Vietnamese."""
    if not OPENAI_API_KEY or not comments:
        return ""
    sample = "\n".join(
        f"- [{c.get('author_name', '?')}] {c.get('message', '')[:200]}"
        for c in comments[:50]
    )
    prompt = (
        f"You are a social media analyst for the Vietnam market.\n"
        f"Analyze these Facebook comments about {brand}.\n\n"
        f"Comments:\n{sample}\n\n"
        f"Provide in Vietnamese:\n"
        f"1. Tổng quan cảm xúc (tích cực / tiêu cực / trung lập — tỷ lệ ước tính)\n"
        f"2. Vấn đề chính được nhắc đến (top 3-5 themes)\n"
        f"3. Phản hồi tiêu cực nổi bật (2-3 ví dụ cụ thể)\n"
        f"4. Phản hồi tích cực nổi bật (2-3 ví dụ cụ thể)\n"
        f"5. Đề xuất cải thiện cho team Product/Marketing\n"
    )
    try:
        resp = _get_openai().chat.completions.create(
            model="gpt-4o-mini", temperature=0.3,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logger.warning(f"Sentiment LLM failed: {e}")
        return ""


# ---------------------------------------------------------------------------
# Tool functions (exported, registered by research_mcp.py)
# ---------------------------------------------------------------------------

def search_facebook_mentions(
    query: str, brand: str = "", top_k: int = 15,
) -> str:
    """
    Search Facebook posts and comments about a brand or topic. Uses bilingual
    semantic search (EN + VI) on indexed Facebook content. If nothing relevant
    is in the DB, falls back to a live Facebook search via RapidAPI.
    ALWAYS answer in Vietnamese or English.

    Args:
        query: What to search for, e.g. "MoMo lỗi thanh toán", "ZaloPay khuyến mãi"
        brand: Filter by brand: MoMo, ZaloPay, Grab, Shopee, etc. (optional)
        top_k: Number of results (default 15)
    """
    if not PINECONE_API_KEY or not OPENAI_API_KEY:
        return "Error: PINECONE_API_KEY or OPENAI_API_KEY not set."

    variants = _bilingual_queries(query)
    try:
        qvecs = _embed(variants)
    except Exception as e:
        return f"Embedding error: {e}"

    filters: dict = {}
    if brand:
        filters["brand"] = {"$eq": brand}

    index = _get_index()
    merged: dict = {}
    for qvec in qvecs:
        res = index.query(
            vector=qvec, top_k=top_k, include_metadata=True,
            namespace=FB_NAMESPACE, filter=filters or None,
        )
        for m in res.matches:
            if m.id not in merged or m.score > merged[m.id].score:
                merged[m.id] = m
    matches = sorted(merged.values(), key=lambda m: m.score, reverse=True)[:top_k]

    best = matches[0].score if matches else 0.0

    # Fallback: live search if DB has weak results
    if best < 0.35 and RAPIDAPI_KEY:
        try:
            live_posts = _search_public_posts(query, limit=10)
            if live_posts:
                n = _index_posts(live_posts, brand or "unknown", source_type="search")
                # Re-query after indexing
                if n > 0:
                    merged2: dict = {}
                    for qvec in qvecs:
                        res = index.query(
                            vector=qvec, top_k=top_k, include_metadata=True,
                            namespace=FB_NAMESPACE, filter=filters or None,
                        )
                        for m in res.matches:
                            if m.id not in merged2 or m.score > merged2[m.id].score:
                                merged2[m.id] = m
                    matches = sorted(merged2.values(), key=lambda m: m.score, reverse=True)[:top_k]
        except Exception as e:
            logger.warning(f"Facebook live fallback failed: {e}")

    if not matches:
        if not RAPIDAPI_KEY:
            return "No results. Set RAPIDAPI_FB_KEY to enable Facebook search."
        return "No Facebook mentions found for this query."

    out = [f"Found {len(matches)} Facebook results for: '{query}'\n"]
    for i, m in enumerate(matches):
        md = m.metadata
        st = md.get("source_type", "")
        label = "Comment" if "comment" in st else "Post"
        out.append(
            f"\n--- {label} {i + 1} (score: {m.score:.3f}) ---\n"
            f"Brand: {md.get('brand', '?')}\n"
            f"Author: {md.get('author_name', '?')}\n"
            f"Date: {md.get('published_at', '?')}\n"
            f"URL: {md.get('url', '')}\n"
            f"Reactions: {md.get('reactions_count', 0)} | Comments: {md.get('comments_count', 0)}\n"
            f"Group: {md.get('group_name', '')}\n"
            f"{md.get('text', '')}\n"
        )
    return "".join(out)


def get_facebook_post_comments(post_url_or_id: str, brand: str = "", limit: int = 30) -> str:
    """
    Fetch comments on a specific Facebook post. Indexes them for future search.
    ALWAYS answer in Vietnamese or English.

    Args:
        post_url_or_id: Facebook post ID or URL
        brand: Brand name to tag these comments with (optional)
        limit: Max comments to fetch (default 30)
    """
    if not RAPIDAPI_KEY:
        return "Error: RAPIDAPI_FB_KEY not set."

    # Extract post_id from URL if needed
    post_id = post_url_or_id.strip()
    m = re.search(r"/posts/(\d+)", post_id)
    if m:
        post_id = m.group(1)
    m = re.search(r"story_fbid=(\d+)", post_id)
    if m:
        post_id = m.group(1)

    try:
        comments = _get_post_comments(post_id, limit=limit)
    except Exception as e:
        return f"Failed to fetch comments: {e}"

    if not comments:
        return f"No comments found for post {post_id}."

    # Index
    if brand:
        _index_comments(comments, brand, post_id)

    lines = [f"Fetched {len(comments)} comments on post {post_id}:\n"]
    for i, c in enumerate(comments, 1):
        author = c.get("author", {}).get("name", "?")
        msg = c.get("message", "")[:200]
        reactions = c.get("reactions_count", 0)
        replies = c.get("replies_count", 0)
        lines.append(
            f"  {i}. [{author}] {msg}\n"
            f"     Reactions: {reactions} | Replies: {replies}\n"
        )
    return "\n".join(lines)


def get_brand_page_posts(brand: str, limit: int = 10) -> str:
    """
    Fetch the latest posts from a brand's official Facebook page.
    ALWAYS answer in Vietnamese or English.

    Args:
        brand: Brand name as configured in fb_pages.json: MoMo, ZaloPay, Grab, Shopee, etc.
        limit: Number of posts to fetch (default 10)
    """
    if not RAPIDAPI_KEY:
        return "Error: RAPIDAPI_FB_KEY not set."

    pages = _load_fb_pages()
    cfg = pages.get(brand)
    if not cfg:
        available = ", ".join(pages.keys())
        return f"Brand '{brand}' not found. Available: {available}"

    page_id = cfg.get("page_id", "")
    if not page_id:
        return f"No page_id configured for {brand}."

    try:
        posts = _get_page_posts(page_id, limit=limit)
    except Exception as e:
        return f"Failed to fetch posts: {e}"

    if not posts:
        return f"No posts found for {brand}."

    _index_posts(posts, brand, source_type="page")

    lines = [f"Latest {len(posts)} posts from {brand} ({cfg.get('url', '')}):\n"]
    for i, p in enumerate(posts, 1):
        msg = (p.get("message") or "")[:200]
        ts = _ts_to_iso(p.get("timestamp"))
        reactions = p.get("reactions", {})
        lines.append(
            f"\n[{i}] {ts} — {msg}\n"
            f"    Reactions: {p.get('reactions_count', 0)} "
            f"(like={reactions.get('like', 0)}, love={reactions.get('love', 0)}, "
            f"angry={reactions.get('angry', 0)}, haha={reactions.get('haha', 0)})\n"
            f"    Comments: {p.get('comments_count', 0)} | Shares: {p.get('reshare_count', 0)}\n"
            f"    URL: {p.get('url', '')}"
        )
    return "\n".join(lines)


def get_facebook_sentiment(brand: str, days_back: int = 30) -> str:
    """
    Analyze user sentiment about a brand from indexed Facebook comments and posts.
    Uses AI to summarize themes, complaints, and positive feedback in Vietnamese.
    ALWAYS answer in Vietnamese or English.

    Args:
        brand: Brand name: MoMo, ZaloPay, Grab, Shopee, etc.
        days_back: How many days of data to analyze (default 30)
    """
    _init_fb_db()
    conn = sqlite3.connect(DB_PATH)

    # Get comments
    try:
        comment_rows = conn.execute(
            "SELECT message, author_name, reactions_count FROM fb_comments "
            "WHERE brand = ? ORDER BY created_time DESC LIMIT 100",
            (brand,),
        ).fetchall()
    except sqlite3.OperationalError:
        comment_rows = []

    # Get search posts (user mentions)
    try:
        post_rows = conn.execute(
            "SELECT message, author_name, reactions_count FROM fb_posts "
            "WHERE brand = ? AND source_type = 'search' "
            "ORDER BY timestamp DESC LIMIT 50",
            (brand,),
        ).fetchall()
    except sqlite3.OperationalError:
        post_rows = []

    conn.close()

    all_items = [
        {"message": msg, "author_name": author, "reactions_count": r}
        for msg, author, r in (comment_rows + post_rows)
        if msg and len(msg) > 10
    ]

    if not all_items:
        return f"No Facebook data for {brand}. Run crawl_facebook first."

    # Stats
    total = len(all_items)
    head = f"Facebook Sentiment for {brand}: {total} posts/comments analyzed\n"

    # LLM analysis
    summary = _sentiment_analysis(brand, all_items)
    if summary:
        return f"{head}\n{summary}"
    return f"{head}\n({total} items available but LLM analysis unavailable — set OPENAI_API_KEY.)"


def search_facebook_ads(
    query: str, country: str = "VN", active_status: str = "ALL",
) -> str:
    """
    Search Meta Ad Library for competitor ads. Shows ad creatives, page info,
    and targeting. Useful for competitive ad intelligence.
    ALWAYS answer in Vietnamese or English.

    Args:
        query: Brand or keyword to search ads for, e.g. "ZaloPay", "e-wallet"
        country: Country code (default VN)
        active_status: ALL, ACTIVE, or INACTIVE (default ALL)
    """
    if not RAPIDAPI_KEY:
        return "Error: RAPIDAPI_FB_KEY not set."

    try:
        ads = _search_ads(query, country=country, active_status=active_status)
    except Exception as e:
        return f"Ad search failed: {e}"

    if not ads:
        return f"No ads found for '{query}' in {country}."

    lines = [f"Found {len(ads)} ads for '{query}' (country={country}):\n"]
    for i, ad in enumerate(ads[:15], 1):
        snap = ad.get("snapshot", {})
        page_name = snap.get("page_name", "?")
        is_active = ad.get("is_active", False)
        cta = snap.get("cta_text", "")

        # Get ad body from cards or caption
        body = ""
        cards = snap.get("cards") or []
        if cards:
            body = (cards[0].get("body") or cards[0].get("title") or "")[:200]
        if not body:
            body = (snap.get("caption") or "")[:200]

        lines.append(
            f"\n[{i}] {'ACTIVE' if is_active else 'INACTIVE'} — {page_name}\n"
            f"    {body}\n"
            f"    CTA: {cta}\n"
            f"    Ad ID: {ad.get('ad_archive_id', '?')}"
        )
    return "\n".join(lines)


def crawl_facebook(brands: list[str] | None = None) -> str:
    """
    Crawl Facebook page posts, public mentions, and comments for configured brands.
    Runs in background. Indexes into Pinecone + SQLite for semantic search.
    ALWAYS answer in Vietnamese or English.

    Args:
        brands: Only crawl these brands (optional, all if empty).
                Available: MoMo, ZaloPay, Grab, Shopee, ShopeePay, VNPay, ViettelMoney, Techcombank, VPBank
    """
    if not RAPIDAPI_KEY:
        return "Error: RAPIDAPI_FB_KEY not set."
    if not PINECONE_API_KEY or not OPENAI_API_KEY:
        return "Error: PINECONE_API_KEY or OPENAI_API_KEY not set."
    if _fb_crawl_status["running"]:
        return (
            f"Facebook crawl already running: {_fb_crawl_status['done']}/"
            f"{_fb_crawl_status['total']} brands. Current: {_fb_crawl_status['current']}"
        )

    _init_fb_db()
    _fb_crawl_status.update({
        "running": True, "total": 0, "done": 0, "current": "starting...",
        "errors": [], "new_posts": 0, "new_comments": 0,
    })
    threading.Thread(target=_crawl_background, args=(brands,), daemon=True).start()
    target = ", ".join(brands) if brands else "all brands"
    return f"Facebook crawl started for {target}. Use get_facebook_status to check progress."


def get_facebook_status() -> str:
    """Check the status of Facebook crawl and indexed data.
    ALWAYS answer in Vietnamese or English."""
    lines = ["== Facebook Intelligence =="]

    if _fb_crawl_status["running"]:
        lines.append(
            f"Crawl: IN PROGRESS ({_fb_crawl_status['done']}/"
            f"{_fb_crawl_status['total']} brands)"
        )
        lines.append(f"Current: {_fb_crawl_status['current']}")
    elif _fb_crawl_status["total"] > 0:
        lines.append(
            f"Crawl: COMPLETE ({_fb_crawl_status['done']}/"
            f"{_fb_crawl_status['total']} brands)"
        )
    else:
        lines.append("No Facebook crawl has run yet.")

    lines.append(
        f"New posts: {_fb_crawl_status['new_posts']} | "
        f"New comments: {_fb_crawl_status['new_comments']}"
    )

    if _fb_crawl_status["errors"]:
        for e in _fb_crawl_status["errors"][-5:]:
            lines.append(f"  ! {e[:120]}")

    _init_fb_db()
    conn = sqlite3.connect(DB_PATH)
    try:
        post_stats = conn.execute(
            "SELECT brand, source_type, COUNT(*) FROM fb_posts "
            "GROUP BY brand, source_type ORDER BY brand"
        ).fetchall()
    except sqlite3.OperationalError:
        post_stats = []
    try:
        comment_stats = conn.execute(
            "SELECT brand, COUNT(*) FROM fb_comments GROUP BY brand ORDER BY brand"
        ).fetchall()
    except sqlite3.OperationalError:
        comment_stats = []
    conn.close()

    if post_stats:
        lines.append("\nIndexed posts:")
        for brand, stype, count in post_stats:
            lines.append(f"  {brand}/{stype}: {count}")
    if comment_stats:
        lines.append("\nIndexed comments:")
        for brand, count in comment_stats:
            lines.append(f"  {brand}: {count}")

    return "\n".join(lines)


if __name__ == "__main__":
    _init_fb_db()
    mcp.run(transport="stdio")
