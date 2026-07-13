import hashlib
import json
import logging
import os
import re
import sqlite3
import threading
import time
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from urllib.parse import urljoin, urlparse

import feedparser
import httpx
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

logger = logging.getLogger(__name__)

# --- Paths ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(PROJECT_ROOT, "config")

# --- Config ---
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "competitor-content")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
TAVILY_URL = "https://api.tavily.com/search"
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536
RENDER_SELF_URL = os.getenv("RENDER_EXTERNAL_URL", "")
COMPETITORS_CONFIG = os.path.join(CONFIG_DIR, "competitors.json")
DB_PATH = os.getenv("DB_PATH", os.path.join(PROJECT_ROOT, "competitor_intel.db"))

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
CRAWL_DELAY = 2  # seconds between requests
FALLBACK_SCORE_THRESHOLD = 0.35  # competitor search below this falls back to Tavily web search
TAVILY_MIN_SCORE = 0.5  # drop low-relevance Tavily results before indexing/returning

mcp = FastMCP(
    name="Competitor Intelligence MCP Server",
    transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
)

# --- Lazy-init clients ---
_pc_client = None
_pc_index = None
_oai_client = None


def _get_index():
    global _pc_client, _pc_index
    if _pc_index is None:
        _pc_client = Pinecone(api_key=PINECONE_API_KEY)
        existing = [idx.name for idx in _pc_client.list_indexes()]
        if PINECONE_INDEX_NAME not in existing:
            _pc_client.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=EMBEDDING_DIMENSION,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            time.sleep(5)
        _pc_index = _pc_client.Index(PINECONE_INDEX_NAME)
    return _pc_index


def _get_openai():
    global _oai_client
    if _oai_client is None:
        _oai_client = OpenAI(api_key=OPENAI_API_KEY)
    return _oai_client


# --- SQLite ---
def _init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS crawl_sources (
            id TEXT PRIMARY KEY,
            competitor_name TEXT NOT NULL,
            source_url TEXT NOT NULL,
            source_type TEXT DEFAULT 'blog',
            js_render INTEGER DEFAULT 0,
            last_crawled_at TEXT,
            enabled INTEGER DEFAULT 1
        );
        CREATE TABLE IF NOT EXISTS crawled_pages (
            id TEXT PRIMARY KEY,
            source_id TEXT REFERENCES crawl_sources(id),
            url TEXT NOT NULL UNIQUE,
            title TEXT,
            content_hash TEXT,
            published_at TEXT,
            first_seen_at TEXT NOT NULL,
            last_updated_at TEXT NOT NULL
        );
    """)
    conn.commit()

    # Initialize knowledge graph tables
    from kg_extract import _init_kg_db
    _init_kg_db()
    conn.close()


def _load_competitors():
    if not os.path.exists(COMPETITORS_CONFIG):
        logger.warning(f"Config not found: {COMPETITORS_CONFIG}")
        return
    with open(COMPETITORS_CONFIG) as f:
        competitors = json.load(f)
    conn = sqlite3.connect(DB_PATH)
    for comp in competitors:
        for source in comp.get("sources", []):
            stype = source.get("type", "blog")
            js = 1 if source.get("js") else 0
            urls = source["url"] if isinstance(source["url"], list) else [source["url"]]
            for url in urls:
                sid = hashlib.md5(url.encode()).hexdigest()
                conn.execute(
                    "INSERT OR IGNORE INTO crawl_sources (id, competitor_name, source_url, source_type, js_render) VALUES (?, ?, ?, ?, ?)",
                    (sid, comp["name"], url, stype, js),
                )
    conn.commit()
    conn.close()


# --- Text helpers ---
def _chunk_text(text: str) -> list[str]:
    chunks, start = [], 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - CHUNK_OVERLAP
    return chunks


def _embed(texts: list[str]) -> list[list[float]]:
    resp = _get_openai().embeddings.create(input=texts, model=EMBEDDING_MODEL)
    return [d.embedding for d in resp.data]


def _bilingual_queries(query: str) -> list[str]:
    """Return the query plus its English & Vietnamese translations so a query in one
    language also retrieves content stored in the other (bridges the weak cross-lingual
    behaviour of text-embedding-3-small). Falls back to the original query on any error.
    Shared by the RAG (Sub A) and News (Sub D) servers."""
    variants = [query.strip()]
    try:
        resp = _get_openai().chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": (
                    "Translate the user's short search query. Reply with JSON only: "
                    '{"en": "<English>", "vi": "<Vietnamese>"}. Keep proper nouns unchanged.'
                )},
                {"role": "user", "content": query},
            ],
        )
        data = json.loads(resp.choices[0].message.content)
        for v in (data.get("en"), data.get("vi")):
            if v and v.strip() and v.strip().lower() not in {x.lower() for x in variants}:
                variants.append(v.strip())
    except Exception as e:
        logger.warning(f"Query translation failed, using original only: {e}")
    return variants


# Vietnamese-specific letters (dot-below, horn, breve, đ) — not shared with es/pt/fr,
# so this reliably distinguishes Vietnamese text from other accented Latin scripts.
# Shared by the News (Sub D) and Reviews (Sub B) servers.
_VI_RE = re.compile(r"[đĐăĂơƠưƯạảấầẩẫậắằẳẵặẹẻẽếềểễệịỉọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ]")


def _detect_lang(text: str) -> str:
    return "vi" if _VI_RE.search(text or "") else "en"


# --- Tavily web-search fallback ---
def _tavily_search(query: str, max_results: int = 5, min_score: float = TAVILY_MIN_SCORE) -> list[dict]:
    payload = {
        "api_key": TAVILY_API_KEY,
        "query": query,
        "search_depth": "advanced",
        "max_results": max_results,
        "include_raw_content": False,
    }
    with httpx.Client(timeout=30) as c:
        r = c.post(TAVILY_URL, json=payload)
        r.raise_for_status()
        results = r.json().get("results", [])
    # Drop low-relevance hits (Tavily can rank off-topic pages high for vague queries).
    return [x for x in results if x.get("score", 0) >= min_score]


def _index_tavily_results(results: list[dict], competitor: str) -> int:
    """Embed Tavily results into Pinecone (default namespace, same store as crawled pages)
    so they enrich future searches. Deterministic ID per URL → re-runs overwrite, not dup."""
    if not results:
        return 0
    now = datetime.now(timezone.utc).isoformat()
    texts, metas = [], []
    for res in results:
        url = res.get("url", "")
        title = res.get("title", "") or ""
        content = res.get("content", "") or ""
        text = f"{title}. {content}".strip()
        if not url or len(text) < 40:
            continue
        texts.append(text)
        metas.append({
            "competitor_name": competitor or "unknown",
            "url": url,
            "title": title,
            "source_type": "web_search",
            "published_at": (res.get("published_date") or "")[:10],
            "crawled_at": now,
            "chunk_index": 0,
            "text": text,
        })
    if not texts:
        return 0
    embeddings = _embed(texts)
    vectors = [
        {"id": hashlib.md5(md["url"].encode()).hexdigest() + "_web", "values": emb, "metadata": md}
        for emb, md in zip(embeddings, metas)
    ]
    index = _get_index()
    for b in range(0, len(vectors), 100):
        index.upsert(vectors=vectors[b:b + 100])
    return len(vectors)


def _format_web_results(query: str, results: list[dict]) -> str:
    out = [f"Found {len(results)} web results for: '{query}' (not in DB yet — via web search)\n"]
    for i, res in enumerate(results):
        out.append(
            f"\n--- Web Result {i + 1} ---\n"
            f"Title: {res.get('title', '?')}\n"
            f"URL: {res.get('url', '')}\n"
            f"Content: {res.get('content', '')}\n"
        )
    return "".join(out)


# --- Crawling ---
_HTTP_HEADERS = {"User-Agent": "CompetitorIntelBot/1.0 (research)"}


def _fetch(url: str) -> str | None:
    try:
        with httpx.Client(timeout=30, follow_redirects=True) as c:
            r = c.get(url, headers=_HTTP_HEADERS)
            r.raise_for_status()
            return r.text
    except Exception as e:
        logger.error(f"Fetch failed {url}: {e}")
        return None


def _fetch_rendered(url: str) -> tuple[str, str] | None:
    """Use Jina Reader to fetch JS-rendered page content."""
    try:
        with httpx.Client(timeout=60, follow_redirects=True) as c:
            r = c.get(
                f"https://r.jina.ai/{url}",
                headers={"Accept": "text/plain", "X-Return-Format": "text"},
            )
            r.raise_for_status()
            text = r.text.strip()
            if not text:
                return None
            lines = [l.strip() for l in text.splitlines() if l.strip()]
            title = lines[0] if lines else ""
            return title, "\n".join(lines)
    except Exception as e:
        logger.error(f"Jina render failed {url}: {e}")
        return None


def _extract_content(html: str) -> tuple[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
        tag.decompose()
    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else ""
    body = (
        soup.find("article")
        or soup.find("main")
        or soup.find(class_=["post-content", "article-content", "entry-content", "blog-content"])
    )
    text = (body or soup).get_text(separator="\n", strip=True)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return title, "\n".join(lines)


def _extract_links(html: str, base_url: str) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    base_domain = urlparse(base_url).netloc
    seen, links = set(), []
    skip = {"#", "javascript:", "mailto:", "/tag/", "/category/", "/author/", "/page/", "/search"}
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.startswith("/"):
            href = urljoin(base_url, href)
        elif not href.startswith("http"):
            continue
        if href in seen or urlparse(href).netloc != base_domain:
            continue
        if any(s in href.lower() for s in skip):
            continue
        seen.add(href)
        text = a.get_text(strip=True)
        if text and len(text) > 10:
            links.append({"url": href, "title": text})
    return links


def _parse_sitemap(base_url: str) -> list[dict]:
    """Parse sitemap.xml from a site to discover article URLs."""
    parsed = urlparse(base_url)
    sitemap_url = f"{parsed.scheme}://{parsed.netloc}/sitemap.xml"
    try:
        with httpx.Client(timeout=30, follow_redirects=True) as c:
            r = c.get(sitemap_url, headers=_HTTP_HEADERS)
            r.raise_for_status()
        root = ET.fromstring(r.text)
        ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
        # Get URLs that are sub-paths of the source URL
        base_path = parsed.path.rstrip("/")
        articles = []
        for loc in root.findall(".//sm:url", ns):
            url_el = loc.find("sm:loc", ns)
            if url_el is None:
                continue
            url = url_el.text.strip()
            url_path = urlparse(url).path.rstrip("/")
            # Only include URLs that are deeper than the base path
            if url_path.startswith(base_path) and url_path != base_path and url_path.count("/") > base_path.count("/"):
                slug = url_path.rsplit("/", 1)[-1]
                articles.append({"url": url, "title": slug})
        return articles
    except Exception as e:
        logger.warning(f"Sitemap parse failed for {sitemap_url}: {e}")
        return []


def _parse_rss(url: str) -> list[dict]:
    feed = feedparser.parse(url)
    entries = []
    for e in feed.entries:
        pub = None
        if hasattr(e, "published_parsed") and e.published_parsed:
            pub = datetime(*e.published_parsed[:6]).isoformat()
        entries.append({"url": e.link, "title": e.get("title", ""), "published_at": pub})
    return entries


# --- Crawl engine ---
_crawl_status = {
    "running": False, "total": 0, "done": 0, "current": "",
    "errors": [], "new_articles": 0, "updated_articles": 0,
}


def _keep_alive():
    url = RENDER_SELF_URL or "http://localhost:10000"
    while _crawl_status["running"]:
        try:
            urllib.request.urlopen(url, timeout=10)
        except Exception:
            pass
        time.sleep(300)


def _crawl_source(source_id: str, name: str, source_url: str, stype: str, js_render: bool = False):
    conn = sqlite3.connect(DB_PATH)
    index = _get_index()

    articles = _parse_rss(source_url) if stype == "rss" else []
    if not articles:
        if js_render:
            # Try sitemap first to discover individual article URLs
            articles = _parse_sitemap(source_url)
            if not articles:
                # Fallback: treat the whole page as a single article
                result = _fetch_rendered(source_url)
                if result:
                    articles = [{"url": source_url, "title": result[0]}]
        else:
            html = _fetch(source_url)
            if html:
                articles = _extract_links(html, source_url)

    for art in articles:
        art_url = art["url"]
        _crawl_status["current"] = f"{name}: {art_url}"

        existing = conn.execute(
            "SELECT id, content_hash FROM crawled_pages WHERE url = ?", (art_url,)
        ).fetchone()

        time.sleep(CRAWL_DELAY)
        if js_render:
            result = _fetch_rendered(art_url)
            if not result:
                continue
            title, content = result
        else:
            html = _fetch(art_url)
            if not html:
                continue
            title, content = _extract_content(html)

        if len(content) < 100:
            continue

        title = art.get("title") or title
        chash = hashlib.md5(content.encode()).hexdigest()
        now = datetime.now(timezone.utc).isoformat()

        if existing and existing[1] == chash:
            continue  # content unchanged

        chunks = _chunk_text(content)
        if not chunks:
            continue

        try:
            embeddings = _embed(chunks)
        except Exception as e:
            logger.error(f"Embed failed {art_url}: {e}")
            _crawl_status["errors"].append(f"Embed: {art_url}")
            continue

        page_id = hashlib.md5(art_url.encode()).hexdigest()
        vectors = []
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            vectors.append({
                "id": f"{page_id}_{i}",
                "values": emb,
                "metadata": {
                    "competitor_name": name,
                    "url": art_url,
                    "title": title,
                    "source_type": stype,
                    "published_at": art.get("published_at") or "",
                    "crawled_at": now,
                    "chunk_index": i,
                    "text": chunk,
                },
            })

        for batch_start in range(0, len(vectors), 100):
            index.upsert(vectors=vectors[batch_start : batch_start + 100])

        # Knowledge graph extraction (runs inline, non-blocking for crawl)
        try:
            from kg_extract import extract_knowledge
            extract_knowledge(content, competitor_name=name, source_url=art_url)
        except Exception as e:
            logger.warning(f"[kg] extraction failed for {art_url}: {e}")

        if existing:
            conn.execute(
                "UPDATE crawled_pages SET content_hash=?, last_updated_at=?, title=? WHERE url=?",
                (chash, now, title, art_url),
            )
            _crawl_status["updated_articles"] += 1
        else:
            conn.execute(
                "INSERT INTO crawled_pages (id,source_id,url,title,content_hash,published_at,first_seen_at,last_updated_at) VALUES (?,?,?,?,?,?,?,?)",
                (page_id, source_id, art_url, title, chash, art.get("published_at"), now, now),
            )
            _crawl_status["new_articles"] += 1
        conn.commit()

    conn.execute(
        "UPDATE crawl_sources SET last_crawled_at=? WHERE id=?",
        (datetime.now(timezone.utc).isoformat(), source_id),
    )
    conn.commit()
    conn.close()


def _crawl_background(competitor_name: str = ""):
    threading.Thread(target=_keep_alive, daemon=True).start()
    try:
        conn = sqlite3.connect(DB_PATH)
        if competitor_name:
            sources = conn.execute(
                "SELECT id, competitor_name, source_url, source_type, js_render FROM crawl_sources WHERE enabled=1 AND competitor_name=?",
                (competitor_name,),
            ).fetchall()
        else:
            sources = conn.execute(
                "SELECT id, competitor_name, source_url, source_type, js_render FROM crawl_sources WHERE enabled=1"
            ).fetchall()
        conn.close()

        _crawl_status["total"] = len(sources)
        for i, (sid, cname, url, stype, js) in enumerate(sources):
            _crawl_status["done"] = i
            logger.info(f"Crawling ({i + 1}/{len(sources)}): {cname} — {url}")
            try:
                _crawl_source(sid, cname, url, stype, js_render=bool(js))
            except Exception as exc:
                logger.exception(f"Source failed: {url}")
                _crawl_status["errors"].append(f"Source: {url} — {type(exc).__name__}: {exc}")

        _crawl_status["done"] = len(sources)
        _crawl_status["current"] = ""
        logger.info(
            f"Crawl done: {_crawl_status['new_articles']} new, "
            f"{_crawl_status['updated_articles']} updated, "
            f"{len(_crawl_status['errors'])} errors"
        )
    except Exception as e:
        logger.exception("Crawl failed")
        _crawl_status["errors"].append(str(e))
    finally:
        _crawl_status["running"] = False


def auto_crawl_on_startup():
    """Start background crawl on server startup if configured."""
    if not PINECONE_API_KEY or not OPENAI_API_KEY:
        logger.info("Skipping auto-crawl: PINECONE_API_KEY or OPENAI_API_KEY not set.")
        return
    _init_db()
    _load_competitors()
    conn = sqlite3.connect(DB_PATH)
    count = conn.execute("SELECT COUNT(*) FROM crawl_sources WHERE enabled=1").fetchone()[0]
    conn.close()
    if count == 0:
        logger.info("No competitor sources configured.")
        return
    logger.info(f"Starting background crawl for {count} sources...")
    _crawl_status.update({
        "running": True, "total": 0, "done": 0, "current": "starting...",
        "errors": [], "new_articles": 0, "updated_articles": 0,
    })
    threading.Thread(target=_crawl_background, daemon=True).start()


# --- MCP Tools ---

@mcp.tool()
def search_competitor_content(
    query: str,
    competitor_name: str = "",
    source_type: str = "",
    date_from: str = "",
    date_to: str = "",
    top_k: int = 10,
) -> str:
    """
    Search competitor blog/content marketing for information relevant to your query.
    Query and content may be in any language (search runs in both EN and VI); if nothing
    relevant is in the DB, it falls back to a live web search and indexes the results.
    ALWAYS answer the user in Vietnamese or English regardless of the source language.

    Args:
        query: What to search for
        competitor_name: Filter by competitor (optional)
        source_type: Filter by type: blog, rss, case_study, press, web_search (optional)
        date_from: From date YYYY-MM-DD (optional)
        date_to: To date YYYY-MM-DD (optional)
        top_k: Number of results (default 10)
    """
    if not PINECONE_API_KEY or not OPENAI_API_KEY:
        return "Error: PINECONE_API_KEY or OPENAI_API_KEY not set."

    # Cross-lingual: competitor content is mostly Vietnamese, so search with the query
    # in both EN and VI and merge, letting a query in either language find results.
    variants = _bilingual_queries(query)
    try:
        qvecs = _embed(variants)
    except Exception as e:
        return f"Embedding error: {e}"

    filters = {}
    if competitor_name:
        filters["competitor_name"] = {"$eq": competitor_name}
    if source_type:
        filters["source_type"] = {"$eq": source_type}
    if date_from and date_to:
        filters["published_at"] = {"$gte": date_from, "$lte": date_to}
    elif date_from:
        filters["published_at"] = {"$gte": date_from}
    elif date_to:
        filters["published_at"] = {"$lte": date_to}

    index = _get_index()
    merged: dict = {}
    for qvec in qvecs:
        res = index.query(
            vector=qvec, top_k=top_k, include_metadata=True,
            filter=filters if filters else None,
        )
        for m in res.matches:
            if m.id not in merged or m.score > merged[m.id].score:
                merged[m.id] = m
    matches = sorted(merged.values(), key=lambda m: m.score, reverse=True)[:top_k]

    # Fallback: nothing in the DB (or only weak matches) -> web search via Tavily and
    # index the results, so the DB self-enriches over time.
    best = matches[0].score if matches else 0.0
    if best < FALLBACK_SCORE_THRESHOLD and TAVILY_API_KEY:
        try:
            web = _tavily_search(query)
            if web:
                _index_tavily_results(web, competitor_name)
                return _format_web_results(query, web)
        except Exception as e:
            logger.warning(f"Tavily fallback failed: {e}")

    if not matches:
        msg = "No results found."
        if not TAVILY_API_KEY:
            msg += " (Set TAVILY_API_KEY to enable web-search fallback.)"
        return msg

    out = [f"Found {len(matches)} results for: '{query}'\n"]
    for i, m in enumerate(matches):
        md = m.metadata
        out.append(
            f"\n--- Result {i + 1} (score: {m.score:.3f}) ---\n"
            f"Competitor: {md.get('competitor_name', '?')}\n"
            f"Title: {md.get('title', '?')}\n"
            f"URL: {md.get('url', '')}\n"
            f"Published: {md.get('published_at', '?')}\n"
            f"Content: {md.get('text', '')}\n"
        )
    return "".join(out)


@mcp.tool()
def web_search_and_index(query: str, competitor_name: str = "") -> str:
    """
    Search the web via Tavily and index the results into the vector DB, enriching future
    searches. Use when competitor info isn't in the DB yet (search_competitor_content also
    calls this automatically when it finds nothing relevant).
    Sources may be in any language; ALWAYS answer the user in Vietnamese or English.

    Args:
        query: What to search the web for
        competitor_name: Tag results with this competitor (optional)
    """
    if not TAVILY_API_KEY:
        return "Error: TAVILY_API_KEY not set."
    if not PINECONE_API_KEY or not OPENAI_API_KEY:
        return "Error: PINECONE_API_KEY or OPENAI_API_KEY not set."
    try:
        results = _tavily_search(query)
    except Exception as e:
        return f"Tavily error: {e}"
    if not results:
        return "No web results found."
    n = _index_tavily_results(results, competitor_name)
    return f"Indexed {n} web result(s) into the DB.\n\n" + _format_web_results(query, results)


@mcp.tool()
def list_competitor_topics(competitor_name: str, date_from: str = "", date_to: str = "") -> str:
    """
    List articles published by a competitor.

    Args:
        competitor_name: Competitor name
        date_from: From date YYYY-MM-DD (optional)
        date_to: To date YYYY-MM-DD (optional)
    """
    conn = sqlite3.connect(DB_PATH)
    q = """
        SELECT cp.title, cp.url, cp.published_at, cp.first_seen_at
        FROM crawled_pages cp JOIN crawl_sources cs ON cp.source_id = cs.id
        WHERE cs.competitor_name = ?
    """
    params: list = [competitor_name]
    if date_from:
        q += " AND COALESCE(cp.published_at, cp.first_seen_at) >= ?"
        params.append(date_from)
    if date_to:
        q += " AND COALESCE(cp.published_at, cp.first_seen_at) <= ?"
        params.append(date_to)
    q += " ORDER BY COALESCE(cp.published_at, cp.first_seen_at) DESC"

    rows = conn.execute(q, params).fetchall()
    conn.close()

    if not rows:
        return f"No articles found for {competitor_name}."

    lines = [f"Articles by {competitor_name} ({len(rows)} total):\n"]
    for title, url, pub, seen in rows:
        d = (pub or seen or "?")[:10]
        lines.append(f"- [{d}] {title}\n  {url}")
    return "\n".join(lines)


@mcp.tool()
def compare_competitor_messaging(competitor_names: list[str], topic: str, top_k: int = 5) -> str:
    """
    Compare how different competitors communicate about a specific topic.

    Args:
        competitor_names: List of competitor names to compare
        topic: Topic to compare messaging about
        top_k: Results per competitor (default 5)
    """
    if not PINECONE_API_KEY or not OPENAI_API_KEY:
        return "Error: PINECONE_API_KEY or OPENAI_API_KEY not set."

    try:
        qvec = _embed([topic])[0]
    except Exception as e:
        return f"Embedding error: {e}"

    index = _get_index()
    sections = []
    for name in competitor_names:
        results = index.query(
            vector=qvec, top_k=top_k, include_metadata=True,
            filter={"competitor_name": {"$eq": name}},
        )
        sec = f"\n{'=' * 50}\n{name}\n{'=' * 50}"
        if not results.matches:
            sec += "\nNo content found."
        else:
            for i, m in enumerate(results.matches):
                md = m.metadata
                sec += (
                    f"\n\n[{i + 1}] {md.get('title', '?')} (score: {m.score:.3f})\n"
                    f"URL: {md.get('url', '')}\n"
                    f"{md.get('text', '')}"
                )
        sections.append(sec)

    return f"Messaging comparison for: '{topic}'\n" + "\n".join(sections)


@mcp.tool()
def get_crawl_status(competitor_name: str = "") -> str:
    """
    Check crawler status and indexed content stats.

    Args:
        competitor_name: Filter by competitor (optional)
    """
    lines = []
    if _crawl_status["running"]:
        lines.append(f"Crawl: IN PROGRESS ({_crawl_status['done']}/{_crawl_status['total']} sources)")
        lines.append(f"Current: {_crawl_status['current']}")
    elif _crawl_status["total"] > 0:
        lines.append(f"Crawl: COMPLETE ({_crawl_status['done']}/{_crawl_status['total']} sources)")
    else:
        lines.append("No crawl has run yet.")

    lines.append(f"New: {_crawl_status['new_articles']}, Updated: {_crawl_status['updated_articles']}")

    if _crawl_status["errors"]:
        lines.append(f"\nErrors ({len(_crawl_status['errors'])}):")
        for e in _crawl_status["errors"][-5:]:
            lines.append(f"  - {e}")

    conn = sqlite3.connect(DB_PATH)
    if competitor_name:
        sources = conn.execute(
            "SELECT cs.competitor_name, cs.source_url, cs.source_type, cs.last_crawled_at, COUNT(cp.id) "
            "FROM crawl_sources cs LEFT JOIN crawled_pages cp ON cs.id = cp.source_id "
            "WHERE cs.competitor_name = ? GROUP BY cs.id",
            (competitor_name,),
        ).fetchall()
    else:
        sources = conn.execute(
            "SELECT cs.competitor_name, cs.source_url, cs.source_type, cs.last_crawled_at, COUNT(cp.id) "
            "FROM crawl_sources cs LEFT JOIN crawled_pages cp ON cs.id = cp.source_id "
            "GROUP BY cs.id ORDER BY cs.competitor_name"
        ).fetchall()
    conn.close()

    if sources:
        lines.append("\nSources:")
        for name, url, stype, last, count in sources:
            last_str = last[:16] if last else "never"
            lines.append(f"  [{name}] {stype} | {count} articles | last: {last_str}")
            lines.append(f"    {url}")

    return "\n".join(lines)


@mcp.tool()
def trigger_crawl(competitor_name: str = "") -> str:
    """
    Manually start a crawl. Crawls all competitors or a specific one.

    Args:
        competitor_name: Only crawl this competitor (optional, all if empty)
    """
    if not PINECONE_API_KEY or not OPENAI_API_KEY:
        return "Error: PINECONE_API_KEY or OPENAI_API_KEY not set."

    if _crawl_status["running"]:
        return f"Crawl already running: {_crawl_status['done']}/{_crawl_status['total']} sources."

    _init_db()
    _load_competitors()
    _crawl_status.update({
        "running": True, "total": 0, "done": 0, "current": "starting...",
        "errors": [], "new_articles": 0, "updated_articles": 0,
    })
    threading.Thread(target=_crawl_background, args=(competitor_name,), daemon=True).start()
    target = competitor_name or "all competitors"
    return f"Crawl started for {target}. Use get_crawl_status to check progress."


if __name__ == "__main__":
    _init_db()
    _load_competitors()
    mcp.run(transport="stdio")
