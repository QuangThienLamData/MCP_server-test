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
from youtube_transcript_api import YouTubeTranscriptApi

from rag_mcp import (
    DB_PATH, OPENAI_API_KEY, PINECONE_API_KEY,
    _bilingual_queries, _chunk_text, _embed, _get_index,
)

load_dotenv()

logger = logging.getLogger(__name__)

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "")
YT_API_BASE = "https://www.googleapis.com/youtube/v3"
YT_NAMESPACE = "youtube"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
FALLBACK_SCORE_THRESHOLD = 0.40  # below this, fallback to YouTube API search

mcp = FastMCP(
    name="YouTube Intelligence MCP Server",
    transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
)

_yt_crawl_status = {
    "running": False, "total": 0, "done": 0, "current": "",
    "errors": [], "new_videos": 0,
}


# --- SQLite ---
def _init_yt_db():
    conn = sqlite3.connect(DB_PATH)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS youtube_videos (
            id TEXT PRIMARY KEY,
            title TEXT,
            channel TEXT,
            topic TEXT,
            url TEXT NOT NULL UNIQUE,
            view_count INTEGER DEFAULT 0,
            published_at TEXT,
            transcript_lang TEXT,
            transcript_hash TEXT,
            indexed_at TEXT NOT NULL
        );
    """)
    conn.commit()
    conn.close()


# --- YouTube Data API ---
def _yt_search(topic: str, max_results: int = 10, order: str = "viewCount") -> list[dict]:
    """Search YouTube for videos on a topic. Returns list of {video_id, title, channel, published_at}."""
    if not YOUTUBE_API_KEY:
        return []
    params = {
        "part": "snippet",
        "q": topic,
        "type": "video",
        "order": order,
        "maxResults": min(max_results, 50),
        "key": YOUTUBE_API_KEY,
    }
    try:
        with httpx.Client(timeout=30) as c:
            r = c.get(f"{YT_API_BASE}/search", params=params)
            r.raise_for_status()
            data = r.json()
    except Exception as e:
        logger.error(f"YouTube search failed: {e}")
        return []

    results = []
    for item in data.get("items", []):
        vid = item["id"].get("videoId")
        if not vid:
            continue
        snippet = item["snippet"]
        results.append({
            "video_id": vid,
            "title": snippet.get("title", ""),
            "channel": snippet.get("channelTitle", ""),
            "published_at": snippet.get("publishedAt", ""),
        })
    return results


def _yt_video_stats(video_ids: list[str]) -> dict[str, int]:
    """Get view counts for a batch of video IDs."""
    if not YOUTUBE_API_KEY or not video_ids:
        return {}
    params = {
        "part": "statistics",
        "id": ",".join(video_ids),
        "key": YOUTUBE_API_KEY,
    }
    try:
        with httpx.Client(timeout=30) as c:
            r = c.get(f"{YT_API_BASE}/videos", params=params)
            r.raise_for_status()
            data = r.json()
    except Exception as e:
        logger.error(f"YouTube stats failed: {e}")
        return {}

    stats = {}
    for item in data.get("items", []):
        vid = item["id"]
        vc = item.get("statistics", {}).get("viewCount", "0")
        stats[vid] = int(vc)
    return stats


# --- Transcript extraction via youtube-transcript-api ---
_ytt_api = YouTubeTranscriptApi()


def _extract_transcript(video_id: str, langs: list[str] | None = None) -> tuple[str, str] | None:
    """Extract transcript from a YouTube video using InnerTube API.
    Tries preferred languages first, then falls back to any available transcript
    (including auto-generated). Returns (text, lang) or None."""
    if langs is None:
        langs = ["vi", "en"]
    # 1. Try preferred languages
    try:
        transcript = _ytt_api.fetch(video_id, languages=langs)
        text = " ".join(s.text.strip() for s in transcript if s.text.strip())
        if text:
            return text, transcript.language
    except Exception:
        pass
    # 2. Fall back to any available transcript (auto-generated included)
    try:
        available = _ytt_api.list(video_id)
        for t in available:
            try:
                transcript = _ytt_api.fetch(video_id, languages=[t.language_code])
                text = " ".join(s.text.strip() for s in transcript if s.text.strip())
                if text:
                    return text, t.language_code
            except Exception:
                continue
    except Exception as e:
        logger.error(f"Transcript fetch failed for {video_id}: {e}")
    return None


# --- Indexing engine ---
def _index_video(video: dict, topic: str) -> bool:
    """Extract transcript, chunk, embed, and store in Pinecone + SQLite."""
    video_id = video["video_id"]
    url = f"https://www.youtube.com/watch?v={video_id}"

    conn = sqlite3.connect(DB_PATH)
    existing = conn.execute("SELECT transcript_hash FROM youtube_videos WHERE id = ?", (video_id,)).fetchone()

    result = _extract_transcript(video_id)
    if not result:
        conn.close()
        return False

    transcript, lang = result
    if len(transcript) < 50:
        conn.close()
        return False

    thash = hashlib.md5(transcript.encode()).hexdigest()
    if existing and existing[0] == thash:
        conn.close()
        return False  # unchanged

    chunks = _chunk_text(transcript)
    if not chunks:
        conn.close()
        return False

    try:
        embeddings = _embed(chunks)
    except Exception as e:
        logger.error(f"Embed failed {video_id}: {e}")
        conn.close()
        return False

    now = datetime.now(timezone.utc).isoformat()
    title = video.get("title", "")
    channel = video.get("channel", "")
    published = video.get("published_at", "")
    view_count = video.get("view_count", 0)

    vectors = []
    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        vectors.append({
            "id": f"yt_{video_id}_{i}",
            "values": emb,
            "metadata": {
                "source_type": "youtube",
                "video_id": video_id,
                "url": url,
                "title": title,
                "channel": channel,
                "topic": topic,
                "published_at": published[:10] if published else "",
                "view_count": view_count,
                "lang": lang,
                "chunk_index": i,
                "text": chunk,
            },
        })

    index = _get_index()
    for b in range(0, len(vectors), 100):
        index.upsert(vectors=vectors[b:b + 100], namespace=YT_NAMESPACE)

    if existing:
        conn.execute(
            "UPDATE youtube_videos SET title=?, channel=?, topic=?, view_count=?, transcript_lang=?, transcript_hash=?, indexed_at=? WHERE id=?",
            (title, channel, topic, view_count, lang, thash, now, video_id),
        )
    else:
        conn.execute(
            "INSERT INTO youtube_videos (id, title, channel, topic, url, view_count, published_at, transcript_lang, transcript_hash, indexed_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (video_id, title, channel, topic, url, view_count, published, lang, thash, now),
        )
    conn.commit()
    conn.close()
    return True


def _crawl_topic_background(topic: str, max_videos: int, order: str):
    """Background worker: search YouTube, extract transcripts, index."""
    try:
        videos = _yt_search(topic, max_results=max_videos, order=order)
        if not videos:
            _yt_crawl_status["errors"].append(f"No videos found for '{topic}'")
            return

        # Fetch view counts
        vids = [v["video_id"] for v in videos]
        stats = _yt_video_stats(vids)
        for v in videos:
            v["view_count"] = stats.get(v["video_id"], 0)

        _yt_crawl_status["total"] = len(videos)
        for i, video in enumerate(videos):
            _yt_crawl_status["done"] = i
            _yt_crawl_status["current"] = f"{video['title'][:60]}..."
            logger.info(f"YouTube ({i + 1}/{len(videos)}): {video['title'][:60]}")
            try:
                if _index_video(video, topic):
                    _yt_crawl_status["new_videos"] += 1
            except Exception as exc:
                logger.exception(f"Video failed: {video['video_id']}")
                _yt_crawl_status["errors"].append(f"{video['video_id']}: {type(exc).__name__}: {exc}")
            time.sleep(1)  # rate limit

        _yt_crawl_status["done"] = len(videos)
        _yt_crawl_status["current"] = ""
        logger.info(f"YouTube crawl done: {_yt_crawl_status['new_videos']} new, {len(_yt_crawl_status['errors'])} errors")
    except Exception as e:
        logger.exception("YouTube crawl failed")
        _yt_crawl_status["errors"].append(str(e))
    finally:
        _yt_crawl_status["running"] = False


# --- MCP Tools ---

@mcp.tool()
def crawl_youtube_topic(
    topic: str,
    max_videos: int = 10,
    order: str = "viewCount",
) -> str:
    """
    Search YouTube for the hottest videos on a topic, extract their transcripts,
    and index them into the vector DB for semantic search.

    Args:
        topic: Search query / topic to find videos about
        max_videos: Number of videos to process (default 10, max 50)
        order: Sort order — 'viewCount' (most popular), 'date' (newest), 'relevance' (default: viewCount)
    """
    if not YOUTUBE_API_KEY:
        return "Error: YOUTUBE_API_KEY not set."
    if not PINECONE_API_KEY or not OPENAI_API_KEY:
        return "Error: PINECONE_API_KEY or OPENAI_API_KEY not set."
    if _yt_crawl_status["running"]:
        return f"YouTube crawl already running: {_yt_crawl_status['done']}/{_yt_crawl_status['total']} videos."

    _init_yt_db()
    max_videos = min(max(1, max_videos), 50)
    if order not in ("viewCount", "date", "relevance"):
        order = "viewCount"

    _yt_crawl_status.update({
        "running": True, "total": 0, "done": 0, "current": "searching...",
        "errors": [], "new_videos": 0,
    })
    threading.Thread(target=_crawl_topic_background, args=(topic, max_videos, order), daemon=True).start()
    return f"YouTube crawl started for '{topic}' (up to {max_videos} videos, order={order}). Use get_youtube_status to check progress."


def _live_search_and_index(query: str, max_videos: int = 5) -> list[dict]:
    """Search YouTube API, extract transcripts, index into Pinecone on the fly.
    Returns list of video dicts that were successfully indexed."""
    _init_yt_db()
    videos = _yt_search(query, max_results=max_videos, order="relevance")
    if not videos:
        return []
    stats = _yt_video_stats([v["video_id"] for v in videos])
    for v in videos:
        v["view_count"] = stats.get(v["video_id"], 0)

    indexed = []
    for v in videos:
        try:
            if _index_video(v, query):
                indexed.append(v)
        except Exception:
            logger.warning(f"Live index failed for {v['video_id']}")
        time.sleep(1)
    return indexed


def _format_matches(query: str, matches, source_label: str = "") -> str:
    label = f" ({source_label})" if source_label else ""
    out = [f"Found {len(matches)} results for: '{query}'{label}\n"]
    for i, m in enumerate(matches):
        md = m.metadata if hasattr(m, "metadata") else m
        out.append(
            f"\n--- Result {i + 1} (score: {m.score:.3f}) ---\n"
            f"Title: {md.get('title', '?')}\n"
            f"Channel: {md.get('channel', '?')}\n"
            f"URL: {md.get('url', '')}\n"
            f"Views: {md.get('view_count', '?')} | Topic: {md.get('topic', '?')}\n"
            f"Transcript: {md.get('text', '')}\n"
        )
    return "".join(out)


def _search_db(query: str, topic: str = "", channel: str = "", top_k: int = 10) -> list:
    """Search Pinecone for indexed YouTube transcripts. Returns sorted matches."""
    variants = _bilingual_queries(query)
    qvecs = _embed(variants)

    filters: dict = {"source_type": {"$eq": "youtube"}}
    if topic:
        filters["topic"] = {"$eq": topic}
    if channel:
        filters["channel"] = {"$eq": channel}

    index = _get_index()
    merged: dict = {}
    for qvec in qvecs:
        res = index.query(
            vector=qvec, top_k=top_k, include_metadata=True,
            filter=filters, namespace=YT_NAMESPACE,
        )
        for m in res.matches:
            if m.id not in merged or m.score > merged[m.id].score:
                merged[m.id] = m
    return sorted(merged.values(), key=lambda m: m.score, reverse=True)[:top_k]


@mcp.tool()
def search_video_content(
    query: str,
    topic: str = "",
    channel: str = "",
    top_k: int = 10,
) -> str:
    """
    Search YouTube video transcripts. Searches indexed DB first; if nothing relevant
    is found, automatically searches YouTube API, extracts transcripts, indexes them,
    and returns the results. Supports bilingual search (EN/VI).

    Args:
        query: What to search for
        topic: Filter by crawled topic (optional)
        channel: Filter by channel name (optional)
        top_k: Number of results (default 10)
    """
    if not PINECONE_API_KEY or not OPENAI_API_KEY:
        return "Error: PINECONE_API_KEY or OPENAI_API_KEY not set."

    # Step 1: Search existing DB
    try:
        matches = _search_db(query, topic=topic, channel=channel, top_k=top_k)
    except Exception as e:
        return f"Search error: {e}"

    best_score = matches[0].score if matches else 0.0

    # Step 2: If good results in DB, return them
    if best_score >= FALLBACK_SCORE_THRESHOLD:
        return _format_matches(query, matches, "from indexed DB")

    # Step 3: Fallback — search YouTube API, extract & index, then re-search
    if not YOUTUBE_API_KEY:
        if matches:
            return _format_matches(query, matches, "from indexed DB — weak matches")
        return "No results in DB. Set YOUTUBE_API_KEY to enable live YouTube search fallback."

    logger.info(f"DB score {best_score:.3f} < {FALLBACK_SCORE_THRESHOLD}, falling back to YouTube API")
    try:
        indexed = _live_search_and_index(query, max_videos=5)
    except Exception as e:
        logger.warning(f"YouTube fallback failed: {e}")
        if matches:
            return _format_matches(query, matches, "from indexed DB — weak matches")
        return f"No results in DB and YouTube fallback failed: {e}"

    if not indexed:
        if matches:
            return _format_matches(query, matches, "from indexed DB — weak matches")
        return "No results found in DB, and no YouTube videos with transcripts found for this query."

    # Re-search DB now that new content is indexed
    try:
        matches = _search_db(query, top_k=top_k)
    except Exception as e:
        return f"Re-search error after indexing: {e}"

    newly = ", ".join(v["title"][:40] for v in indexed[:3])
    suffix = f"\n\n(Auto-indexed {len(indexed)} new video(s): {newly})"
    if matches:
        return _format_matches(query, matches, "after live YouTube indexing") + suffix
    return f"Indexed {len(indexed)} video(s) but no transcript matches found.{suffix}"


@mcp.tool()
def get_video_transcript(video_url: str) -> str:
    """
    Extract and return the transcript of a single YouTube video (without indexing).

    Args:
        video_url: YouTube video URL or video ID
    """
    # Extract video ID from URL
    video_id = video_url.strip()
    for pattern in [r"v=([a-zA-Z0-9_-]{11})", r"youtu\.be/([a-zA-Z0-9_-]{11})", r"^([a-zA-Z0-9_-]{11})$"]:
        m = re.search(pattern, video_id)
        if m:
            video_id = m.group(1)
            break

    result = _extract_transcript(video_id)
    if not result:
        return f"No transcript found for video {video_id}. The video may not have captions."

    transcript, lang = result
    return f"Language: {lang}\nLength: {len(transcript)} chars\n\n{transcript}"


@mcp.tool()
def list_indexed_videos(topic: str = "") -> str:
    """
    List YouTube videos that have been indexed.

    Args:
        topic: Filter by topic (optional)
    """
    _init_yt_db()
    conn = sqlite3.connect(DB_PATH)
    if topic:
        rows = conn.execute(
            "SELECT title, channel, topic, url, view_count, published_at, transcript_lang FROM youtube_videos WHERE topic = ? ORDER BY view_count DESC",
            (topic,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT title, channel, topic, url, view_count, published_at, transcript_lang FROM youtube_videos ORDER BY view_count DESC"
        ).fetchall()
    conn.close()

    if not rows:
        return "No videos indexed yet. Use crawl_youtube_topic to start."

    lines = [f"Indexed videos ({len(rows)} total):\n"]
    for title, channel, tp, url, views, pub, lang in rows:
        pub_str = pub[:10] if pub else "?"
        lines.append(
            f"- [{pub_str}] {title}\n"
            f"  Channel: {channel} | Views: {views:,} | Lang: {lang} | Topic: {tp}\n"
            f"  {url}"
        )
    return "\n".join(lines)


@mcp.tool()
def get_youtube_status() -> str:
    """Check the status of the current YouTube crawl."""
    if _yt_crawl_status["running"]:
        lines = [f"YouTube crawl: IN PROGRESS ({_yt_crawl_status['done']}/{_yt_crawl_status['total']} videos)"]
        lines.append(f"Current: {_yt_crawl_status['current']}")
    elif _yt_crawl_status["total"] > 0:
        lines = [f"YouTube crawl: COMPLETE ({_yt_crawl_status['done']}/{_yt_crawl_status['total']} videos)"]
    else:
        lines = ["No YouTube crawl has run yet."]

    lines.append(f"New videos indexed: {_yt_crawl_status['new_videos']}")

    if _yt_crawl_status["errors"]:
        lines.append(f"\nErrors ({len(_yt_crawl_status['errors'])}):")
        for e in _yt_crawl_status["errors"][-5:]:
            lines.append(f"  - {e[:120]}")

    return "\n".join(lines)


def trigger_crawl(**kwargs) -> str:
    """Called by the cron endpoint in main.py."""
    return "YouTube crawl must be triggered with a topic via crawl_youtube_topic tool."


if __name__ == "__main__":
    _init_yt_db()
    mcp.run(transport="stdio")
