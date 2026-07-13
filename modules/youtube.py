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

from modules.rag import (
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

# RapidAPI YouTube transcript fallback (for cloud IPs blocked by YouTube)
# Uses "YouTube Transcriptor" API on RapidAPI (by benrhzala90).
# Subscribe at: https://rapidapi.com/benrhzala90/api/youtube-transcriptor
# Uses the same RapidAPI key as TikTok/Facebook.
RAPIDAPI_KEY = os.getenv("RAPIDAPI_TT_KEY", "") or os.getenv("RAPIDAPI_FB_KEY", "")
RAPIDAPI_YT_HOST = os.getenv("RAPIDAPI_YT_HOST", "youtube-transcriptor.p.rapidapi.com")

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


# --- Transcript extraction ---
_ytt_api = YouTubeTranscriptApi()
_ip_blocked = False  # set True after first IP block detection, skip further direct attempts


def _is_cloud_ip_block(exc: Exception) -> bool:
    """Detect if the error is a permanent cloud IP block (not a temporary rate limit)."""
    msg = str(exc)
    cls = type(exc).__name__
    # Cloud provider IP block: permanent, switch to RapidAPI
    if "cloud provider" in msg.lower() or "IPBlocked" in cls:
        return True
    # Generic "blocking" + "IP" = likely cloud block
    if "blocking" in msg.lower() and "ip" in msg.lower():
        return True
    return False


def _extract_via_library(video_id: str, langs: list[str]) -> tuple[str, str] | None:
    """Try youtube-transcript-api (direct, works from non-cloud IPs)."""
    global _ip_blocked
    if _ip_blocked:
        return None
    # 1. Try preferred languages
    try:
        transcript = _ytt_api.fetch(video_id, languages=langs)
        text = " ".join(s.text.strip() for s in transcript if s.text.strip())
        if text:
            return text, transcript.language
    except Exception as e:
        if _is_cloud_ip_block(e):
            _ip_blocked = True
            logger.warning("YouTube cloud IP blocked — switching to RapidAPI for all future requests")
            return None
    # 2. Fall back to any available transcript
    try:
        available = _ytt_api.list(video_id)
        for t in available:
            try:
                transcript = _ytt_api.fetch(video_id, languages=[t.language_code])
                text = " ".join(s.text.strip() for s in transcript if s.text.strip())
                if text:
                    return text, t.language_code
            except Exception as e2:
                if _is_cloud_ip_block(e2):
                    _ip_blocked = True
                    logger.warning("YouTube cloud IP blocked — switching to RapidAPI")
                    return None
                continue
    except Exception as e:
        if _is_cloud_ip_block(e):
            _ip_blocked = True
            logger.warning("YouTube cloud IP blocked — switching to RapidAPI")
        else:
            logger.warning(f"Transcript unavailable for {video_id}: {type(e).__name__}")
    return None


def _extract_via_rapidapi(video_id: str, lang: str = "vi") -> tuple[str, str] | None:
    """Fetch transcript via RapidAPI (works from cloud IPs).
    Requires subscription to a YouTube transcript API on RapidAPI."""
    if not RAPIDAPI_KEY or not RAPIDAPI_YT_HOST:
        return None
    headers = {
        "x-rapidapi-key": RAPIDAPI_KEY,
        "x-rapidapi-host": RAPIDAPI_YT_HOST,
    }
    with httpx.Client(timeout=30) as c:
        for attempt in range(3):
            try:
                r = c.get(
                    f"https://{RAPIDAPI_YT_HOST}/transcript",
                    params={"video_id": video_id, "lang": lang},
                    headers=headers,
                )
                if r.status_code == 429:
                    wait = 2 ** (attempt + 1)
                    logger.info(f"RapidAPI rate limited, waiting {wait}s...")
                    time.sleep(wait)
                    continue
                if r.status_code in (403, 404):
                    return None
                r.raise_for_status()
                data = r.json()
            except httpx.HTTPStatusError:
                return None
            except Exception as e:
                logger.warning(f"RapidAPI transcript failed for {video_id}: {e}")
                return None

            # youtube-transcriptor API returns:
            # [{"transcriptionAsText": "full text...", "transcription": [{"subtitle":..}], ...}]
            text = ""
            if isinstance(data, list) and data and isinstance(data[0], dict):
                item = data[0]
                # Prefer the pre-joined text
                text = item.get("transcriptionAsText", "")
                # Fallback: join subtitle segments
                if not text:
                    segs = item.get("transcription", [])
                    text = " ".join(s.get("subtitle", "") for s in segs if isinstance(s, dict))
            elif isinstance(data, dict):
                text = data.get("transcriptionAsText", "")
                if not text:
                    segs = data.get("transcription", data.get("subtitles", []))
                    text = " ".join(s.get("subtitle", s.get("text", "")) for s in segs if isinstance(s, dict))

            text = text.strip()
            if text:
                return text, lang
            return None
    return None


def _extract_transcript(video_id: str, langs: list[str] | None = None) -> tuple[str, str] | None:
    """Extract transcript from a YouTube video.
    Tries youtube-transcript-api first (direct), falls back to RapidAPI proxy
    when YouTube blocks cloud IPs. Returns (text, lang) or None."""
    if langs is None:
        langs = ["vi", "en"]

    # 1. Direct (fast, free, works from non-cloud IPs)
    result = _extract_via_library(video_id, langs)
    if result:
        return result

    # 2. RapidAPI fallback (works from cloud IPs)
    if RAPIDAPI_KEY:
        for lang in langs:
            result = _extract_via_rapidapi(video_id, lang=lang)
            if result:
                return result

    return None


# --- Indexing engine ---
def _index_video(video: dict, topic: str, pre_transcript: tuple[str, str] | None = None) -> bool:
    """Chunk, embed, and store in Pinecone + SQLite.
    If pre_transcript is given as (text, lang), skip transcript extraction."""
    video_id = video["video_id"]
    url = f"https://www.youtube.com/watch?v={video_id}"

    conn = sqlite3.connect(DB_PATH)
    existing = conn.execute("SELECT transcript_hash FROM youtube_videos WHERE id = ?", (video_id,)).fetchone()

    if pre_transcript:
        transcript, lang = pre_transcript
    else:
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


def _live_search_and_extract(query: str, max_videos: int = 5) -> list[dict]:
    """Search YouTube API, extract transcripts. Returns list of dicts with video info + transcript.
    Does NOT index into Pinecone — caller decides whether to index (in background)."""
    _init_yt_db()
    videos = _yt_search(query, max_results=max_videos, order="relevance")
    if not videos:
        return []
    stats = _yt_video_stats([v["video_id"] for v in videos])
    for v in videos:
        v["view_count"] = stats.get(v["video_id"], 0)

    results = []
    for v in videos:
        try:
            result = _extract_transcript(v["video_id"])
            if result:
                text, lang = result
                if len(text) >= 50:
                    results.append({**v, "transcript": text, "lang": lang})
        except Exception:
            logger.warning(f"Transcript failed for {v['video_id']}")
        time.sleep(1)
    return results


def _format_live_results(query: str, results: list[dict]) -> str:
    """Format live YouTube search results (with transcripts) for direct return."""
    out = [f"Found {len(results)} YouTube videos for: '{query}' (live search)\n"]
    for i, r in enumerate(results):
        out.append(
            f"\n--- Result {i + 1} ---\n"
            f"Title: {r.get('title', '?')}\n"
            f"Channel: {r.get('channel', '?')}\n"
            f"URL: https://www.youtube.com/watch?v={r['video_id']}\n"
            f"Views: {r.get('view_count', '?')}\n"
            f"Transcript ({r['lang']}): {r['transcript'][:800]}\n"
        )
    return "".join(out)


def _background_index_results(results: list[dict], topic: str):
    """Background worker: index pre-extracted video results into Pinecone + SQLite."""
    for r in results:
        try:
            _index_video(r, topic, pre_transcript=(r["transcript"], r["lang"]))
        except Exception as e:
            logger.warning(f"Background index failed for {r.get('video_id')}: {e}")


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

    # Step 3: Fallback — search YouTube API, extract transcripts, return directly
    if not YOUTUBE_API_KEY:
        if matches:
            return _format_matches(query, matches, "from indexed DB — weak matches")
        return "No results in DB. Set YOUTUBE_API_KEY to enable live YouTube search fallback."

    logger.info(f"DB score {best_score:.3f} < {FALLBACK_SCORE_THRESHOLD}, falling back to YouTube API")
    try:
        extracted = _live_search_and_extract(query, max_videos=5)
    except Exception as e:
        logger.warning(f"YouTube fallback failed: {e}")
        if matches:
            return _format_matches(query, matches, "from indexed DB — weak matches")
        return f"No results in DB and YouTube fallback failed: {e}"

    if not extracted:
        if matches:
            return _format_matches(query, matches, "from indexed DB — weak matches")
        return "No results found in DB, and no YouTube videos with transcripts found for this query."

    # Return results directly, index in background
    threading.Thread(
        target=_background_index_results, args=(extracted, query), daemon=True,
    ).start()
    return _format_live_results(query, extracted)


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
