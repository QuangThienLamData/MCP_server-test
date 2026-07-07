"""TikTok KOL/KOC Intelligence — evaluate and compare influencers in Vietnam.

Uses TikTok's official Research API (OAuth client credentials).
No Playwright/browser needed — lightweight HTTP-only, token auto-refreshes.

Setup:
  1. Apply for Research API access at developers.tiktok.com
  2. Set env vars: TIKTOK_CLIENT_KEY, TIKTOK_CLIENT_SECRET
"""

import asyncio
import json
import logging
import os
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from statistics import mean, stdev

import httpx
from dotenv import load_dotenv

from rag_mcp import DB_PATH, OPENAI_API_KEY, _get_openai

load_dotenv()

logger = logging.getLogger(__name__)

TIKTOK_CLIENT_KEY = os.getenv("TIKTOK_CLIENT_KEY", "")
TIKTOK_CLIENT_SECRET = os.getenv("TIKTOK_CLIENT_SECRET", "")

BASE_URL = "https://open.tiktokapis.com/v2"

VIDEO_FIELDS = (
    "id,video_description,create_time,username,region_code,"
    "like_count,comment_count,share_count,view_count,"
    "favorites_count,video_duration,hashtag_names,music_id"
)
USER_FIELDS = (
    "display_name,bio_description,avatar_url,is_verified,"
    "follower_count,following_count,video_count,likes_count"
)
COMMENT_FIELDS = "id,text,like_count,reply_count,create_time"

# ---------------------------------------------------------------------------
# Token management (auto-refresh, 2h TTL)
# ---------------------------------------------------------------------------

_access_token: str | None = None
_token_expires_at: float = 0.0


async def _get_token() -> str:
    """Get a valid access token, refreshing automatically if expired."""
    global _access_token, _token_expires_at
    now = time.time()
    if _access_token and now < _token_expires_at - 120:
        return _access_token

    if not TIKTOK_CLIENT_KEY or not TIKTOK_CLIENT_SECRET:
        raise RuntimeError(
            "TIKTOK_CLIENT_KEY / TIKTOK_CLIENT_SECRET not set. "
            "Apply for Research API access at developers.tiktok.com, "
            "then set these env vars."
        )

    async with httpx.AsyncClient(timeout=15) as c:
        resp = await c.post(
            f"{BASE_URL}/oauth/token/",
            data={
                "client_key": TIKTOK_CLIENT_KEY,
                "client_secret": TIKTOK_CLIENT_SECRET,
                "grant_type": "client_credentials",
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        resp.raise_for_status()
        data = resp.json()

    _access_token = data["access_token"]
    _token_expires_at = now + data.get("expires_in", 7200)
    logger.info("[tiktok] Access token refreshed, expires in %ds", data.get("expires_in", 7200))
    return _access_token


async def _api_post(path: str, body: dict, fields: str = "") -> dict:
    """Make an authenticated POST to the Research API."""
    token = await _get_token()
    url = f"{BASE_URL}{path}"
    if fields:
        url += f"?fields={fields}"
    async with httpx.AsyncClient(timeout=30) as c:
        resp = await c.post(
            url,
            json=body,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
        )
    data = resp.json()
    err = data.get("error", {})
    if err.get("code") and err["code"] != "ok":
        raise RuntimeError(f"TikTok API error: {err.get('message', err['code'])} (log_id: {err.get('log_id', '?')})")
    return data.get("data", data)


# ---------------------------------------------------------------------------
# SQLite
# ---------------------------------------------------------------------------

def _init_tiktok_db():
    """Create TikTok tables (idempotent)."""
    conn = sqlite3.connect(DB_PATH)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS tiktok_kols (
            username        TEXT PRIMARY KEY,
            display_name    TEXT,
            bio             TEXT,
            verified        INTEGER DEFAULT 0,
            follower_count  INTEGER DEFAULT 0,
            following_count INTEGER DEFAULT 0,
            likes_count     INTEGER DEFAULT 0,
            video_count     INTEGER DEFAULT 0,
            avatar_url      TEXT,
            tier            TEXT,
            tracked         INTEGER DEFAULT 0,
            raw_data        TEXT,
            updated_at      TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS tiktok_videos (
            video_id        TEXT PRIMARY KEY,
            username        TEXT NOT NULL,
            description     TEXT,
            create_time     TEXT,
            region_code     TEXT,
            view_count      INTEGER DEFAULT 0,
            like_count      INTEGER DEFAULT 0,
            comment_count   INTEGER DEFAULT 0,
            share_count     INTEGER DEFAULT 0,
            favorites_count INTEGER DEFAULT 0,
            video_duration  INTEGER DEFAULT 0,
            hashtag_names   TEXT DEFAULT '[]',
            music_id        TEXT,
            engagement_rate REAL DEFAULT 0,
            updated_at      TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_tv_username ON tiktok_videos(username);
        CREATE INDEX IF NOT EXISTS idx_tv_create   ON tiktok_videos(create_time);
    """)
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tier(followers: int) -> str:
    if followers >= 1_000_000:
        return "Mega (1M+)"
    if followers >= 500_000:
        return "Macro (500K-1M)"
    if followers >= 100_000:
        return "Mid-tier (100K-500K)"
    if followers >= 10_000:
        return "Micro (10K-100K)"
    if followers >= 1_000:
        return "Nano (1K-10K)"
    return "Emerging (<1K)"


def _er(views: int, likes: int, comments: int, shares: int) -> float:
    if views == 0:
        return 0.0
    return round((likes + comments + shares) / views * 100, 2)


def _fmt(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def _date_range(days_back: int = 30) -> tuple[str, str]:
    """Return (start_date, end_date) in YYYYMMDD format for the API."""
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=min(days_back, 30))
    return start.strftime("%Y%m%d"), end.strftime("%Y%m%d")


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

async def _fetch_profile(username: str) -> dict:
    """Fetch user profile from TikTok Research API and cache in SQLite."""
    data = await _api_post("/research/user/info/", {"username": username}, USER_FIELDS)

    profile = {
        "username": username,
        "display_name": data.get("display_name", ""),
        "bio": data.get("bio_description", ""),
        "verified": 1 if data.get("is_verified") else 0,
        "follower_count": data.get("follower_count", 0),
        "following_count": data.get("following_count", 0),
        "likes_count": data.get("likes_count", 0),
        "video_count": data.get("video_count", 0),
        "avatar_url": data.get("avatar_url", ""),
    }
    profile["tier"] = _tier(profile["follower_count"])

    now = datetime.now(timezone.utc).isoformat()
    conn = sqlite3.connect(DB_PATH)
    existing = conn.execute(
        "SELECT tracked FROM tiktok_kols WHERE username = ?", (username,)
    ).fetchone()
    tracked = existing[0] if existing else 0
    conn.execute(
        "INSERT OR REPLACE INTO tiktok_kols "
        "(username,display_name,bio,verified,follower_count,following_count,"
        "likes_count,video_count,avatar_url,tier,tracked,raw_data,updated_at) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            username, profile["display_name"], profile["bio"],
            profile["verified"], profile["follower_count"],
            profile["following_count"], profile["likes_count"],
            profile["video_count"], profile["avatar_url"],
            profile["tier"], tracked,
            json.dumps(data, ensure_ascii=False, default=str), now,
        ),
    )
    conn.commit()
    conn.close()
    return profile


async def _fetch_videos(username: str, count: int = 20, days_back: int = 30) -> list[dict]:
    """Fetch recent videos for a user via video query endpoint."""
    start_date, end_date = _date_range(days_back)
    videos: list[dict] = []
    cursor = 0
    search_id = ""

    while len(videos) < count:
        batch_size = min(count - len(videos), 100)
        body: dict = {
            "query": {
                "and": [
                    {"operation": "EQ", "field_name": "username", "field_values": [username]},
                ],
            },
            "start_date": start_date,
            "end_date": end_date,
            "max_count": batch_size,
        }
        if cursor:
            body["cursor"] = cursor
        if search_id:
            body["search_id"] = search_id

        data = await _api_post("/research/video/query/", body, VIDEO_FIELDS)
        vids = data.get("videos", [])
        if not vids:
            break

        for v in vids:
            views = v.get("view_count", 0)
            likes = v.get("like_count", 0)
            comments = v.get("comment_count", 0)
            shares = v.get("share_count", 0)
            favs = v.get("favorites_count", 0)
            ct = v.get("create_time", 0)
            create_time = (
                datetime.fromtimestamp(ct, tz=timezone.utc).isoformat() if ct else ""
            )
            hashtags = v.get("hashtag_names", []) or []

            vid = {
                "video_id": str(v.get("id", "")),
                "username": username,
                "description": v.get("video_description", ""),
                "create_time": create_time,
                "region_code": v.get("region_code", ""),
                "view_count": views,
                "like_count": likes,
                "comment_count": comments,
                "share_count": shares,
                "favorites_count": favs,
                "video_duration": v.get("video_duration", 0),
                "hashtag_names": hashtags,
                "music_id": str(v.get("music_id", "")),
                "engagement_rate": _er(views, likes, comments, shares),
            }
            videos.append(vid)

        if not data.get("has_more"):
            break
        cursor = data.get("cursor", 0)
        search_id = data.get("search_id", "")

    # Cache in SQLite
    now = datetime.now(timezone.utc).isoformat()
    conn = sqlite3.connect(DB_PATH)
    for vid in videos:
        conn.execute(
            "INSERT OR REPLACE INTO tiktok_videos "
            "(video_id,username,description,create_time,region_code,view_count,"
            "like_count,comment_count,share_count,favorites_count,video_duration,"
            "hashtag_names,music_id,engagement_rate,updated_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                vid["video_id"], username, vid["description"], vid["create_time"],
                vid["region_code"], vid["view_count"], vid["like_count"],
                vid["comment_count"], vid["share_count"], vid["favorites_count"],
                vid["video_duration"], json.dumps(vid["hashtag_names"]),
                vid["music_id"], vid["engagement_rate"], now,
            ),
        )
    conn.commit()
    conn.close()
    return videos


async def _search_videos(
    keyword: str = "",
    hashtag: str = "",
    region: str = "",
    count: int = 20,
    days_back: int = 30,
) -> list[dict]:
    """Search videos by keyword/hashtag/region."""
    start_date, end_date = _date_range(days_back)
    conditions = []
    if keyword:
        conditions.append({"operation": "EQ", "field_name": "keyword", "field_values": [keyword]})
    if hashtag:
        conditions.append({"operation": "EQ", "field_name": "hashtag_name", "field_values": [hashtag.lstrip("#")]})
    if region:
        conditions.append({"operation": "EQ", "field_name": "region_code", "field_values": [region.upper()]})
    if not conditions:
        raise ValueError("At least one of keyword, hashtag, or region is required.")

    body = {
        "query": {"and": conditions},
        "start_date": start_date,
        "end_date": end_date,
        "max_count": min(count, 100),
    }
    data = await _api_post("/research/video/query/", body, VIDEO_FIELDS)
    return data.get("videos", [])


# ---------------------------------------------------------------------------
# Metrics & Scoring
# ---------------------------------------------------------------------------

def _compute_metrics(profile: dict, videos: list[dict]) -> dict:
    """Derive performance metrics from profile + video data."""
    if not videos:
        return {}

    views = [v["view_count"] for v in videos]
    likes = [v["like_count"] for v in videos]
    comments = [v["comment_count"] for v in videos]
    shares = [v["share_count"] for v in videos]
    saves = [v["favorites_count"] for v in videos]
    ers = [v["engagement_rate"] for v in videos if v["engagement_rate"] > 0]

    dates = sorted([v["create_time"] for v in videos if v["create_time"]])
    if len(dates) >= 2:
        first = datetime.fromisoformat(dates[0])
        last = datetime.fromisoformat(dates[-1])
        span = max((last - first).days, 1)
        posts_per_week = round(len(dates) / span * 7, 1)
    else:
        posts_per_week = 0

    all_tags: list[str] = []
    for v in videos:
        all_tags.extend(v["hashtag_names"])
    tag_freq: dict[str, int] = {}
    for h in all_tags:
        tag_freq[h] = tag_freq.get(h, 0) + 1
    top_hashtags = sorted(tag_freq.items(), key=lambda x: x[1], reverse=True)[:10]

    avg_view = mean(views) if views else 0
    viral_count = sum(1 for p in views if p > avg_view * 2) if avg_view else 0
    followers = max(profile.get("follower_count", 1), 1)

    return {
        "avg_views": round(mean(views)) if views else 0,
        "median_views": round(sorted(views)[len(views) // 2]) if views else 0,
        "max_views": max(views, default=0),
        "min_views": min(views, default=0),
        "total_views": sum(views),
        "avg_likes": round(mean(likes)) if likes else 0,
        "avg_comments": round(mean(comments)) if comments else 0,
        "avg_shares": round(mean(shares)) if shares else 0,
        "avg_saves": round(mean(saves)) if saves else 0,
        "avg_engagement_rate": round(mean(ers), 2) if ers else 0,
        "engagement_std": round(stdev(ers), 2) if len(ers) > 1 else 0,
        "posts_per_week": posts_per_week,
        "videos_analyzed": len(videos),
        "top_hashtags": top_hashtags,
        "viral_ratio": round(viral_count / len(videos) * 100, 1),
        "views_to_follower": round(avg_view / followers * 100, 1),
        "likes_per_follower": round(profile.get("likes_count", 0) / followers, 1),
    }


def _score_kol(profile: dict, metrics: dict) -> dict:
    """Score a KOL on 5 dimensions (0-100) → weighted overall."""
    avg_views = metrics.get("avg_views", 0)
    if avg_views >= 1_000_000:   reach = 100
    elif avg_views >= 500_000:   reach = 90
    elif avg_views >= 100_000:   reach = 75
    elif avg_views >= 50_000:    reach = 60
    elif avg_views >= 10_000:    reach = 45
    elif avg_views >= 5_000:     reach = 30
    elif avg_views >= 1_000:     reach = 20
    else:                        reach = 10

    e = metrics.get("avg_engagement_rate", 0)
    if e >= 10:   engagement = 100
    elif e >= 7:  engagement = 90
    elif e >= 5:  engagement = 80
    elif e >= 3:  engagement = 65
    elif e >= 2:  engagement = 50
    elif e >= 1:  engagement = 35
    else:         engagement = 15

    ppw = metrics.get("posts_per_week", 0)
    if ppw >= 5:   freq = 100
    elif ppw >= 3: freq = 80
    elif ppw >= 1: freq = 50
    else:          freq = 20
    e_std = metrics.get("engagement_std", 0)
    consistency = max(freq - int(e_std * 5), 10)

    vr = metrics.get("viral_ratio", 0)
    if vr >= 30:   virality = 100
    elif vr >= 20: virality = 80
    elif vr >= 10: virality = 60
    else:          virality = max(int(vr * 4), 10)

    lpf = metrics.get("likes_per_follower", 0)
    if lpf >= 50:   growth = 100
    elif lpf >= 20: growth = 80
    elif lpf >= 10: growth = 60
    elif lpf >= 5:  growth = 40
    else:           growth = 20

    overall = round(
        reach * 0.25 + engagement * 0.30 + consistency * 0.15
        + virality * 0.15 + growth * 0.15
    )
    return {
        "overall": overall,
        "reach": reach,
        "engagement": engagement,
        "consistency": consistency,
        "virality": virality,
        "growth_potential": growth,
    }


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def _format_profile(p: dict) -> str:
    return (
        f"@{p['username']} ({p['display_name']})\n"
        f"  Bio: {p['bio'][:200]}\n"
        f"  Verified: {'Yes' if p['verified'] else 'No'} | Tier: {p['tier']}\n"
        f"  Followers: {_fmt(p['follower_count'])} | Following: {_fmt(p['following_count'])}\n"
        f"  Total Likes: {_fmt(p['likes_count'])} | Videos: {_fmt(p['video_count'])}"
    )


def _format_video(v: dict, idx: int) -> str:
    tags = " ".join(f"#{h}" for h in v["hashtag_names"][:5])
    return (
        f"\n  [{idx}] {v['description'][:100]}\n"
        f"      Views: {_fmt(v['view_count'])} | Likes: {_fmt(v['like_count'])} | "
        f"Comments: {_fmt(v['comment_count'])} | Shares: {_fmt(v['share_count'])} | "
        f"Saves: {_fmt(v['favorites_count'])}\n"
        f"      ER: {v['engagement_rate']}% | Duration: {v['video_duration']}s | "
        f"Posted: {v['create_time'][:10] if v['create_time'] else '?'}\n"
        f"      {tags}"
    )


# ---------------------------------------------------------------------------
# LLM evaluation
# ---------------------------------------------------------------------------

def _llm_evaluate(profile: dict, metrics: dict, scores: dict) -> str:
    """GPT-4o-mini campaign evaluation summary in Vietnamese."""
    prompt = (
        "You are a KOL/influencer marketing expert for the Vietnam market.\n"
        "Analyze this TikTok KOL and provide a concise evaluation.\n\n"
        f"Profile: @{profile['username']} ({profile['display_name']})\n"
        f"  Bio: {profile['bio'][:300]}\n"
        f"  Tier: {profile['tier']} | Followers: {_fmt(profile['follower_count'])}\n"
        f"  Total Likes: {_fmt(profile['likes_count'])}\n\n"
        f"Performance (last {metrics.get('videos_analyzed', 0)} videos):\n"
        f"  Avg Views: {_fmt(metrics.get('avg_views', 0))}\n"
        f"  Avg ER: {metrics.get('avg_engagement_rate', 0)}%\n"
        f"  Posts/week: {metrics.get('posts_per_week', 0)}\n"
        f"  Viral ratio: {metrics.get('viral_ratio', 0)}%\n\n"
        f"Scores: Overall {scores['overall']}, Reach {scores['reach']}, "
        f"Engagement {scores['engagement']}, Consistency {scores['consistency']}, "
        f"Virality {scores['virality']}, Growth {scores['growth_potential']}\n\n"
        "Provide in Vietnamese:\n"
        "1. Tong quan (1-2 cau)\n"
        "2. Diem manh (2-3 bullets)\n"
        "3. Rui ro / Diem yeu (2-3 bullets)\n"
        "4. Loai campaign phu hop\n"
        "5. De xuat (hop tac / khong / can them data)\n"
    )
    resp = _get_openai().chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.3,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Tool functions (registered by research_mcp.py)
# ---------------------------------------------------------------------------

async def get_kol_profile(username: str):
    """
    Fetch a TikTok KOL/KOC profile with key metrics: followers, total likes,
    video count, tier classification, bio, and verification status.
    Data is cached in SQLite for later comparison.
    ALWAYS answer in Vietnamese or English.

    Args:
        username: TikTok username (without @), e.g. "therock"
    """
    try:
        profile = await _fetch_profile(username)
    except Exception as e:
        return f"Failed to fetch @{username}: {e}"
    return _format_profile(profile)


async def get_kol_videos(username: str, count: int = 20):
    """
    Fetch recent TikTok videos of a KOL/KOC with performance metrics per video:
    views, likes, comments, shares, saves, engagement rate, duration, hashtags.
    Covers the last 30 days of content.
    ALWAYS answer in Vietnamese or English.

    Args:
        username: TikTok username (without @)
        count: Number of recent videos to fetch (default 20, max 100)
    """
    count = min(max(count, 1), 100)
    try:
        videos = await _fetch_videos(username, count)
    except Exception as e:
        return f"Failed to fetch videos for @{username}: {e}"
    if not videos:
        return f"No videos found for @{username} in the last 30 days."
    lines = [f"Recent {len(videos)} videos for @{username}:\n"]
    for i, v in enumerate(videos, 1):
        lines.append(_format_video(v, i))
    return "\n".join(lines)


async def analyze_kol(username: str, video_count: int = 30):
    """
    Deep analysis of a TikTok KOL/KOC: fetches profile + recent videos, computes
    engagement metrics, scores on 5 dimensions (reach, engagement, consistency,
    virality, growth), and generates an AI-powered evaluation with campaign
    recommendations for the Vietnam market.
    ALWAYS answer in Vietnamese or English.

    Args:
        username: TikTok username (without @)
        video_count: How many recent videos to analyze (default 30, max 100)
    """
    video_count = min(max(video_count, 5), 100)
    try:
        profile = await _fetch_profile(username)
        videos = await _fetch_videos(username, video_count)
    except Exception as e:
        return f"Failed to analyze @{username}: {e}"

    if not videos:
        return f"No videos found for @{username} in the last 30 days — cannot analyze."

    metrics = _compute_metrics(profile, videos)
    scores = _score_kol(profile, metrics)

    top_tags = ", ".join(f"#{h} ({c})" for h, c in metrics.get("top_hashtags", [])[:8])

    lines = [
        f"=== KOL Analysis: @{username} ===\n",
        _format_profile(profile),
        f"\n--- Performance Metrics (last {len(videos)} videos) ---",
        f"  Avg Views: {_fmt(metrics['avg_views'])} | Median: {_fmt(metrics['median_views'])}",
        f"  Max Views: {_fmt(metrics['max_views'])} | Min: {_fmt(metrics['min_views'])}",
        f"  Avg Likes: {_fmt(metrics['avg_likes'])} | Avg Comments: {_fmt(metrics['avg_comments'])}",
        f"  Avg Shares: {_fmt(metrics['avg_shares'])} | Avg Saves: {_fmt(metrics['avg_saves'])}",
        f"  Avg Engagement Rate: {metrics['avg_engagement_rate']}% (+/-{metrics['engagement_std']}%)",
        f"  Posting Frequency: {metrics['posts_per_week']} videos/week",
        f"  Viral Ratio: {metrics['viral_ratio']}% (videos > 2x avg views)",
        f"  Views/Follower: {metrics['views_to_follower']}%",
        f"  Top Hashtags: {top_tags}",
        f"\n--- Scores (0-100) ---",
        f"  Overall:     {scores['overall']}/100",
        f"  Reach:       {scores['reach']}/100",
        f"  Engagement:  {scores['engagement']}/100",
        f"  Consistency: {scores['consistency']}/100",
        f"  Virality:    {scores['virality']}/100",
        f"  Growth:      {scores['growth_potential']}/100",
    ]

    if OPENAI_API_KEY:
        try:
            summary = await asyncio.to_thread(_llm_evaluate, profile, metrics, scores)
            lines.append(f"\n--- AI Evaluation ---\n{summary}")
        except Exception as e:
            logger.warning(f"[tiktok] LLM evaluation failed: {e}")

    return "\n".join(lines)


async def compare_kols(usernames: str):
    """
    Compare multiple TikTok KOLs/KOCs side by side on all metrics and scores.
    Fetches fresh data for each, then ranks them by reach, engagement, and overall.
    ALWAYS answer in Vietnamese or English.

    Args:
        usernames: Comma-separated TikTok usernames (2-5), e.g. "user1,user2,user3"
    """
    names = [u.strip().lstrip("@") for u in usernames.split(",") if u.strip()]
    if len(names) < 2:
        return "Please provide at least 2 usernames separated by commas."
    if len(names) > 5:
        names = names[:5]

    results: list[dict] = []
    errors: list[str] = []

    for name in names:
        try:
            profile = await _fetch_profile(name)
            videos = await _fetch_videos(name, 30)
            metrics = _compute_metrics(profile, videos) if videos else {}
            scores = _score_kol(profile, metrics) if metrics else {
                "overall": 0, "reach": 0, "engagement": 0,
                "consistency": 0, "virality": 0, "growth_potential": 0,
            }
            results.append({
                "profile": profile, "metrics": metrics, "scores": scores,
                "video_count": len(videos),
            })
        except Exception as e:
            errors.append(f"@{name}: {e}")

    if not results:
        return "Failed to fetch any KOL data.\n" + "\n".join(errors)

    lines = ["=== KOL Comparison ===\n"]

    header = f"{'Metric':<25}"
    for r in results:
        header += f"{'@' + r['profile']['username']:<20}"
    lines.append(header)
    lines.append("-" * (25 + 20 * len(results)))

    rows = [
        ("Tier", lambda r: r["profile"]["tier"].split(" (")[0]),
        ("Followers", lambda r: _fmt(r["profile"]["follower_count"])),
        ("Total Likes", lambda r: _fmt(r["profile"]["likes_count"])),
        ("Videos", lambda r: str(r["profile"]["video_count"])),
        ("Verified", lambda r: "Yes" if r["profile"]["verified"] else "No"),
        ("Avg Views", lambda r: _fmt(r["metrics"].get("avg_views", 0))),
        ("Avg Likes", lambda r: _fmt(r["metrics"].get("avg_likes", 0))),
        ("Avg Comments", lambda r: _fmt(r["metrics"].get("avg_comments", 0))),
        ("Avg Shares", lambda r: _fmt(r["metrics"].get("avg_shares", 0))),
        ("Avg Saves", lambda r: _fmt(r["metrics"].get("avg_saves", 0))),
        ("Avg ER%", lambda r: f"{r['metrics'].get('avg_engagement_rate', 0)}%"),
        ("Posts/Week", lambda r: str(r["metrics"].get("posts_per_week", 0))),
        ("Viral Ratio", lambda r: f"{r['metrics'].get('viral_ratio', 0)}%"),
        ("Views/Follower", lambda r: f"{r['metrics'].get('views_to_follower', 0)}%"),
        ("-- Scores --", lambda r: ""),
        ("Overall", lambda r: f"{r['scores']['overall']}/100"),
        ("Reach", lambda r: f"{r['scores']['reach']}/100"),
        ("Engagement", lambda r: f"{r['scores']['engagement']}/100"),
        ("Consistency", lambda r: f"{r['scores']['consistency']}/100"),
        ("Virality", lambda r: f"{r['scores']['virality']}/100"),
        ("Growth", lambda r: f"{r['scores']['growth_potential']}/100"),
    ]

    for label, fn in rows:
        row = f"{label:<25}"
        for r in results:
            row += f"{fn(r):<20}"
        lines.append(row)

    lines.append("\n-- Winners --")
    best_reach = max(results, key=lambda r: r["scores"]["reach"])
    best_engage = max(results, key=lambda r: r["scores"]["engagement"])
    best_overall = max(results, key=lambda r: r["scores"]["overall"])
    best_value = max(results, key=lambda r: r["metrics"].get("avg_engagement_rate", 0))

    lines.append(f"  Best Reach:      @{best_reach['profile']['username']} ({best_reach['scores']['reach']}/100)")
    lines.append(f"  Best Engagement: @{best_engage['profile']['username']} ({best_engage['scores']['engagement']}/100)")
    lines.append(f"  Best Overall:    @{best_overall['profile']['username']} ({best_overall['scores']['overall']}/100)")
    lines.append(f"  Best Value (ER): @{best_value['profile']['username']} ({best_value['metrics'].get('avg_engagement_rate', 0)}%)")

    if errors:
        lines.append(f"\nErrors: {'; '.join(errors)}")

    return "\n".join(lines)


async def discover_kols(keyword: str, region: str = "VN", count: int = 10):
    """
    Discover TikTok KOLs by searching videos with a keyword in a specific region.
    Finds unique creators from matching videos and fetches their profiles.
    ALWAYS answer in Vietnamese or English.

    Args:
        keyword: Search keyword, e.g. "beauty tips", "fintech", "review san pham"
        region: Country code (default "VN" for Vietnam). Use "US", "JP", etc.
        count: Max number of KOLs to discover (default 10)
    """
    try:
        raw_videos = await _search_videos(keyword=keyword, region=region, count=min(count * 3, 100))
    except Exception as e:
        return f"Search failed: {e}"

    if not raw_videos:
        return f"No videos found for '{keyword}' in region {region}."

    # Extract unique usernames, ordered by view_count (most popular first)
    seen: set[str] = set()
    unique_users: list[str] = []
    sorted_vids = sorted(raw_videos, key=lambda v: v.get("view_count", 0), reverse=True)
    for v in sorted_vids:
        u = v.get("username", "")
        if u and u not in seen:
            seen.add(u)
            unique_users.append(u)
        if len(unique_users) >= count:
            break

    # Fetch profiles for each discovered user
    lines = [f"Discovered {len(unique_users)} KOLs for '{keyword}' in {region}:\n"]
    for i, uname in enumerate(unique_users, 1):
        try:
            profile = await _fetch_profile(uname)
            v = "Yes" if profile["verified"] else "No"
            lines.append(
                f"  {i}. @{uname} ({profile['display_name']}) — {profile['tier']}\n"
                f"     Bio: {profile['bio'][:100]}\n"
                f"     Followers: {_fmt(profile['follower_count'])} | "
                f"Likes: {_fmt(profile['likes_count'])} | "
                f"Videos: {profile['video_count']} | Verified: {v}"
            )
        except Exception as e:
            lines.append(f"  {i}. @{uname} — (profile fetch failed: {e})")

    return "\n".join(lines)


async def get_hashtag_performance(hashtag: str, region: str = "", count: int = 20):
    """
    Analyze a TikTok hashtag's performance by fetching recent videos using it.
    Shows total/avg engagement, top creators, and content stats.
    ALWAYS answer in Vietnamese or English.

    Args:
        hashtag: Hashtag name (without #), e.g. "fintechvietnam"
        region: Filter by country code, e.g. "VN" (optional)
        count: Number of videos to analyze (default 20, max 100)
    """
    hashtag = hashtag.lstrip("#")
    count = min(max(count, 5), 100)
    try:
        raw_videos = await _search_videos(hashtag=hashtag, region=region, count=count)
    except Exception as e:
        return f"Failed to fetch #{hashtag}: {e}"

    if not raw_videos:
        return f"No videos found for #{hashtag}."

    total_views = sum(v.get("view_count", 0) for v in raw_videos)
    total_likes = sum(v.get("like_count", 0) for v in raw_videos)
    total_comments = sum(v.get("comment_count", 0) for v in raw_videos)
    total_shares = sum(v.get("share_count", 0) for v in raw_videos)
    avg_views = total_views // len(raw_videos) if raw_videos else 0
    avg_er = _er(
        avg_views,
        total_likes // len(raw_videos),
        total_comments // len(raw_videos),
        total_shares // len(raw_videos),
    )

    # Top creators
    creator_views: dict[str, int] = {}
    for v in raw_videos:
        u = v.get("username", "?")
        creator_views[u] = creator_views.get(u, 0) + v.get("view_count", 0)
    top_creators = sorted(creator_views.items(), key=lambda x: x[1], reverse=True)[:5]

    lines = [
        f"=== Hashtag: #{hashtag} ===\n",
        f"  Videos analyzed: {len(raw_videos)} (last 30 days)",
        f"  Total Views: {_fmt(total_views)} | Avg Views: {_fmt(avg_views)}",
        f"  Total Likes: {_fmt(total_likes)} | Total Comments: {_fmt(total_comments)}",
        f"  Total Shares: {_fmt(total_shares)} | Avg ER: {avg_er}%",
        f"\n  Top creators by views:",
    ]
    for i, (creator, views) in enumerate(top_creators, 1):
        lines.append(f"    {i}. @{creator} — {_fmt(views)} views")

    # Top video
    best = max(raw_videos, key=lambda v: v.get("view_count", 0))
    lines.append(
        f"\n  Top video: @{best.get('username', '?')}\n"
        f"    {best.get('video_description', '')[:100]}\n"
        f"    Views: {_fmt(best.get('view_count', 0))} | Likes: {_fmt(best.get('like_count', 0))}"
    )

    return "\n".join(lines)


async def get_video_comments(video_id: str, count: int = 20):
    """
    Fetch comments on a specific TikTok video. Useful for sentiment analysis
    and understanding audience reactions to KOL content.
    ALWAYS answer in Vietnamese or English.

    Args:
        video_id: TikTok video ID (from get_kol_videos results)
        count: Number of comments to fetch (default 20, max 100)
    """
    count = min(max(count, 1), 100)
    try:
        data = await _api_post(
            "/research/video/comment/list/",
            {"video_id": int(video_id), "max_count": count},
            COMMENT_FIELDS,
        )
    except Exception as e:
        return f"Failed to fetch comments: {e}"

    comments = data.get("comments", [])
    if not comments:
        return "No comments found for this video."

    lines = [f"Comments for video {video_id} ({len(comments)} fetched):\n"]
    for i, c in enumerate(comments, 1):
        ct = c.get("create_time", 0)
        date = datetime.fromtimestamp(ct, tz=timezone.utc).strftime("%Y-%m-%d") if ct else "?"
        lines.append(
            f"  {i}. [{date}] {c.get('text', '')} "
            f"(likes: {c.get('like_count', 0)}, replies: {c.get('reply_count', 0)})"
        )
    return "\n".join(lines)


def track_kol(username: str) -> str:
    """
    Add a TikTok KOL/KOC to the tracking list for ongoing monitoring.
    ALWAYS answer in Vietnamese or English.

    Args:
        username: TikTok username (without @)
    """
    username = username.lstrip("@")
    conn = sqlite3.connect(DB_PATH)
    existing = conn.execute(
        "SELECT username FROM tiktok_kols WHERE username = ?", (username,)
    ).fetchone()
    if existing:
        conn.execute("UPDATE tiktok_kols SET tracked = 1 WHERE username = ?", (username,))
    else:
        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "INSERT INTO tiktok_kols (username, display_name, bio, tier, tracked, raw_data, updated_at) "
            "VALUES (?, '', '', '', 1, '{}', ?)",
            (username, now),
        )
    conn.commit()
    conn.close()
    return f"@{username} added to tracking list. Use get_kol_profile to fetch latest data."


def untrack_kol(username: str) -> str:
    """
    Remove a TikTok KOL/KOC from the tracking list.
    ALWAYS answer in Vietnamese or English.

    Args:
        username: TikTok username (without @)
    """
    username = username.lstrip("@")
    conn = sqlite3.connect(DB_PATH)
    conn.execute("UPDATE tiktok_kols SET tracked = 0 WHERE username = ?", (username,))
    conn.commit()
    conn.close()
    return f"@{username} removed from tracking list."


def get_tracked_kols() -> str:
    """
    List all tracked TikTok KOLs/KOCs with their latest cached metrics.
    ALWAYS answer in Vietnamese or English.
    """
    conn = sqlite3.connect(DB_PATH)
    try:
        rows = conn.execute(
            "SELECT username, display_name, tier, follower_count, likes_count, "
            "video_count, verified, updated_at "
            "FROM tiktok_kols WHERE tracked = 1 ORDER BY follower_count DESC"
        ).fetchall()
    except sqlite3.OperationalError:
        rows = []
    conn.close()

    if not rows:
        return "No KOLs are being tracked. Use track_kol(username) to add one."

    lines = [f"Tracked KOLs ({len(rows)}):\n"]
    for username, nick, tier, followers, likes, vids, verified, updated in rows:
        v = " (verified)" if verified else ""
        lines.append(
            f"  @{username}{v} ({nick}) — {tier}\n"
            f"    Followers: {_fmt(followers)} | Total Likes: {_fmt(likes)} | "
            f"Videos: {vids} | Updated: {updated[:10] if updated else '?'}"
        )
    return "\n".join(lines)
