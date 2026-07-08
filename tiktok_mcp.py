"""TikTok KOL/KOC Intelligence — evaluate and compare influencers in Vietnam.

Uses RapidAPI tiktok-api23 proxy to fetch KOL profiles, videos, hashtags, and
search results. No Playwright/Chromium needed — pure HTTP API calls.

Endpoints used (tiktok-api23.p.rapidapi.com):
  GET /api/user/info          — profile by uniqueId
  GET /api/user/posts         — videos by secUid (paginated)
  GET /api/search/general     — search videos by keyword
  GET /api/challenge/info     — hashtag stats
  GET /api/challenge/posts    — videos under a hashtag
  GET /api/user/followers     — follower list
"""

import json
import logging
import os
import sqlite3
import time
from datetime import datetime, timezone
from statistics import mean, stdev

import httpx
from dotenv import load_dotenv

from rag_mcp import DB_PATH, OPENAI_API_KEY, _get_openai

load_dotenv()

logger = logging.getLogger(__name__)

RAPIDAPI_KEY = os.getenv("RAPIDAPI_TT_KEY", "") or os.getenv("RAPIDAPI_FB_KEY", "")
RAPIDAPI_HOST = "tiktok-api23.p.rapidapi.com"
RAPIDAPI_BASE = f"https://{RAPIDAPI_HOST}"


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def _tt_api(path: str, params: dict, timeout: int = 20, retries: int = 3) -> dict:
    """GET request to TikTok RapidAPI with retry on 429."""
    headers = {
        "x-rapidapi-key": RAPIDAPI_KEY,
        "x-rapidapi-host": RAPIDAPI_HOST,
    }
    with httpx.Client(timeout=timeout) as c:
        for attempt in range(retries):
            r = c.get(f"{RAPIDAPI_BASE}{path}", params=params, headers=headers)
            if r.status_code == 429:
                wait = 2 ** (attempt + 1)
                logger.info(f"TikTok API rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            if r.status_code == 204:
                return {}
            r.raise_for_status()
            return r.json()
    return {}


# ---------------------------------------------------------------------------
# SQLite (same schema as before)
# ---------------------------------------------------------------------------

def _init_tiktok_db():
    """Create TikTok tables (idempotent)."""
    conn = sqlite3.connect(DB_PATH)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS tiktok_kols (
            username        TEXT PRIMARY KEY,
            sec_uid         TEXT,
            nickname        TEXT,
            bio             TEXT,
            verified        INTEGER DEFAULT 0,
            follower_count  INTEGER DEFAULT 0,
            following_count INTEGER DEFAULT 0,
            heart_count     INTEGER DEFAULT 0,
            video_count     INTEGER DEFAULT 0,
            digg_count      INTEGER DEFAULT 0,
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
            play_count      INTEGER DEFAULT 0,
            digg_count      INTEGER DEFAULT 0,
            comment_count   INTEGER DEFAULT 0,
            share_count     INTEGER DEFAULT 0,
            collect_count   INTEGER DEFAULT 0,
            duration        INTEGER DEFAULT 0,
            hashtags        TEXT DEFAULT '[]',
            music_title     TEXT,
            engagement_rate REAL DEFAULT 0,
            raw_data        TEXT,
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


def _engagement_rate(plays: int, likes: int, comments: int, shares: int) -> float:
    if plays == 0:
        return 0.0
    return round((likes + comments + shares) / plays * 100, 2)


def _fmt(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def _parse_int(val) -> int:
    """Parse a value that might be str or int to int."""
    if val is None:
        return 0
    try:
        return int(val)
    except (ValueError, TypeError):
        return 0


# ---------------------------------------------------------------------------
# Data fetching via RapidAPI
# ---------------------------------------------------------------------------

def _fetch_profile(username: str) -> dict:
    """Fetch a user profile from TikTok API and cache in SQLite."""
    if not RAPIDAPI_KEY:
        raise RuntimeError("RAPIDAPI_TT_KEY not set. Set it to your RapidAPI key.")

    data = _tt_api("/api/user/info", {"uniqueId": username})
    if not data:
        raise RuntimeError(f"Empty response for @{username}")

    user_info = data.get("userInfo", {})
    ud = user_info.get("user", {})
    st = user_info.get("stats", {})

    if not ud.get("uniqueId"):
        raise RuntimeError(f"User @{username} not found.")

    profile = {
        "username": ud.get("uniqueId", username),
        "sec_uid": ud.get("secUid", ""),
        "nickname": ud.get("nickname", ""),
        "bio": ud.get("signature", ""),
        "verified": 1 if ud.get("verified") else 0,
        "follower_count": _parse_int(st.get("followerCount", 0)),
        "following_count": _parse_int(st.get("followingCount", 0)),
        "heart_count": _parse_int(st.get("heartCount", st.get("heart", 0))),
        "video_count": _parse_int(st.get("videoCount", 0)),
        "digg_count": _parse_int(st.get("diggCount", 0)),
        "avatar_url": ud.get("avatarLarger", ""),
        "tier": "",
    }
    profile["tier"] = _tier(profile["follower_count"])

    # Persist
    now = datetime.now(timezone.utc).isoformat()
    conn = sqlite3.connect(DB_PATH)
    existing = conn.execute(
        "SELECT tracked FROM tiktok_kols WHERE username = ?", (username,)
    ).fetchone()
    tracked = existing[0] if existing else 0
    conn.execute(
        "INSERT OR REPLACE INTO tiktok_kols "
        "(username,sec_uid,nickname,bio,verified,follower_count,following_count,"
        "heart_count,video_count,digg_count,avatar_url,tier,tracked,raw_data,updated_at) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            profile["username"], profile["sec_uid"], profile["nickname"],
            profile["bio"], profile["verified"], profile["follower_count"],
            profile["following_count"], profile["heart_count"],
            profile["video_count"], profile["digg_count"], profile["avatar_url"],
            profile["tier"], tracked,
            json.dumps(data, ensure_ascii=False, default=str), now,
        ),
    )
    conn.commit()
    conn.close()
    return profile


def _fetch_videos(username: str, sec_uid: str = "", count: int = 20) -> list[dict]:
    """Fetch recent videos for a user via RapidAPI and cache in SQLite."""
    if not RAPIDAPI_KEY:
        raise RuntimeError("RAPIDAPI_TT_KEY not set.")

    # Need secUid for user/posts — fetch profile first if not provided
    if not sec_uid:
        profile = _fetch_profile(username)
        sec_uid = profile.get("sec_uid", "")
    if not sec_uid:
        raise RuntimeError(f"Could not get secUid for @{username}")

    videos: list[dict] = []
    now = datetime.now(timezone.utc).isoformat()
    conn = sqlite3.connect(DB_PATH)
    cursor = "0"
    fetched = 0

    while fetched < count:
        batch = min(count - fetched, 30)
        data = _tt_api("/api/user/posts", {
            "secUid": sec_uid, "count": str(batch), "cursor": cursor,
        })
        if not data:
            break

        item_data = data.get("data", data)
        items = item_data.get("itemList", [])
        if not items:
            break

        for d in items:
            stats = d.get("statsV2", d.get("stats", {}))

            play = _parse_int(stats.get("playCount", 0))
            digg = _parse_int(stats.get("diggCount", 0))
            comment = _parse_int(stats.get("commentCount", 0))
            share = _parse_int(stats.get("shareCount", 0))
            collect = _parse_int(stats.get("collectCount", 0))

            hashtags = [
                t.get("hashtagName", "")
                for t in d.get("textExtra", [])
                if t.get("hashtagName")
            ]
            music = d.get("music", {})
            ct = d.get("createTime", 0)
            create_time = (
                datetime.fromtimestamp(int(ct), tz=timezone.utc).isoformat() if ct else ""
            )

            vid = {
                "video_id": str(d.get("id", "")),
                "username": username,
                "description": d.get("desc", ""),
                "create_time": create_time,
                "play_count": play,
                "digg_count": digg,
                "comment_count": comment,
                "share_count": share,
                "collect_count": collect,
                "duration": d.get("video", {}).get("duration", 0),
                "hashtags": hashtags,
                "music_title": music.get("title", ""),
                "engagement_rate": _engagement_rate(play, digg, comment, share),
            }
            videos.append(vid)

            conn.execute(
                "INSERT OR REPLACE INTO tiktok_videos "
                "(video_id,username,description,create_time,play_count,digg_count,"
                "comment_count,share_count,collect_count,duration,hashtags,"
                "music_title,engagement_rate,raw_data,updated_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    vid["video_id"], username, vid["description"], vid["create_time"],
                    play, digg, comment, share, collect, vid["duration"],
                    json.dumps(hashtags), vid["music_title"], vid["engagement_rate"],
                    json.dumps(d, ensure_ascii=False, default=str), now,
                ),
            )

        fetched += len(items)
        has_more = item_data.get("hasMore", False)
        next_cursor = str(item_data.get("cursor", ""))
        if not has_more or not next_cursor or next_cursor == cursor:
            break
        cursor = next_cursor
        time.sleep(0.5)

    conn.commit()
    conn.close()
    return videos


# ---------------------------------------------------------------------------
# Metrics & Scoring (unchanged logic)
# ---------------------------------------------------------------------------

def _compute_metrics(profile: dict, videos: list[dict]) -> dict:
    if not videos:
        return {}

    plays = [v["play_count"] for v in videos]
    diggs = [v["digg_count"] for v in videos]
    comments = [v["comment_count"] for v in videos]
    shares = [v["share_count"] for v in videos]
    saves = [v["collect_count"] for v in videos]
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
        all_tags.extend(v["hashtags"])
    tag_freq: dict[str, int] = {}
    for h in all_tags:
        tag_freq[h] = tag_freq.get(h, 0) + 1
    top_hashtags = sorted(tag_freq.items(), key=lambda x: x[1], reverse=True)[:10]

    avg_play = mean(plays) if plays else 0
    viral_count = sum(1 for p in plays if p > avg_play * 2) if avg_play else 0
    followers = max(profile.get("follower_count", 1), 1)

    return {
        "avg_views": round(mean(plays)) if plays else 0,
        "median_views": round(sorted(plays)[len(plays) // 2]) if plays else 0,
        "max_views": max(plays, default=0),
        "min_views": min(plays, default=0),
        "total_views": sum(plays),
        "avg_likes": round(mean(diggs)) if diggs else 0,
        "avg_comments": round(mean(comments)) if comments else 0,
        "avg_shares": round(mean(shares)) if shares else 0,
        "avg_saves": round(mean(saves)) if saves else 0,
        "avg_engagement_rate": round(mean(ers), 2) if ers else 0,
        "engagement_std": round(stdev(ers), 2) if len(ers) > 1 else 0,
        "posts_per_week": posts_per_week,
        "videos_analyzed": len(videos),
        "top_hashtags": top_hashtags,
        "viral_ratio": round(viral_count / len(videos) * 100, 1),
        "views_to_follower": round(avg_play / followers * 100, 1),
        "likes_per_follower": round(profile.get("heart_count", 0) / followers, 1),
    }


def _score_kol(profile: dict, metrics: dict) -> dict:
    avg_views = metrics.get("avg_views", 0)
    if avg_views >= 1_000_000:   reach = 100
    elif avg_views >= 500_000:   reach = 90
    elif avg_views >= 100_000:   reach = 75
    elif avg_views >= 50_000:    reach = 60
    elif avg_views >= 10_000:    reach = 45
    elif avg_views >= 5_000:     reach = 30
    elif avg_views >= 1_000:     reach = 20
    else:                        reach = 10

    er = metrics.get("avg_engagement_rate", 0)
    if er >= 10:   engagement = 100
    elif er >= 7:  engagement = 90
    elif er >= 5:  engagement = 80
    elif er >= 3:  engagement = 65
    elif er >= 2:  engagement = 50
    elif er >= 1:  engagement = 35
    else:          engagement = 15

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


def _format_profile(p: dict) -> str:
    return (
        f"@{p['username']} ({p['nickname']})\n"
        f"  Bio: {p['bio'][:200]}\n"
        f"  Verified: {'Yes' if p['verified'] else 'No'} | Tier: {p['tier']}\n"
        f"  Followers: {_fmt(p['follower_count'])} | Following: {_fmt(p['following_count'])}\n"
        f"  Total Likes: {_fmt(p['heart_count'])} | Videos: {_fmt(p['video_count'])}"
    )


def _format_video(v: dict, idx: int) -> str:
    tags = " ".join(f"#{h}" for h in v["hashtags"][:5])
    return (
        f"\n  [{idx}] {v['description'][:100]}\n"
        f"      Views: {_fmt(v['play_count'])} | Likes: {_fmt(v['digg_count'])} | "
        f"Comments: {_fmt(v['comment_count'])} | Shares: {_fmt(v['share_count'])} | "
        f"Saves: {_fmt(v['collect_count'])}\n"
        f"      ER: {v['engagement_rate']}% | Duration: {v['duration']}s | "
        f"Posted: {v['create_time'][:10] if v['create_time'] else '?'}\n"
        f"      {tags}"
    )


# ---------------------------------------------------------------------------
# LLM evaluation
# ---------------------------------------------------------------------------

def _llm_evaluate(profile: dict, metrics: dict, scores: dict) -> str:
    prompt = (
        "You are a KOL/influencer marketing expert for the Vietnam market.\n"
        "Analyze this TikTok KOL and provide a concise evaluation.\n\n"
        f"Profile: @{profile['username']} ({profile['nickname']})\n"
        f"  Bio: {profile['bio'][:300]}\n"
        f"  Tier: {profile['tier']} | Followers: {_fmt(profile['follower_count'])}\n"
        f"  Total Likes: {_fmt(profile['heart_count'])}\n\n"
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
# Tool functions (exported, registered by research_mcp.py)
# ---------------------------------------------------------------------------

def get_kol_profile(username: str) -> str:
    """
    Fetch a TikTok KOL/KOC profile with key metrics: followers, likes, video count,
    tier classification, and bio. Data is cached for comparison.
    ALWAYS answer in Vietnamese or English.

    Args:
        username: TikTok username (without @), e.g. "therock"
    """
    if not RAPIDAPI_KEY:
        return "Error: RAPIDAPI_TT_KEY not set."
    try:
        profile = _fetch_profile(username)
    except Exception as e:
        return f"Failed to fetch @{username}: {e}"
    return _format_profile(profile)


def get_kol_videos(username: str, count: int = 20) -> str:
    """
    Fetch recent TikTok videos of a KOL/KOC with performance metrics per video:
    views, likes, comments, shares, saves, engagement rate, duration, hashtags.
    ALWAYS answer in Vietnamese or English.

    Args:
        username: TikTok username (without @)
        count: Number of recent videos to fetch (default 20, max 50)
    """
    if not RAPIDAPI_KEY:
        return "Error: RAPIDAPI_TT_KEY not set."
    count = min(max(count, 1), 50)
    try:
        videos = _fetch_videos(username, count=count)
    except Exception as e:
        return f"Failed to fetch videos for @{username}: {e}"

    if not videos:
        return f"No videos found for @{username}."

    lines = [f"Recent {len(videos)} videos for @{username}:\n"]
    for i, v in enumerate(videos, 1):
        lines.append(_format_video(v, i))
    return "\n".join(lines)


def analyze_kol(username: str, video_count: int = 30) -> str:
    """
    Deep analysis of a TikTok KOL/KOC: fetches profile + recent videos, computes
    engagement metrics, scores on 5 dimensions (reach, engagement, consistency,
    virality, growth potential), and generates an AI-powered evaluation with
    campaign recommendations for the Vietnam market.
    ALWAYS answer in Vietnamese or English.

    Args:
        username: TikTok username (without @)
        video_count: How many recent videos to analyze (default 30)
    """
    if not RAPIDAPI_KEY:
        return "Error: RAPIDAPI_TT_KEY not set."
    video_count = min(max(video_count, 5), 50)
    try:
        profile = _fetch_profile(username)
        videos = _fetch_videos(username, sec_uid=profile.get("sec_uid", ""), count=video_count)
    except Exception as e:
        return f"Failed to analyze @{username}: {e}"

    if not videos:
        return f"No videos found for @{username} — cannot analyze."

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
            summary = _llm_evaluate(profile, metrics, scores)
            lines.append(f"\n--- AI Evaluation ---\n{summary}")
        except Exception as e:
            logger.warning(f"[tiktok] LLM evaluation failed: {e}")

    return "\n".join(lines)


def compare_kols(usernames: str) -> str:
    """
    Compare multiple TikTok KOLs/KOCs side by side on all metrics and scores.
    Fetches fresh data for each, then ranks them by reach, engagement, and overall.
    ALWAYS answer in Vietnamese or English.

    Args:
        usernames: Comma-separated TikTok usernames (2-5), e.g. "user1,user2,user3"
    """
    if not RAPIDAPI_KEY:
        return "Error: RAPIDAPI_TT_KEY not set."
    names = [u.strip().lstrip("@") for u in usernames.split(",") if u.strip()]
    if len(names) < 2:
        return "Please provide at least 2 usernames separated by commas."
    if len(names) > 5:
        names = names[:5]

    results: list[dict] = []
    errors: list[str] = []

    for name in names:
        try:
            profile = _fetch_profile(name)
            videos = _fetch_videos(name, sec_uid=profile.get("sec_uid", ""), count=30)
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
        time.sleep(0.5)

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
        ("Total Likes", lambda r: _fmt(r["profile"]["heart_count"])),
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


def search_tiktok_users(query: str, count: int = 10) -> str:
    """
    Search for TikTok users/KOLs by keyword. Uses general search and extracts
    unique authors. Useful for discovering KOLs in a specific niche.
    ALWAYS answer in Vietnamese or English.

    Args:
        query: Search query, e.g. "vietnam beauty", "fintech review"
        count: Number of results (default 10, max 30)
    """
    if not RAPIDAPI_KEY:
        return "Error: RAPIDAPI_TT_KEY not set."
    count = min(max(count, 1), 30)

    data = _tt_api("/api/search/general", {
        "keyword": query, "count": str(count), "cursor": "0",
    })
    items = data.get("item_list", [])

    if not items:
        return f"No results found for '{query}'."

    # Extract unique authors
    seen: set[str] = set()
    users: list[dict] = []
    for item in items:
        author = item.get("author", {})
        uid = author.get("uniqueId", "")
        if not uid or uid in seen:
            continue
        seen.add(uid)
        author_stats = item.get("authorStats", {})
        users.append({
            "username": uid,
            "nickname": author.get("nickname", ""),
            "bio": author.get("signature", "")[:100],
            "verified": author.get("verified", False),
            "followers": _parse_int(author_stats.get("followerCount", 0)),
            "likes": _parse_int(author_stats.get("heartCount", author_stats.get("heart", 0))),
            "videos": _parse_int(author_stats.get("videoCount", 0)),
        })

    if not users:
        return f"No users found for '{query}'."

    lines = [f"Found {len(users)} users for '{query}':\n"]
    for i, u in enumerate(users, 1):
        v = " (verified)" if u["verified"] else ""
        lines.append(
            f"  {i}. @{u['username']}{v} ({u['nickname']})\n"
            f"     {u['bio']}\n"
            f"     Followers: {_fmt(u['followers'])} | Likes: {_fmt(u['likes'])} | "
            f"Videos: {u['videos']} | Tier: {_tier(u['followers'])}"
        )
    return "\n".join(lines)


def get_hashtag_info(hashtag: str) -> str:
    """
    Get TikTok hashtag statistics and top videos. Useful for understanding
    hashtag performance and finding KOLs in specific niches.
    ALWAYS answer in Vietnamese or English.

    Args:
        hashtag: Hashtag name (without #), e.g. "fintechvietnam"
    """
    if not RAPIDAPI_KEY:
        return "Error: RAPIDAPI_TT_KEY not set."
    hashtag = hashtag.lstrip("#")

    # Get hashtag info
    data = _tt_api("/api/challenge/info", {"challengeName": hashtag})
    ch = data.get("challengeInfo", data)
    challenge = ch.get("challenge", ch)
    stats = ch.get("stats", data.get("stats", {}))

    challenge_id = challenge.get("id", "")
    lines = [
        f"=== Hashtag: #{challenge.get('title', hashtag)} ===\n",
        f"  Description: {challenge.get('desc') or 'N/A'}",
        f"  Views: {_fmt(_parse_int(stats.get('viewCount', 0)))}",
        f"  Videos: {_fmt(_parse_int(stats.get('videoCount', 0)))}",
    ]

    # Get top videos for this hashtag
    if challenge_id:
        try:
            vdata = _tt_api("/api/challenge/posts", {
                "challengeId": str(challenge_id), "count": "10", "cursor": "0",
            })
            vitems = vdata.get("itemList", [])
            if vitems:
                lines.append(f"\n  Top {len(vitems)} videos:")
                for i, v in enumerate(vitems, 1):
                    st = v.get("statsV2", v.get("stats", {}))
                    author = v.get("author", {})
                    plays = _parse_int(st.get("playCount", 0))
                    likes = _parse_int(st.get("diggCount", 0))
                    er = _engagement_rate(
                        plays, likes,
                        _parse_int(st.get("commentCount", 0)),
                        _parse_int(st.get("shareCount", 0)),
                    )
                    lines.append(
                        f"    {i}. @{author.get('uniqueId', '?')} — {v.get('desc', '')[:80]}\n"
                        f"       Views: {_fmt(plays)} | Likes: {_fmt(likes)} | ER: {er}%"
                    )
        except Exception as e:
            lines.append(f"\n  (Could not fetch top videos: {e})")

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
            "INSERT INTO tiktok_kols (username, nickname, bio, tier, tracked, raw_data, updated_at) "
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
            "SELECT username, nickname, tier, follower_count, heart_count, "
            "video_count, verified, updated_at "
            "FROM tiktok_kols WHERE tracked = 1 ORDER BY follower_count DESC"
        ).fetchall()
    except sqlite3.OperationalError:
        rows = []
    conn.close()

    if not rows:
        return "No KOLs are being tracked. Use track_kol(username) to add one."

    lines = [f"Tracked KOLs ({len(rows)}):\n"]
    for username, nick, tier, followers, hearts, vids, verified, updated in rows:
        v = " (verified)" if verified else ""
        lines.append(
            f"  @{username}{v} ({nick}) — {tier}\n"
            f"    Followers: {_fmt(followers)} | Total Likes: {_fmt(hearts)} | "
            f"Videos: {vids} | Updated: {updated[:10] if updated else '?'}"
        )
    return "\n".join(lines)
