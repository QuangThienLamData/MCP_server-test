import json
import logging
import os
import sqlite3
import threading
import time
from datetime import date, datetime, timedelta, timezone

import requests
from dotenv import load_dotenv
from google_play_scraper import Sort
from google_play_scraper import app as gp_app
from google_play_scraper import reviews as gp_reviews
from google_play_scraper import reviews_all as gp_reviews_all
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

# Reuse shared infra from the RAG module: same Pinecone index (namespace app_reviews),
# same OpenAI embedding + cross-lingual + language-detection helpers, same SQLite DB.
from rag_mcp import (
    DB_PATH, OPENAI_API_KEY, PINECONE_API_KEY,
    _bilingual_queries, _detect_lang, _embed, _get_index, _get_openai,
)

load_dotenv()

logger = logging.getLogger(__name__)

REVIEWS_NAMESPACE = "app_reviews"
APPS_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "apps.json")
VN_TZ = timezone(timedelta(hours=7))

GP_APP_ID = "com.mservice.momotransfer"
GP_SORT_MAP = {"newest": Sort.NEWEST, "rating": Sort.RATING, "relevance": Sort.MOST_RELEVANT}
GP_FIELDS = ["reviewId", "userName", "score", "content",
             "thumbsUpCount", "reviewCreatedVersion", "at",
             "replyContent", "repliedAt"]

IOS_APP_ID = 918751511
IOS_LIMIT = 20  # Apple gioi han toi da 20 review/lan goi (amp-api)
IOS_SORT_MAP = {"recent": "mostRecent", "helpful": "mostHelpful"}
IOS_HEADERS = {
    "accept": "*/*",
    "authorization": "Bearer",
    "origin": "https://apps.apple.com",
    "referer": "https://apps.apple.com/",
    "user-agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/148.0.0.0 Safari/537.36 Edg/148.0.0.0"),
    "x-apple-client-version": "2622.5.0-external",
}

mcp = FastMCP(
    name="App Reviews MCP Server",
    transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
)

_review_crawl_status = {
    "running": False, "total": 0, "done": 0, "current": "", "errors": [], "new_reviews": 0,
}


def parse_date(s):
    if not s:
        return None
    return datetime.strptime(s, "%Y-%m-%d").date()


def _load_apps() -> dict:
    if not os.path.exists(APPS_CONFIG):
        logger.warning(f"Apps config not found: {APPS_CONFIG}")
        return {}
    with open(APPS_CONFIG, encoding="utf-8") as f:
        return json.load(f)


# ============================ Google Play crawl ============================
def gp_in_range(dt, from_date, to_date):
    d = dt.date() if isinstance(dt, datetime) else dt
    if from_date and d < from_date:
        return False
    if to_date and d > to_date:
        return False
    return True


def gp_app_summary(app_id, lang, country):
    info = gp_app(app_id, lang=lang, country=country)
    return {
        "app_id": app_id, "title": info.get("title"),
        "score": info.get("score"), "ratings": info.get("ratings"),
        "reviews": info.get("reviews"),
        "fetched_at": datetime.now().isoformat(timespec="seconds"),
    }


def gp_get_reviews(app_id, lang, country, sort, count, score_filter, fetch_all,
                   from_date=None, to_date=None):
    """Lay review Google Play; sort=NEWEST co the dung som khi vuot moc from_date."""
    has_range = bool(from_date or to_date)
    if fetch_all:
        print("Dang lay TAT CA review (co the mat vai phut)...", flush=True)
        result = gp_reviews_all(app_id, lang=lang, country=country, sort=sort,
                                filter_score_with=score_filter, sleep_milliseconds=100)
        if has_range:
            result = [r for r in result if gp_in_range(r["at"], from_date, to_date)]
        return result

    collected, token = [], None
    batch_size = 200
    while len(collected) < count:
        want = min(batch_size, count - len(collected))
        result, token = gp_reviews(app_id, lang=lang, country=country, sort=sort,
                                   count=want, filter_score_with=score_filter,
                                   continuation_token=token)
        if not result:
            break
        if not has_range:
            collected.extend(result)
        else:
            stop = False
            for r in result:
                d = r["at"].date()
                if to_date and d > to_date:
                    continue
                if from_date and d < from_date:
                    stop = True
                    break
                collected.append(r)
                if len(collected) >= count:
                    break
            if stop:
                print(f"  Da toi moc {from_date}, dung. Tong {len(collected)} review.", flush=True)
                break
        print(f"  Da lay {len(collected)}/{count} review...", flush=True)
        if token is None:
            break
    return collected[:count]


def run_google_play(last_days=3, from_date=None, to_date=None, count=500,
                    sort="newest", score=None, fetch_all=False,
                    lang="vi", country="vn", app_id=GP_APP_ID):
    """Crawl Google Play. Tra ve list review (khong con import BQ)."""
    fd, td = parse_date(from_date), parse_date(to_date)
    if last_days and not fd and not td and not fetch_all:
        td = date.today()
        fd = td - timedelta(days=last_days - 1)
        print(f"[Auto] Lay {last_days} ngay gan nhat: {fd} -> {td}", flush=True)

    has_range = bool(fd or td)
    if has_range and sort != "newest":
        print(f"[Luu y] Loc theo ngay nen dung 'newest'; chuyen tu '{sort}'.")
        sort = "newest"
    if has_range and count == 500:
        count = 10**9

    print("== Google Play: thong tin tong quan ==", flush=True)
    s = gp_app_summary(app_id, lang, country)
    print(f"  {s['title']} | diem {s['score']} | rating {(s['ratings'] or 0):,} | review {(s['reviews'] or 0):,}")

    rows = gp_get_reviews(app_id, lang, country, GP_SORT_MAP[sort],
                          count, score, fetch_all, from_date=fd, to_date=td)
    print(f"Tong cong lay duoc {len(rows)} review.")
    return rows


# ============================ App Store crawl ============================
def ios_review_date(iso):
    """Phan ngay theo gio VN; xu ly ca '...Z' (UTC) lan '...-07:00' (RSS)."""
    s = iso.strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return datetime.strptime(iso[:10], "%Y-%m-%d").date()
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(VN_TZ).date()


def ios_date_ok(iso, from_date, to_date):
    if not iso:
        return False
    rd = ios_review_date(iso)
    if from_date and rd < from_date:
        return False
    if to_date and rd > to_date:
        return False
    return True


def ios_app_summary(app_id, country):
    r = requests.get("https://itunes.apple.com/lookup",
                     params={"id": app_id, "country": country},
                     headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    r.raise_for_status()
    res = r.json().get("results")
    if not res:
        raise RuntimeError(f"Khong tim thay app id={app_id} o country={country}")
    a = res[0]
    return {
        "app_id": app_id, "title": a.get("trackName"),
        "score": a.get("averageUserRating"), "ratings": a.get("userRatingCount"),
        "version": a.get("version"),
        "fetched_at": datetime.now().isoformat(timespec="seconds"),
    }


def ios_flatten(review_id, attr):
    dr = attr.get("developerResponse") or {}
    return {
        "reviewId": review_id, "userName": attr.get("userName"),
        "rating": attr.get("rating"), "title": attr.get("title"),
        "review": attr.get("review"), "date": attr.get("date"),
        "isEdited": attr.get("isEdited"),
        "developerResponse": dr.get("body"),
        "developerResponseAt": dr.get("modified"),
    }


def ios_get_reviews(app_id, country, sort, max_count,
                    from_date=None, to_date=None, sleep=0.7, max_scan=None):
    base = f"https://apps.apple.com/api/apps/v1/catalog/{country}/apps/{app_id}/reviews"
    sort_param = IOS_SORT_MAP[sort]
    has_range = bool(from_date or to_date)
    rows, seen = [], set()
    offset, scanned = 0, 0
    while len(rows) < max_count:
        if max_scan and scanned >= max_scan:
            print(f"  Da quet toi gioi han max_scan={max_scan}, dung.", flush=True)
            break
        params = {"l": "en-GB", "platform": "web", "limit": IOS_LIMIT,
                  "offset": offset, "sort": sort_param}
        data = None
        waits = [5, 10, 20, 40, 60, 60]
        for attempt in range(len(waits)):
            r = requests.get(base, headers=IOS_HEADERS, params=params, timeout=30)
            if r.status_code == 200:
                data = r.json()
                break
            if r.status_code in (429, 500, 502, 503):
                w = waits[attempt]
                print(f"  offset {offset}: HTTP {r.status_code} (bi chan toc do), cho {w}s...", flush=True)
                time.sleep(w)
                continue
            print(f"  offset {offset}: HTTP {r.status_code}, dung lai.", flush=True)
            return rows
        if data is None:
            print(f"  offset {offset}: van bi chan, dung lai. (da co {len(rows)} review)", flush=True)
            break

        items = data.get("data", [])
        if not items:
            print(f"  offset {offset}: het review.", flush=True)
            break

        for d in items:
            rid = d.get("id")
            attr = d.get("attributes", {})
            scanned += 1
            # NOTE: amp-api "mostRecent" is not strictly date-ordered, so we cannot early-break
            # on from_date (it would drop valid reviews). Bounded by max_scan instead.
            if has_range and not ios_date_ok(attr.get("date"), from_date, to_date):
                continue
            if rid in seen:
                continue
            seen.add(rid)
            rows.append(ios_flatten(rid, attr))
            if len(rows) >= max_count:
                break

        offset += IOS_LIMIT
        if offset % 400 == 0 or len(rows) >= max_count:
            if has_range:
                print(f"  Da quet {scanned} review, khop {len(rows)} (offset {offset})...", flush=True)
            else:
                print(f"  Da lay {len(rows)} review (offset {offset})...", flush=True)

        if "next" not in data:
            print(f"  Het review (da quet {scanned}).", flush=True)
            break
        time.sleep(sleep)
    return rows[:max_count]


def ios_rss_label(d, *path):
    for k in path:
        d = d.get(k) if isinstance(d, dict) else None
    return d.get("label") if isinstance(d, dict) else None


def ios_rss_flatten(e):
    rating = ios_rss_label(e, "im:rating")
    return {
        "reviewId": ios_rss_label(e, "id"),
        "userName": ios_rss_label(e, "author", "name"),
        "rating": int(rating) if rating and str(rating).isdigit() else None,
        "title": ios_rss_label(e, "title"),
        "review": ios_rss_label(e, "content"),
        "date": ios_rss_label(e, "updated"),
        "isEdited": None,
        "developerResponse": None,
        "developerResponseAt": None,
    }


def ios_rss_page(app_id, country, page):
    url = (f"https://itunes.apple.com/{country}/rss/customerreviews/"
           f"page={page}/id={app_id}/sortby=mostrecent/json")
    for w in (0, 5, 10, 20):
        if w:
            print(f"  RSS trang {page}: cho {w}s roi thu lai...", flush=True)
            time.sleep(w)
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
        if r.status_code == 200:
            feed = r.json().get("feed", {}) or {}
            entries = feed.get("entry", []) or []
            if isinstance(entries, dict):
                entries = [entries]
            return [e for e in entries if isinstance(e, dict) and "im:rating" in e]
        if r.status_code not in (429, 503):
            print(f"  RSS trang {page}: HTTP {r.status_code}, dung.", flush=True)
            return None
    print(f"  RSS trang {page}: van bi chan sau nhieu lan thu.", flush=True)
    return None


def ios_get_reviews_rss(app_id, country, from_date=None, to_date=None,
                        max_pages=10, sleep=0.5):
    """Tra ve (rows, reached). reached=True neu da cham review cu hon from_date
    HOAC het review; False = co the chua phu het."""
    rows, seen = [], set()
    reached = (from_date is None)
    for page in range(1, max_pages + 1):
        entries = ios_rss_page(app_id, country, page)
        if entries is None:
            break
        if not entries:
            print(f"  RSS trang {page}: het review.", flush=True)
            reached = True
            break
        stop = False
        for e in entries:
            rec = ios_rss_flatten(e)
            rd = ios_review_date(rec["date"]) if rec["date"] else None
            if from_date and rd and rd < from_date:
                reached, stop = True, True
                break
            if to_date and rd and rd > to_date:
                continue
            if not rec["reviewId"] or rec["reviewId"] in seen:
                continue
            seen.add(rec["reviewId"])
            rows.append(rec)
        print(f"  RSS trang {page}: tong khop {len(rows)} review...", flush=True)
        if stop:
            break
        time.sleep(sleep)
    return rows, reached


def run_appstore(last_days=3, from_date=None, to_date=None, count=1000,
                 sort="recent", country="vn", app_id=IOS_APP_ID,
                 sleep=0.7, max_scan=None, full_scan=False, rss_pages=10):
    """Crawl App Store. Tra ve list review (khong con import BQ)."""
    fd, td = parse_date(from_date), parse_date(to_date)
    if last_days and not fd and not td:
        td = date.today()
        fd = td - timedelta(days=last_days - 1)
        print(f"[Auto] Lay {last_days} ngay gan nhat: {fd} -> {td}", flush=True)

    has_range = bool(fd or td)
    max_count = count
    if has_range and count == 1000:
        max_count = 10**9

    print("== App Store: thong tin tong quan ==", flush=True)
    s = ios_app_summary(app_id, country)
    print(f"  {s['title']} | diem {s['score']} | rating {(s.get('ratings') or 0):,}")

    if has_range and not full_scan:
        print("  (dung RSS mostrecent - dung som theo ngay)", flush=True)
        rows, reached = ios_get_reviews_rss(app_id, country, fd, td,
                                            max_pages=rss_pages, sleep=sleep)
        if not rows:
            # Some storefronts (e.g. 'vn') return an empty RSS feed — fall back to amp-api,
            # bounded by max_scan so we don't scan the whole review history.
            print("  RSS rong; chuyen sang amp-api (gioi han max_scan).", flush=True)
            rows = ios_get_reviews(app_id, country, "recent", max_count,
                                   from_date=fd, to_date=td, sleep=sleep,
                                   max_scan=max_scan or 500)
        elif not reached:
            print(f"  [Canh bao] RSS chi co ~{rss_pages * 50} review gan nhat va chua cham "
                  f"moc {fd}; ket qua co the thieu. Dat full_scan=True neu can sau hon.", flush=True)
    else:
        rows = ios_get_reviews(app_id, country, sort, max_count,
                               from_date=fd, to_date=td, sleep=sleep, max_scan=max_scan)
    print(f"Tong cong lay duoc {len(rows)} review.")
    return rows


# ============================ Normalize + index ============================
def _normalize_gp(r: dict, app_name: str) -> dict:
    at = r.get("at")
    d = at.date().isoformat() if hasattr(at, "date") else (str(at)[:10] if at else "")
    return {
        "reviewId": str(r.get("reviewId", "")),
        "app": app_name, "platform": "android",
        "rating": int(r.get("score") or 0),
        "title": "",
        "text": (r.get("content") or "").strip(),
        "review_date": d,
        "version": r.get("reviewCreatedVersion") or "",
        "has_reply": bool(r.get("replyContent")),
        "thumbs_up": int(r.get("thumbsUpCount") or 0),
        "user_name": r.get("userName") or "",
    }


def _normalize_ios(r: dict, app_name: str) -> dict:
    ds = r.get("date")
    try:
        d = ios_review_date(ds).isoformat() if ds else ""
    except Exception:
        d = (ds or "")[:10]
    return {
        "reviewId": str(r.get("reviewId", "")),
        "app": app_name, "platform": "ios",
        "rating": int(r.get("rating") or 0),
        "title": r.get("title") or "",
        "text": (r.get("review") or "").strip(),
        "review_date": d,
        "version": "",
        "has_reply": bool(r.get("developerResponse")),
        "thumbs_up": 0,
        "user_name": r.get("userName") or "",
    }


def _init_reviews_db():
    conn = sqlite3.connect(DB_PATH)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS app_reviews (
            id TEXT PRIMARY KEY,
            app TEXT,
            platform TEXT,
            rating INTEGER,
            title TEXT,
            text TEXT,
            review_date TEXT,
            version TEXT,
            language TEXT,
            has_reply INTEGER,
            thumbs_up INTEGER,
            user_name TEXT,
            crawled_at TEXT NOT NULL
        );
    """)
    conn.commit()
    conn.close()


def _index_reviews(items: list[dict]) -> int:
    """Embed new reviews into Pinecone (namespace app_reviews) + record in SQLite.
    Dedup by id = {platform}_{app}_{reviewId}."""
    if not items:
        return 0
    conn = sqlite3.connect(DB_PATH)
    pending = []
    for it in items:
        rid = it["reviewId"]
        text = it["text"]
        if not rid or not text:
            continue
        vid = f"{it['platform']}_{it['app']}_{rid}"
        if conn.execute("SELECT 1 FROM app_reviews WHERE id = ?", (vid,)).fetchone():
            continue
        embed_text = f"{it['title']}. {text}".strip() if it["title"] else text
        if len(embed_text) < 10:
            continue
        it["_vid"] = vid
        it["_embed_text"] = embed_text
        pending.append(it)

    if not pending:
        conn.close()
        return 0

    try:
        embeddings = _embed([p["_embed_text"] for p in pending])
    except Exception as e:
        logger.error(f"Review embed failed: {e}")
        _review_crawl_status["errors"].append(f"Embed: {e}")
        conn.close()
        return 0

    now = datetime.now(timezone.utc).isoformat()
    vectors = []
    for p, emb in zip(pending, embeddings):
        lang = _detect_lang(p["_embed_text"])
        vectors.append({
            "id": p["_vid"], "values": emb,
            "metadata": {
                "app": p["app"], "platform": p["platform"], "rating": p["rating"],
                "review_date": p["review_date"], "version": p["version"],
                "language": lang, "has_reply": p["has_reply"],
                "title": p["title"], "text": p["text"][:2000],
            },
        })
        conn.execute(
            "INSERT OR IGNORE INTO app_reviews "
            "(id,app,platform,rating,title,text,review_date,version,language,has_reply,thumbs_up,user_name,crawled_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (p["_vid"], p["app"], p["platform"], p["rating"], p["title"], p["text"],
             p["review_date"], p["version"], lang, int(p["has_reply"]), p["thumbs_up"],
             p["user_name"], now),
        )

    index = _get_index()
    for b in range(0, len(vectors), 100):
        index.upsert(vectors=vectors[b:b + 100], namespace=REVIEWS_NAMESPACE)
    conn.commit()
    conn.close()
    return len(vectors)


def _crawl_reviews_background(app_filter: str, platform_filter: str, last_days: int):
    try:
        apps = _load_apps()
        targets = []
        for name, plats in apps.items():
            if app_filter and name.lower() != app_filter.lower():
                continue
            for plat, aid in plats.items():
                if platform_filter and plat != platform_filter:
                    continue
                targets.append((name, plat, aid))
        _review_crawl_status["total"] = len(targets)
        for i, (name, plat, aid) in enumerate(targets):
            _review_crawl_status["done"] = i
            _review_crawl_status["current"] = f"{name}/{plat}"
            try:
                if plat == "android":
                    raw = run_google_play(last_days=last_days, app_id=aid)
                    norm = [_normalize_gp(r, name) for r in raw]
                else:
                    raw = run_appstore(last_days=last_days, app_id=int(aid))
                    norm = [_normalize_ios(r, name) for r in raw]
                _review_crawl_status["new_reviews"] += _index_reviews(norm)
            except Exception as exc:
                logger.exception(f"Review crawl failed: {name}/{plat}")
                _review_crawl_status["errors"].append(f"{name}/{plat}: {type(exc).__name__}: {exc}")
        _review_crawl_status["done"] = len(targets)
        _review_crawl_status["current"] = ""
        logger.info(f"Review crawl done: {_review_crawl_status['new_reviews']} new, "
                    f"{len(_review_crawl_status['errors'])} errors")
    except Exception as e:
        logger.exception("Review crawl failed")
        _review_crawl_status["errors"].append(str(e))
    finally:
        _review_crawl_status["running"] = False


def _dist_str(dist: dict) -> str:
    return " ".join(f"{s}★:{dist.get(s, 0)}" for s in range(5, 0, -1))


def _review_insights_llm(app: str, dist: dict, samples: list[dict], focus_area: str) -> str:
    if not samples:
        return ""
    body = "\n".join(f"- ({r['rating']}*) {r['text'][:300]}" for r in samples[:40])
    focus = f" Focus specifically on: {focus_area}." if focus_area else ""
    try:
        resp = _get_openai().chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            messages=[
                {"role": "system", "content": (
                    "You analyze mobile app store reviews. Summarize the top user complaint "
                    "themes (with rough frequency) and concrete improvement suggestions. "
                    "Answer in Vietnamese, concise, bullet points."
                )},
                {"role": "user", "content": (
                    f"App: {app}. Rating distribution: {dist}.{focus}\n"
                    f"Recent low-rated reviews:\n{body}\n\n"
                    "Give: (1) top 5 complaint themes, (2) 3-5 improvement suggestions."
                )},
            ],
        )
        return resp.choices[0].message.content
    except Exception as e:
        logger.warning(f"Insight LLM failed: {e}")
        return ""


# ============================ MCP Tools ============================
@mcp.tool()
def search_app_reviews(query: str, app: str = "", platform: str = "", rating_max: int = 0,
                       from_date: str = "", to_date: str = "", top_k: int = 20) -> str:
    """
    Search user reviews (Google Play + App Store) about a specific issue, by meaning.
    Query and reviews may be in any language (searches EN+VI); ALWAYS answer the user in
    Vietnamese or English.

    Args:
        query: What to look for, e.g. "lỗi OTP", "giao dịch chậm"
        app: Filter by app: MoMo, ZaloPay, VNPay (optional)
        platform: Filter by platform: android, ios (optional)
        rating_max: Only reviews with rating <= this (e.g. 2 for complaints; 0 = no filter)
        from_date: From date YYYY-MM-DD (optional)
        to_date: To date YYYY-MM-DD (optional)
        top_k: Number of results (default 20)
    """
    if not PINECONE_API_KEY or not OPENAI_API_KEY:
        return "Error: PINECONE_API_KEY or OPENAI_API_KEY not set."

    variants = _bilingual_queries(query)
    try:
        qvecs = _embed(variants)
    except Exception as e:
        return f"Embedding error: {e}"

    filters: dict = {}
    if app:
        filters["app"] = {"$eq": app}
    if platform:
        filters["platform"] = {"$eq": platform}
    if rating_max:
        filters["rating"] = {"$lte": rating_max}
    if from_date and to_date:
        filters["review_date"] = {"$gte": from_date, "$lte": to_date}
    elif from_date:
        filters["review_date"] = {"$gte": from_date}
    elif to_date:
        filters["review_date"] = {"$lte": to_date}

    index = _get_index()
    merged: dict = {}
    for qvec in qvecs:
        res = index.query(vector=qvec, top_k=top_k, include_metadata=True,
                          namespace=REVIEWS_NAMESPACE, filter=filters or None)
        for m in res.matches:
            if m.id not in merged or m.score > merged[m.id].score:
                merged[m.id] = m
    matches = sorted(merged.values(), key=lambda m: m.score, reverse=True)[:top_k]
    if not matches:
        return "No reviews found. Run trigger_crawl first."

    # Build app store URL lookup from config
    apps_cfg = _load_apps()
    def _store_url(app_name: str, platform: str) -> str:
        ids = apps_cfg.get(app_name, {})
        if platform == "android" and ids.get("android"):
            return f"https://play.google.com/store/apps/details?id={ids['android']}&hl=vi"
        if platform == "ios" and ids.get("ios"):
            return f"https://apps.apple.com/vn/app/id{ids['ios']}"
        return ""

    out = [f"Found {len(matches)} reviews for: '{query}'\n"]
    for m in matches:
        md = m.metadata
        r = int(md.get("rating") or 0)
        app_name = md.get('app', '?')
        plat = md.get('platform', '?')
        link = _store_url(app_name, plat)
        out.append(
            f"\n--- [{app_name}/{plat}] "
            f"{'★' * r}{'☆' * (5 - r)} {md.get('review_date', '')} (score {m.score:.3f}) ---\n"
            f"App Store: {link}\n"
            f"{(md.get('title') + chr(10)) if md.get('title') else ''}{md.get('text', '')}\n"
        )
    return "".join(out)


@mcp.tool()
def get_review_insights(app: str, days_back: int = 30, focus_area: str = "") -> str:
    """
    Summarize the top user complaints and improvement suggestions for an app, from recent
    reviews. ALWAYS answer the user in Vietnamese or English.

    Args:
        app: App name: MoMo, ZaloPay, VNPay
        days_back: How many days back to analyze (default 30)
        focus_area: Narrow to an area, e.g. "thanh toán", "onboarding" (optional)
    """
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days_back)).strftime("%Y-%m-%d")
    conn = sqlite3.connect(DB_PATH)
    try:
        rows = conn.execute(
            "SELECT rating, title, text FROM app_reviews "
            "WHERE app = ? AND COALESCE(review_date, crawled_at) >= ? "
            "ORDER BY review_date DESC",
            (app, cutoff),
        ).fetchall()
    except sqlite3.OperationalError:
        rows = []
    conn.close()
    if not rows:
        return f"No reviews for {app} in the last {days_back} days. Run trigger_crawl first."

    dist = {}
    for rating, _, _ in rows:
        dist[rating] = dist.get(rating, 0) + 1
    total = len(rows)
    avg = sum((r or 0) * c for r, c in dist.items()) / total if total else 0
    negatives = [
        {"rating": r, "text": f"{t or ''} {x or ''}".strip()}
        for r, t, x in rows if r and r <= 3
    ]
    summary = _review_insights_llm(app, {f"{s}star": dist.get(s, 0) for s in range(5, 0, -1)},
                                   negatives, focus_area)
    head = (f"Insights for {app} (last {days_back} days): {total} reviews, avg {avg:.2f}\n"
            f"{_dist_str(dist)}")
    if summary:
        return f"{head}\n\n{summary}"
    return f"{head}\n\n({len(negatives)} low-rated reviews; LLM summary unavailable.)"


@mcp.tool()
def compare_rating_trend(apps: list[str], from_date: str = "", to_date: str = "") -> str:
    """
    Compare rating volume, average and star distribution across apps over a period.
    ALWAYS answer the user in Vietnamese or English.

    Args:
        apps: App names to compare, e.g. ["MoMo", "ZaloPay"]
        from_date: From date YYYY-MM-DD (optional)
        to_date: To date YYYY-MM-DD (optional)
    """
    conn = sqlite3.connect(DB_PATH)
    lines = [f"Rating comparison ({from_date or 'all'} -> {to_date or 'now'}):"]
    for app in apps:
        q = "SELECT rating, COUNT(*) FROM app_reviews WHERE app = ?"
        params: list = [app]
        if from_date:
            q += " AND review_date >= ?"
            params.append(from_date)
        if to_date:
            q += " AND review_date <= ?"
            params.append(to_date)
        q += " GROUP BY rating"
        try:
            rows = conn.execute(q, params).fetchall()
        except sqlite3.OperationalError:
            rows = []
        if not rows:
            lines.append(f"\n{app}: no reviews")
            continue
        dist = {r: c for r, c in rows}
        total = sum(dist.values())
        avg = sum((r or 0) * c for r, c in dist.items()) / total if total else 0
        lines.append(f"\n{app}: {total} reviews, avg {avg:.2f}\n  {_dist_str(dist)}")
    conn.close()
    return "\n".join(lines)


@mcp.tool()
def trigger_crawl(app: str = "", platform: str = "", last_days: int = 3) -> str:
    """
    Crawl the latest app-store reviews and index them. Meant to be called on a schedule by
    an external cron (e.g. a Render Cron Job), or manually. Runs in the background.

    Args:
        app: Only crawl this app: MoMo, ZaloPay, VNPay (optional, all if empty)
        platform: Only this platform: android, ios (optional, all available if empty)
        last_days: How many recent days to fetch (default 3)
    """
    if not PINECONE_API_KEY or not OPENAI_API_KEY:
        return "Error: PINECONE_API_KEY or OPENAI_API_KEY not set."
    if _review_crawl_status["running"]:
        return f"Review crawl already running: {_review_crawl_status['done']}/{_review_crawl_status['total']}."

    _init_reviews_db()
    _review_crawl_status.update({
        "running": True, "total": 0, "done": 0, "current": "starting...", "errors": [], "new_reviews": 0,
    })
    threading.Thread(target=_crawl_reviews_background, args=(app, platform, last_days), daemon=True).start()
    target = f"{app or 'all apps'}{'/' + platform if platform else ''}"
    return f"Review crawl started for {target} (last {last_days} days). Use get_crawl_status to check."


@mcp.tool()
def get_crawl_status() -> str:
    """Check the review crawler status and indexed review counts per app/platform."""
    lines = []
    if _review_crawl_status["running"]:
        lines.append(f"Review crawl: IN PROGRESS ({_review_crawl_status['done']}/{_review_crawl_status['total']})")
        lines.append(f"Current: {_review_crawl_status['current']}")
    elif _review_crawl_status["total"] > 0:
        lines.append(f"Review crawl: COMPLETE ({_review_crawl_status['done']}/{_review_crawl_status['total']})")
    else:
        lines.append("No review crawl has run yet.")
    lines.append(f"New reviews this run: {_review_crawl_status['new_reviews']}")

    if _review_crawl_status["errors"]:
        lines.append(f"Errors ({len(_review_crawl_status['errors'])}):")
        for e in _review_crawl_status["errors"][-5:]:
            lines.append(f"  - {e}")

    conn = sqlite3.connect(DB_PATH)
    try:
        rows = conn.execute(
            "SELECT app, platform, COUNT(*), ROUND(AVG(rating), 2) FROM app_reviews "
            "GROUP BY app, platform ORDER BY app, platform"
        ).fetchall()
    except sqlite3.OperationalError:
        rows = []
    conn.close()
    if rows:
        lines.append("\nIndexed:")
        for app, platform, count, avg in rows:
            lines.append(f"  {app}/{platform}: {count} reviews, avg {avg}")
    return "\n".join(lines)


if __name__ == "__main__":
    _init_reviews_db()
    mcp.run(transport="stdio")
