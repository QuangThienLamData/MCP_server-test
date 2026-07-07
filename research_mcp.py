"""Unified Research MCP Server — combines Competitor Intelligence, Industry News,
App Reviews, UX Patterns, and YouTube Intelligence into a single MCP endpoint."""

import sqlite3
import threading
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

# Shared utilities (from the base RAG module)
from rag_mcp import (
    DB_PATH, OPENAI_API_KEY, PINECONE_API_KEY, TAVILY_API_KEY,
    _bilingual_queries, _embed, _get_index,
    _init_db, _load_competitors, auto_crawl_on_startup,
    _crawl_status as _competitor_crawl_status,
    _crawl_background as _competitor_crawl_bg,
    _tavily_search, _index_tavily_results, _format_web_results,
    FALLBACK_SCORE_THRESHOLD,
)

# News internals
from news_mcp import (
    GNEWS_API_KEY, NEWS_NAMESPACE,
    _news_crawl_status, _init_news_db,
    _crawl_news_background,
    _format_articles, _gnews_search, _gnews_top_headlines,
)

# Reviews internals
from reviews_mcp import (
    REVIEWS_NAMESPACE,
    _review_crawl_status, _init_reviews_db,
    _crawl_reviews_background,
    _review_insights_llm, _dist_str,
)

# UX (Refero) tools — imported and re-registered below
from refero_mcp import (
    search_ux_patterns, search_user_flows, search_design_styles,
    get_ux_screen, get_ux_flow,
)

# YouTube tools — imported and re-registered below
from youtube_mcp import (
    search_video_content, crawl_youtube_topic, get_video_transcript,
    list_indexed_videos, get_youtube_status,
)

# Knowledge Graph
from kg_extract import (
    _init_kg_db, search_kg, get_relationships, get_kg_stats,
)

# TikTok KOL Intelligence
from tiktok_mcp import (
    _init_tiktok_db,
    get_kol_profile, get_kol_videos, analyze_kol, compare_kols,
    search_tiktok_users, get_hashtag_info,
    track_kol, untrack_kol, get_tracked_kols,
)

load_dotenv()

mcp = FastMCP(
    name="Research MCP Server",
    transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
)


# ═══════════════════ Competitor Intelligence Tools ═══════════════════

@mcp.tool()
def search_competitor_content(
    query: str, competitor_name: str = "", source_type: str = "",
    date_from: str = "", date_to: str = "", top_k: int = 10,
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

    best = matches[0].score if matches else 0.0
    if best < FALLBACK_SCORE_THRESHOLD and TAVILY_API_KEY:
        try:
            web = _tavily_search(query)
            if web:
                _index_tavily_results(web, competitor_name)
                return _format_web_results(query, web)
        except Exception:
            pass

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

    # Enrich with Knowledge Graph context
    try:
        kg_entities = search_kg(query, top_k=5)
        if kg_entities:
            out.append("\n═══ Knowledge Graph Context ═══\n")
            for ent in kg_entities:
                out.append(f"• {ent['name']} ({ent['type']}): {ent['description']}")
                rels = get_relationships(ent["name"])
                if rels:
                    rel_strs = [f"{r['source']} → {r['target']}: {r['description']}" for r in rels[:3]]
                    out.append(f"  Relationships: {'; '.join(rel_strs)}")
                out.append("")
    except Exception:
        pass  # KG enrichment is best-effort

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
def crawl_competitors(competitor_name: str = "") -> str:
    """
    Manually start a competitor content crawl. Crawls all competitors or a specific one.

    Args:
        competitor_name: Only crawl this competitor (optional, all if empty)
    """
    if not PINECONE_API_KEY or not OPENAI_API_KEY:
        return "Error: PINECONE_API_KEY or OPENAI_API_KEY not set."
    if _competitor_crawl_status["running"]:
        return f"Competitor crawl already running: {_competitor_crawl_status['done']}/{_competitor_crawl_status['total']} sources."

    _init_db()
    _load_competitors()
    _competitor_crawl_status.update({
        "running": True, "total": 0, "done": 0, "current": "starting...",
        "errors": [], "new_articles": 0, "updated_articles": 0,
    })
    threading.Thread(target=_competitor_crawl_bg, args=(competitor_name,), daemon=True).start()
    target = competitor_name or "all competitors"
    return f"Competitor crawl started for {target}. Use get_research_status to check progress."


# ═══════════════════ Industry News Tools ═══════════════════

@mcp.tool()
def search_industry_trends(query: str, topic: str = "", from_date: str = "",
                           to_date: str = "", top_k: int = 5) -> str:
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
        return "No indexed news found. Run crawl_news first, or use search_news_realtime."

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

    # Enrich with Knowledge Graph context
    try:
        kg_entities = search_kg(query, top_k=5)
        if kg_entities:
            out.append("\n═══ Knowledge Graph Context ═══\n")
            for ent in kg_entities:
                out.append(f"• {ent['name']} ({ent['type']}): {ent['description']}")
                rels = get_relationships(ent["name"])
                if rels:
                    rel_strs = [f"{r['source']} → {r['target']}: {r['description']}" for r in rels[:3]]
                    out.append(f"  Relationships: {'; '.join(rel_strs)}")
                out.append("")
    except Exception:
        pass

    return "".join(out)


@mcp.tool()
def get_top_headlines(category: str = "technology", lang: str = "en",
                     country: str = "", max_results: int = 10) -> str:
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
        articles = _gnews_top_headlines(category=category, lang=lang,
                                        country=country or None, max_results=max_results)
    except Exception as e:
        return f"GNews error: {e}"
    return _format_articles(articles, f"Top headlines · {category}")


@mcp.tool()
def search_news_realtime(keyword: str, lang: str = "en", from_date: str = "",
                         max_results: int = 10) -> str:
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
        articles = _gnews_search(keyword, lang=lang, from_date=from_date or None,
                                  max_results=max_results)
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
    q = ("SELECT topic, title, source, article_url, published_at FROM news_articles "
         "WHERE COALESCE(published_at, crawled_at) >= ?")
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
        return f"No indexed news in the last {days_back} days. Run crawl_news first."

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
def crawl_news(topics: list[str] | None = None) -> str:
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
        "running": True, "total": 0, "done": 0, "current": "starting...",
        "errors": [], "new_articles": 0,
    })
    threading.Thread(target=_crawl_news_background, args=(topics or [],), daemon=True).start()
    target = ", ".join(topics) if topics else "all topics"
    return f"News crawl started for {target}. Use get_research_status to check progress."


# ═══════════════════ App Reviews Tools ═══════════════════

@mcp.tool()
def search_app_reviews(query: str, app: str = "", platform: str = "",
                       rating_max: int = 0, from_date: str = "", to_date: str = "",
                       top_k: int = 20) -> str:
    """
    Search user reviews (Google Play + App Store) about a specific issue, by meaning.
    Query and reviews may be in any language (searches EN+VI); ALWAYS answer the user in
    Vietnamese or English.

    Args:
        query: What to look for, e.g. "loi OTP", "giao dich cham"
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
        return "No reviews found. Run crawl_reviews first."

    out = [f"Found {len(matches)} reviews for: '{query}'\n"]
    for m in matches:
        md = m.metadata
        r = int(md.get("rating") or 0)
        out.append(
            f"\n--- [{md.get('app', '?')}/{md.get('platform', '?')}] "
            f"{'*' * r}{'.' * (5 - r)} {md.get('review_date', '')} (score {m.score:.3f}) ---\n"
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
        focus_area: Narrow to an area, e.g. "thanh toan", "onboarding" (optional)
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
        return f"No reviews for {app} in the last {days_back} days. Run crawl_reviews first."

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
def crawl_reviews(app: str = "", platform: str = "", last_days: int = 3) -> str:
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
        "running": True, "total": 0, "done": 0, "current": "starting...",
        "errors": [], "new_reviews": 0,
    })
    threading.Thread(target=_crawl_reviews_background, args=(app, platform, last_days),
                     daemon=True).start()
    target = f"{app or 'all apps'}{'/' + platform if platform else ''}"
    return f"Review crawl started for {target} (last {last_days} days). Use get_research_status to check."


# ═══════════════════ Unified Status Tool ═══════════════════

@mcp.tool()
def get_research_status() -> str:
    """Check crawl status and indexed content stats for all research modules
    (competitors, news, reviews)."""
    lines = []

    # Competitors
    lines.append("== Competitor Intelligence ==")
    if _competitor_crawl_status["running"]:
        lines.append(f"Crawl: IN PROGRESS ({_competitor_crawl_status['done']}/{_competitor_crawl_status['total']} sources)")
        lines.append(f"Current: {_competitor_crawl_status['current']}")
    elif _competitor_crawl_status["total"] > 0:
        lines.append(f"Crawl: COMPLETE ({_competitor_crawl_status['done']}/{_competitor_crawl_status['total']} sources)")
    else:
        lines.append("No crawl has run yet.")
    lines.append(f"New: {_competitor_crawl_status['new_articles']}, Updated: {_competitor_crawl_status['updated_articles']}")
    if _competitor_crawl_status["errors"]:
        for e in _competitor_crawl_status["errors"][-3:]:
            lines.append(f"  ! {e}")

    conn = sqlite3.connect(DB_PATH)
    try:
        sources = conn.execute(
            "SELECT cs.competitor_name, COUNT(cp.id) "
            "FROM crawl_sources cs LEFT JOIN crawled_pages cp ON cs.id = cp.source_id "
            "GROUP BY cs.competitor_name ORDER BY cs.competitor_name"
        ).fetchall()
    except sqlite3.OperationalError:
        sources = []
    if sources:
        for name, count in sources:
            lines.append(f"  {name}: {count} articles")

    # News
    lines.append("\n== Industry News ==")
    if _news_crawl_status["running"]:
        lines.append(f"Crawl: IN PROGRESS ({_news_crawl_status['done']}/{_news_crawl_status['total']} topics)")
        lines.append(f"Current: {_news_crawl_status['current']}")
    elif _news_crawl_status["total"] > 0:
        lines.append(f"Crawl: COMPLETE ({_news_crawl_status['done']}/{_news_crawl_status['total']} topics)")
    else:
        lines.append("No news crawl has run yet.")
    lines.append(f"New articles this run: {_news_crawl_status['new_articles']}")
    if _news_crawl_status["errors"]:
        for e in _news_crawl_status["errors"][-3:]:
            lines.append(f"  ! {e}")
    try:
        news_rows = conn.execute(
            "SELECT topic, COUNT(*) FROM news_articles GROUP BY topic ORDER BY topic"
        ).fetchall()
    except sqlite3.OperationalError:
        news_rows = []
    if news_rows:
        for topic, count in news_rows:
            lines.append(f"  {topic}: {count}")

    # Reviews
    lines.append("\n== App Reviews ==")
    if _review_crawl_status["running"]:
        lines.append(f"Crawl: IN PROGRESS ({_review_crawl_status['done']}/{_review_crawl_status['total']})")
        lines.append(f"Current: {_review_crawl_status['current']}")
    elif _review_crawl_status["total"] > 0:
        lines.append(f"Crawl: COMPLETE ({_review_crawl_status['done']}/{_review_crawl_status['total']})")
    else:
        lines.append("No review crawl has run yet.")
    lines.append(f"New reviews this run: {_review_crawl_status['new_reviews']}")
    if _review_crawl_status["errors"]:
        for e in _review_crawl_status["errors"][-3:]:
            lines.append(f"  ! {e}")
    try:
        rev_rows = conn.execute(
            "SELECT app, platform, COUNT(*), ROUND(AVG(rating), 2) FROM app_reviews "
            "GROUP BY app, platform ORDER BY app, platform"
        ).fetchall()
    except sqlite3.OperationalError:
        rev_rows = []
    if rev_rows:
        for app, platform, count, avg in rev_rows:
            lines.append(f"  {app}/{platform}: {count} reviews, avg {avg}")

    conn.close()
    return "\n".join(lines)


# ═══════════════════ Knowledge Graph Tools ═══════════════════

@mcp.tool()
def search_knowledge_graph(
    query: str, entity_type: str = "", top_k: int = 10,
) -> str:
    """
    Search the knowledge graph for entities (companies, products, features, strategies,
    technologies, people, etc.) extracted from competitor content.
    Returns semantically similar entities with their descriptions and relationships.
    ALWAYS answer in Vietnamese or English.

    Args:
        query: What to search for, e.g. "ZaloPay payment features", "MoMo partnerships"
        entity_type: Filter by type: company, product, feature, strategy, technology, person, market, partnership, metric, regulation (optional)
        top_k: Number of results (default 10)
    """
    if not PINECONE_API_KEY or not OPENAI_API_KEY:
        return "Error: PINECONE_API_KEY or OPENAI_API_KEY not set."
    try:
        results = search_kg(query, entity_type=entity_type, top_k=top_k)
    except Exception as e:
        return f"KG search error: {e}"
    if not results:
        return "No knowledge graph entities found for this query."
    lines = []
    for r in results:
        lines.append(f"• **{r['name']}** ({r['type']}) — score {r['score']:.2f}")
        lines.append(f"  {r['description']}")
        if r.get("competitor_names") and r["competitor_names"] != "[]":
            lines.append(f"  Competitors: {r['competitor_names']}")
        # Fetch relationships for this entity
        rels = get_relationships(r["name"])
        if rels:
            rel_strs = [f"{rel['source']} → {rel['target']}: {rel['description']}" for rel in rels[:3]]
            lines.append(f"  Relationships: {'; '.join(rel_strs)}")
        lines.append("")
    return "\n".join(lines)


@mcp.tool()
def get_entity_relationships(entity_name: str) -> str:
    """
    Get all relationships involving a specific entity from the knowledge graph.
    Shows how the entity connects to other companies, products, features, etc.
    ALWAYS answer in Vietnamese or English.

    Args:
        entity_name: The entity name to look up, e.g. "ZaloPay", "MoMo"
    """
    rels = get_relationships(entity_name)
    if not rels:
        return f"No relationships found for '{entity_name}' in the knowledge graph."
    lines = [f"Relationships for **{entity_name}** ({len(rels)} total):\n"]
    for r in rels:
        direction = "→" if r["source"].lower() == entity_name.lower() else "←"
        other = r["target"] if r["source"].lower() == entity_name.lower() else r["source"]
        lines.append(f"• {entity_name} {direction} **{other}** (strength {r['strength']}/10)")
        lines.append(f"  {r['description']}")
        if r.get("keywords"):
            lines.append(f"  Keywords: {r['keywords']}")
    return "\n".join(lines)


@mcp.tool()
def get_knowledge_graph_stats() -> str:
    """
    Get statistics about the knowledge graph: total entities, relationships, and type distribution.
    Useful for understanding how much competitive intelligence has been extracted.
    ALWAYS answer in Vietnamese or English.
    """
    stats = get_kg_stats()
    lines = [
        f"Knowledge Graph Statistics:",
        f"• Entities: {stats['entities']}",
        f"• Relationships: {stats['relationships']}",
    ]
    if stats.get("types"):
        lines.append("• Entity types:")
        for t, count in stats["types"].items():
            lines.append(f"  - {t}: {count}")
    return "\n".join(lines)


# ═══════════════════ UX + YouTube (re-register from their modules) ═══════════════════

for _fn in [
    search_ux_patterns, search_user_flows, search_design_styles, get_ux_screen, get_ux_flow,
    search_video_content, crawl_youtube_topic, get_video_transcript, list_indexed_videos, get_youtube_status,
    get_kol_profile, get_kol_videos, analyze_kol, compare_kols,
    search_tiktok_users, get_hashtag_info,
    track_kol, untrack_kol, get_tracked_kols,
]:
    mcp.tool()(_fn)


# ═══════════════════ Startup ═══════════════════

def init_on_startup():
    """Initialize all DBs and start auto-crawl."""
    _init_news_db()
    _init_reviews_db()
    _init_kg_db()
    _init_tiktok_db()
    auto_crawl_on_startup()


if __name__ == "__main__":
    _init_db()
    _load_competitors()
    _init_news_db()
    _init_reviews_db()
    _init_kg_db()
    _init_tiktok_db()
    mcp.run(transport="stdio")
