import asyncio
import logging
import os
import re

import httpx
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.utilities.types import Image
from mcp.server.transport_security import TransportSecuritySettings

# Reuse the shared cross-lingual helpers (Refero is an English design DB, so Vietnamese
# queries are translated to English before searching).
from rag_mcp import _bilingual_queries, _detect_lang

load_dotenv()

logger = logging.getLogger(__name__)

REFERO_MCP_URL = os.getenv("REFERO_MCP_URL", "https://api.refero.design/mcp")
REFERO_TOKEN = os.getenv("REFERO_TOKEN", "")
_refero_session_id: str | None = None
_refero_req_id = 0

_IMAGE_URL_RE = re.compile(r'https://images\.refero\.design/[^\s\)\]"\']+')
_MAX_INLINE = 3
_MAX_IMG_BYTES = 200_000  # skip images >200KB raw


async def _fetch_image(client: httpx.AsyncClient, url: str) -> Image | None:
    """Fetch one image → Image object or None on failure/oversize."""
    try:
        r = await client.get(url)
        r.raise_for_status()
        if len(r.content) > _MAX_IMG_BYTES:
            return None
        ct = r.headers.get("content-type", "image/png").split(";")[0].strip()
        fmt = ct.split("/")[-1]  # "jpeg", "png", etc.
        return Image(data=r.content, format=fmt)
    except Exception:
        return None


async def _with_image_blocks(text: str, max_images: int = _MAX_INLINE) -> list:
    """Return [cleaned_text, Image, Image, ...] — proper MCP ImageContent blocks.
    Strips inlined URLs from text so they don't appear twice."""
    urls = list(dict.fromkeys(_IMAGE_URL_RE.findall(text)))
    if not urls:
        return [text]
    to_fetch = urls[:max_images]
    async with httpx.AsyncClient(timeout=15, follow_redirects=True) as c:
        results = await asyncio.gather(*[_fetch_image(c, u) for u in to_fetch])
    # Strip fetched URLs from text (successfully fetched ones become image blocks)
    for url, img in zip(to_fetch, results):
        if img is not None:
            text = text.replace(url, "[see image below]")
    parts: list = [text]
    parts.extend(img for img in results if img is not None)
    return parts


def _strip_images(text: str) -> str:
    """Remove image URLs from text — used for flow responses where we want anatomy only."""
    return _IMAGE_URL_RE.sub("[image — open link in browser to view]", text)

mcp = FastMCP(
    name="Refero UX MCP Server",
    transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
)


def _platform(p: str) -> str:
    p = (p or "web").lower()
    return p if p in ("ios", "web") else "web"


async def _english_query(q: str) -> str:
    """Refero search is English; translate Vietnamese queries to English for better hits.
    Runs the sync OpenAI call in a thread to avoid blocking the event loop."""
    if _detect_lang(q) == "en":
        return q
    try:
        variants = await asyncio.to_thread(_bilingual_queries, q)
        for v in variants:
            if _detect_lang(v) == "en":
                return v
    except Exception as e:
        logger.warning(f"[refero] query translation failed: {e}")
    return q


async def _refero_call(tool_name: str, args: dict, *, inline_images: bool = False) -> str | list:
    """Call a Refero MCP tool via direct JSON-RPC over HTTP.
    inline_images=True → return [text, Image, Image, ...] as proper MCP content blocks.
    inline_images=False → strip image URLs (for flows with many screens)."""
    if not REFERO_TOKEN:
        return "Error: REFERO_TOKEN not set."
    global _refero_session_id, _refero_req_id

    logger.info(f"[refero] START {tool_name} args={args}")

    headers = {
        "Authorization": f"Bearer {REFERO_TOKEN}",
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }
    if _refero_session_id:
        headers["Mcp-Session-Id"] = _refero_session_id

    try:
        async with httpx.AsyncClient(timeout=60) as c:
            # Lazy init: send initialize on first call
            if not _refero_session_id:
                logger.info("[refero] initializing session...")
                _refero_req_id += 1
                init_resp = await c.post(REFERO_MCP_URL, json={
                    "jsonrpc": "2.0", "id": _refero_req_id, "method": "initialize",
                    "params": {"protocolVersion": "2025-03-26", "capabilities": {},
                               "clientInfo": {"name": "mcp-server-proxy", "version": "1.0.0"}},
                }, headers=headers)
                init_resp.raise_for_status()
                sid = init_resp.headers.get("mcp-session-id", "")
                if sid:
                    _refero_session_id = sid
                    headers["Mcp-Session-Id"] = sid
                logger.info(f"[refero] session initialized, id={_refero_session_id}")

            # Call the tool
            _refero_req_id += 1
            logger.info(f"[refero] calling tool {tool_name}...")
            resp = await c.post(REFERO_MCP_URL, json={
                "jsonrpc": "2.0", "id": _refero_req_id, "method": "tools/call",
                "params": {"name": tool_name, "arguments": args},
            }, headers=headers)
            logger.info(f"[refero] got response status={resp.status_code}")
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        # Reset session on error so next call re-initializes
        _refero_session_id = None
        logger.warning(f"Refero call failed [{tool_name}]: {e}")
        return f"Refero connection error: {e}"

    if "error" in data:
        _refero_session_id = None
        return f"Refero error: {data['error'].get('message', data['error'])}"

    result = data.get("result", {})
    contents = result.get("content", [])
    text = "\n".join(c["text"] for c in contents if c.get("type") == "text").strip()

    if "NO_SUBSCRIPTION" in text:
        return ("Refero subscription is not active/has expired. Activate at "
                "https://refero.design/mcp/upgrade, then retry.")
    if result.get("isError"):
        return f"Refero error: {text or '(unknown)'}"
    if not text:
        return "(no results)"
    if inline_images:
        return await _with_image_blocks(text)
    return _strip_images(text)


# --- MCP Tools (thin proxies over Refero) ---
@mcp.tool()
async def search_ux_patterns(query: str, platform: str = "web", page: int = 1):
    """
    Find UI/UX screen patterns from real shipped apps via Refero (135k+ screens).
    Returns up to 3 actual screenshots as inline images. ALWAYS answer in Vietnamese or English.

    Args:
        query: What to look for, e.g. "onboarding flow fintech", "empty state payment"
        platform: 'web' or 'ios' (only these two are supported; default 'web')
        page: Result page (default 1)
    """
    eq = await _english_query(query)
    return await _refero_call("refero_search_screens",
                              {"query": eq, "platform": _platform(platform), "page": page},
                              inline_images=True)


@mcp.tool()
async def search_user_flows(query: str, platform: str = "web", page: int = 1):
    """
    Search user flows (sequences of connected screens) from real apps via Refero.
    Returns flow metadata WITHOUT inline screenshots (flows have many screens).
    IMPORTANT: When presenting flow results, describe the SCREEN ANATOMY of each step —
    layout structure, key UI elements, navigation patterns, CTA placement — rather than
    trying to display images. ALWAYS answer in Vietnamese or English.

    Args:
        query: Flow to look for, e.g. "KYC flow", "payment confirmation"
        platform: 'web' or 'ios' (default 'web')
        page: Result page (default 1)
    """
    eq = await _english_query(query)
    return await _refero_call("refero_search_flows",
                              {"query": eq, "platform": _platform(platform), "page": page},
                              inline_images=False)


@mcp.tool()
async def search_design_styles(query: str, page: int = 1):
    """
    Search Refero's curated design styles (visual/style references) by meaning.
    Returns up to 3 actual screenshots as inline images. ALWAYS answer in Vietnamese or English.

    Args:
        query: Style to look for, e.g. "dark fintech dashboard"
        page: Result page (default 1)
    """
    eq = await _english_query(query)
    return await _refero_call("refero_search_styles", {"query": eq, "page": page},
                              inline_images=True)


@mcp.tool()
async def get_ux_screen(screen_id: str):
    """
    Get full details of a Refero screen by its UUID (from a search result).
    Returns the actual screenshot as an inline image. ALWAYS answer in Vietnamese or English.

    Args:
        screen_id: The screen UUID
    """
    return await _refero_call("refero_get_screen", {"screen_id": screen_id},
                              inline_images=True)


@mcp.tool()
async def get_ux_flow(flow_id: int):
    """
    Get full details of a Refero user flow by its numeric id (from a search result).
    Returns flow metadata WITHOUT inline screenshots (flows have many screens).
    IMPORTANT: When presenting flow results, describe the SCREEN ANATOMY of each step —
    layout structure, key UI elements, navigation patterns, CTA placement — rather than
    trying to display images. ALWAYS answer in Vietnamese or English.

    Args:
        flow_id: The flow id (number)
    """
    return await _refero_call("refero_get_flow", {"flow_id": flow_id},
                              inline_images=False)


if __name__ == "__main__":
    mcp.run(transport="stdio")
