import logging
import os

import httpx
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
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

mcp = FastMCP(
    name="Refero UX MCP Server",
    transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
)


def _platform(p: str) -> str:
    p = (p or "web").lower()
    return p if p in ("ios", "web") else "web"


def _english_query(q: str) -> str:
    """Refero search is English; translate Vietnamese queries to English for better hits."""
    if _detect_lang(q) == "en":
        return q
    for v in _bilingual_queries(q):
        if _detect_lang(v) == "en":
            return v
    return q


async def _refero_call(tool_name: str, args: dict) -> str:
    """Call a Refero MCP tool via direct JSON-RPC over HTTP (avoids SDK streamablehttp_client
    hanging on Windows). Lazy-initialises the session on first call."""
    if not REFERO_TOKEN:
        return "Error: REFERO_TOKEN not set."
    global _refero_session_id, _refero_req_id

    headers = {
        "Authorization": f"Bearer {REFERO_TOKEN}",
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }
    if _refero_session_id:
        headers["Mcp-Session-Id"] = _refero_session_id

    try:
        async with httpx.AsyncClient(timeout=60) as c:
            # Lazy init: send initialize + notification on first call
            if not _refero_session_id:
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
                # notifications/initialized is fire-and-forget per MCP spec.
                # Refero responds with an SSE stream that never closes, which
                # poisons httpx's connection pool. Skip it — Refero works without.

            # Call the tool
            _refero_req_id += 1
            resp = await c.post(REFERO_MCP_URL, json={
                "jsonrpc": "2.0", "id": _refero_req_id, "method": "tools/call",
                "params": {"name": tool_name, "arguments": args},
            }, headers=headers)
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
    return text or "(no results)"


# --- MCP Tools (thin proxies over Refero) ---
@mcp.tool()
async def search_ux_patterns(query: str, platform: str = "web", page: int = 1) -> str:
    """
    Find UI/UX screen patterns from real shipped apps via Refero (135k+ screens).
    Query is auto-translated to English (Refero's search language). ALWAYS answer the user
    in Vietnamese or English.

    Args:
        query: What to look for, e.g. "onboarding flow fintech", "empty state payment"
        platform: 'web' or 'ios' (only these two are supported; default 'web')
        page: Result page (default 1)
    """
    return await _refero_call("refero_search_screens",
                              {"query": _english_query(query), "platform": _platform(platform), "page": page})


@mcp.tool()
async def search_user_flows(query: str, platform: str = "web", page: int = 1) -> str:
    """
    Search user flows (sequences of connected screens showing how users complete a task)
    from real apps via Refero. To compare apps, search a feature then inspect flows with
    get_ux_flow. ALWAYS answer the user in Vietnamese or English.

    Args:
        query: Flow to look for, e.g. "KYC flow", "payment confirmation"
        platform: 'web' or 'ios' (default 'web')
        page: Result page (default 1)
    """
    return await _refero_call("refero_search_flows",
                              {"query": _english_query(query), "platform": _platform(platform), "page": page})


@mcp.tool()
async def search_design_styles(query: str, page: int = 1) -> str:
    """
    Search Refero's curated design styles (visual/style references) by meaning.
    ALWAYS answer the user in Vietnamese or English.

    Args:
        query: Style to look for, e.g. "dark fintech dashboard"
        page: Result page (default 1)
    """
    return await _refero_call("refero_search_styles", {"query": _english_query(query), "page": page})


@mcp.tool()
async def get_ux_screen(screen_id: str) -> str:
    """
    Get full details of a Refero screen by its UUID (from a search result).

    Args:
        screen_id: The screen UUID
    """
    return await _refero_call("refero_get_screen", {"screen_id": screen_id})


@mcp.tool()
async def get_ux_flow(flow_id: int) -> str:
    """
    Get full details of a Refero user flow by its numeric id (from a search result).

    Args:
        flow_id: The flow id (number)
    """
    return await _refero_call("refero_get_flow", {"flow_id": flow_id})


if __name__ == "__main__":
    mcp.run(transport="stdio")
