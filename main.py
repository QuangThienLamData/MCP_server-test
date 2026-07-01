from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
import httpx
import research_mcp
from research_mcp import mcp as research_mcp_server, init_on_startup
import contextlib
import os
import uvicorn
from urllib.parse import urlparse

from src.auth import AuthMiddleware
from src.config import settings

BASE_URL = settings.SCALEKIT_RESOURCE_DOCS_URL.rsplit("/research/mcp", 1)[0]

RESEARCH_METADATA = {
    "resource": f"{BASE_URL}/research/mcp",
    "authorization_servers": [settings.SCALEKIT_AUTHORIZATION_SERVERS],
    "bearer_methods_supported": ["header"],
    "resource_documentation": f"{BASE_URL}/research/mcp/docs",
    "scopes_supported": ["search:read"],
}


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    async with contextlib.AsyncExitStack() as stack:
        await stack.enter_async_context(research_mcp_server.session_manager.run())
        print("Starting up MCP Server...")
        init_on_startup()
        yield
        print("Shutting down MCP Server...")

app = FastAPI(title="MCP Server", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/.well-known/oauth-protected-resource/research/mcp")
async def research_oauth_metadata():
    return RESEARCH_METADATA


ALLOWED_IMAGE_HOSTS = {"images.refero.design", "refero.design"}


@app.get("/proxy/image")
async def proxy_image(url: str = ""):
    """Fetch an image from an allowed host and re-serve it, bypassing CORS."""
    if not url:
        return JSONResponse(status_code=400, content={"error": "url parameter required"})
    host = urlparse(url).hostname or ""
    if host not in ALLOWED_IMAGE_HOSTS:
        return JSONResponse(status_code=403, content={"error": f"host '{host}' not allowed"})
    try:
        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as c:
            r = await c.get(url)
            r.raise_for_status()
        ct = r.headers.get("content-type", "image/png")
        return Response(
            content=r.content, media_type=ct,
            headers={"Cache-Control": "public, max-age=86400",
                     "Access-Control-Allow-Origin": "*"},
        )
    except Exception as e:
        return JSONResponse(status_code=502, content={"error": str(e)})


CRON_SECRET = os.getenv("CRON_SECRET", "")


@app.post("/internal/crawl")
async def internal_crawl(request: Request):
    """Trigger in-service crawls (competitors/news/reviews). Guarded by the X-Cron-Secret
    header, NOT Scalekit — meant to be called by a Render Cron Job."""
    if not CRON_SECRET or request.headers.get("X-Cron-Secret") != CRON_SECRET:
        return JSONResponse(status_code=401, content={"error": "unauthorized"})
    target = request.query_params.get("target", "all")
    try:
        last_days = int(request.query_params.get("last_days", "3"))
    except ValueError:
        last_days = 3

    out = {}
    if target in ("rag", "competitors", "all"):
        out["competitors"] = research_mcp.crawl_competitors()
    if target in ("news", "all"):
        out["news"] = research_mcp.crawl_news()
    if target in ("reviews", "all"):
        out["reviews"] = research_mcp.crawl_reviews(last_days=last_days)
    if not out:
        return JSONResponse(status_code=400, content={"error": f"unknown target '{target}'"})
    return {"triggered": target, "results": out}


app.add_middleware(AuthMiddleware)

app.mount("/research", research_mcp_server.streamable_http_app(), name="Research MCP Server")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port, proxy_headers=True, forwarded_allow_ips="*")
