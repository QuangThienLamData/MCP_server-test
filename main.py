from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from gnews_mcp_server import mcp as gnews_mcp_server
from email_mcp import mcp as email_mcp_server
import contextlib
import os
import uvicorn

from src.auth import AuthMiddleware
from src.config import settings

BASE_URL = settings.SCALEKIT_RESOURCE_DOCS_URL.rsplit("/gnews/mcp", 1)[0]

GNEWS_METADATA = {
    "resource": f"{BASE_URL}/gnews/mcp",
    "authorization_servers": [settings.SCALEKIT_AUTHORIZATION_SERVERS],
    "bearer_methods_supported": ["header"],
    "resource_documentation": f"{BASE_URL}/gnews/mcp/docs",
    "scopes_supported": ["search:read"],
}

EMAIL_METADATA = {
    "resource": f"{BASE_URL}/email/mcp",
    "authorization_servers": [settings.SCALEKIT_AUTHORIZATION_SERVERS],
    "bearer_methods_supported": ["header"],
    "resource_documentation": f"{BASE_URL}/email/mcp/docs",
    "scopes_supported": ["search:read"],
}


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    async with contextlib.AsyncExitStack() as stack:
        await stack.enter_async_context(gnews_mcp_server.session_manager.run())
        await stack.enter_async_context(email_mcp_server.session_manager.run())
        print("Starting up MCP Servers...")
        yield
        print("Shutting down MCP Servers...")

app = FastAPI(title="MCP Server", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/.well-known/oauth-protected-resource/gnews/mcp")
async def gnews_oauth_metadata():
    return GNEWS_METADATA


@app.get("/.well-known/oauth-protected-resource/email/mcp")
async def email_oauth_metadata():
    return EMAIL_METADATA


app.add_middleware(AuthMiddleware)

app.mount("/gnews", gnews_mcp_server.streamable_http_app(), name="GNews MCP Server")
app.mount("/email", email_mcp_server.streamable_http_app(), name="Email MCP Server")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port, proxy_headers=True, forwarded_allow_ips="*")
