from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from gnews_mcp_server import mcp as gnews_mcp_server
from email_mcp import mcp as email_mcp_server
import contextlib
import uvicorn

@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    # async with gnews_mcp_server.session_manager.run():
    async with  contextlib.AsyncExitStack() as stack:
        await stack.enter_async_context(gnews_mcp_server.session_manager.run())
        await stack.enter_async_context(email_mcp_server.session_manager.run())
        # Startup code
        print("Starting up the GNews MCP Server...")
        yield
        # Shutdown code
        print("Shutting down the GNews MCP Server...")

app = FastAPI(title="GNews MCP Server", version="1.0.0", lifespan=lifespan)

#CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for simplicity; adjust as needed for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/gnews", gnews_mcp_server.streamable_http_app(), name="GNews MCP Server")
app.mount("/email", email_mcp_server.streamable_http_app(), name="Email MCP Server")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000, log_level="info")
