import json
import logging
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from scalekit import ScalekitClient
from scalekit.common.scalekit import TokenValidationOptions
from starlette.middleware.base import BaseHTTPMiddleware

from .config import settings

logger = logging.getLogger(__name__)

scalekit_client = ScalekitClient(
    settings.SCALEKIT_ENVIRONMENT_URL,
    settings.SCALEKIT_CLIENT_ID,
    settings.SCALEKIT_CLIENT_SECRET
)

BASE_URL = settings.SCALEKIT_RESOURCE_DOCS_URL.rsplit("/research/mcp", 1)[0]


class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Skip Scalekit for well-known discovery + the internal cron endpoint
        # (the latter is guarded by its own X-Cron-Secret check in main.py).
        if request.url.path.startswith("/.well-known/") or request.url.path.startswith("/internal/") or request.url.path.startswith("/proxy/"):
            return await call_next(request)

        try:
            auth_header = request.headers.get("Authorization")
            if not auth_header or not auth_header.startswith("Bearer "):
                raise HTTPException(status_code=401, detail="Missing or invalid authorization header")

            token = auth_header.split(" ")[1]

            audience_name = settings.SCALEKIT_AUDIENCE_NAME.rstrip("/")
            validation_options = TokenValidationOptions(
                issuer=settings.SCALEKIT_ENVIRONMENT_URL,
                audience=[audience_name, audience_name + "/", settings.SCALEKIT_RESOURCE_IDENTIFIER],
            )

            # Only parse body for POST requests (JSON-RPC calls)
            if request.method == "POST":
                request_body = await request.body()
                try:
                    request_data = json.loads(request_body.decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    request_data = {}

                if request_data.get("method") == "tools/call":
                    validation_options.required_scopes = ["search:read"]

            scalekit_client.validate_token(token, options=validation_options)

        except HTTPException as e:
            # Determine which metadata URL to reference based on path
            metadata_url = f"{BASE_URL}/.well-known/oauth-protected-resource/research/mcp"

            return JSONResponse(
                status_code=e.status_code,
                content={
                    "error": "unauthorized" if e.status_code == 401 else "forbidden",
                    "error_description": e.detail,
                },
                headers={
                    "WWW-Authenticate": f'Bearer realm="OAuth", resource_metadata="{metadata_url}"'
                },
            )
        except Exception:
            logger.exception("Token validation failed")
            return JSONResponse(
                status_code=401,
                content={"error": "unauthorized", "error_description": "Token validation failed"},
            )

        return await call_next(request)
