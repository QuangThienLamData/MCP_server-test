import base64
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

BASE_URL = settings.SCALEKIT_RESOURCE_DOCS_URL.rsplit("/gnews/mcp", 1)[0]


class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Skip auth for well-known discovery endpoints
        if request.url.path.startswith("/.well-known/"):
            return await call_next(request)

        try:
            auth_header = request.headers.get("Authorization")
            if not auth_header or not auth_header.startswith("Bearer "):
                raise HTTPException(status_code=401, detail="Missing or invalid authorization header")

            token = auth_header.split(" ")[1]

            # Debug: log token claims to find audience mismatch
            try:
                payload_part = token.split(".")[1]
                payload_part += "=" * (4 - len(payload_part) % 4)
                claims = json.loads(base64.urlsafe_b64decode(payload_part))
                logger.info(f"Token aud: {claims.get('aud')}, iss: {claims.get('iss')}")
                logger.info(f"Expected aud: {settings.SCALEKIT_AUDIENCE_NAME}")
            except Exception:
                logger.warning("Could not decode token for debugging")

            validation_options = TokenValidationOptions(
                issuer=settings.SCALEKIT_ENVIRONMENT_URL,
                audience=[settings.SCALEKIT_AUDIENCE_NAME],
            )

            # Only parse body for POST requests (JSON-RPC calls)
            if request.method == "POST":
                request_body = await request.body()
                try:
                    request_data = json.loads(request_body.decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    request_data = {}

                if request_data.get("method") == "tools/call":
                    validation_options.required_scopes = ["mcp:tools:search:read"]

            scalekit_client.validate_token(token, options=validation_options)

        except HTTPException as e:
            # Determine which metadata URL to reference based on path
            if request.url.path.startswith("/email"):
                metadata_url = f"{BASE_URL}/.well-known/oauth-protected-resource/email/mcp"
            else:
                metadata_url = f"{BASE_URL}/.well-known/oauth-protected-resource/gnews/mcp"

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
