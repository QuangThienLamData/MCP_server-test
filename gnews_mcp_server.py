import os
from typing import Optional
from dotenv import load_dotenv
import requests
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

load_dotenv()

API_KEY = os.getenv("API_KEY")
BASE_URL = "https://gnews.io/api/v4"

mcp = FastMCP(
    name="GNews MCP Server",
    transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
)


@mcp.tool()
def check_api_status() -> dict:
    """Check the GNews API connectivity and API key validity by making a minimal test request."""
    try:
        response = requests.get(
            f"{BASE_URL}/top-headlines",
            params={"apikey": API_KEY, "max": 1, "category": "general"},
            timeout=10,
        )
        return {
            "status": "ok" if response.status_code == 200 else "error",
            "http_status_code": response.status_code,
            "api_key_configured": bool(API_KEY),
            "message": "API is reachable and key is valid." if response.status_code == 200 else response.json().get("errors", response.text),
        }
    except requests.exceptions.ConnectionError:
        return {"status": "error", "message": "Could not connect to GNews API. Check your internet connection."}
    except requests.exceptions.Timeout:
        return {"status": "error", "message": "Request timed out."}


@mcp.tool()
def search_news(
    q: str,
    lang: Optional[str] = None,
    country: Optional[str] = None,
    max: Optional[int] = 10,
    in_: Optional[str] = None,
    nullable: Optional[str] = None,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    sortby: Optional[str] = "publishedAt",
    page: Optional[int] = 1,
) -> dict:
    """Search for news articles using the GNews API.

    Args:
        q: Search keywords (required, max 200 characters). Supports AND, OR, NOT operators and exact phrases with quotes.
        lang: Filter by language (2-letter code, e.g. 'en', 'fr', 'de').
        country: Filter by country (2-letter code, e.g. 'us', 'gb', 'ca').
        max: Number of articles to return (1-100, default 10).
        in_: Attributes to search within — comma-separated: 'title', 'description', 'content'.
        nullable: Allow null values for these attributes — comma-separated: 'description', 'content', 'image'.
        from_date: Return articles published on or after this ISO 8601 date (e.g. '2024-01-01T00:00:00Z').
        to_date: Return articles published on or before this ISO 8601 date.
        sortby: Sort order — 'publishedAt' (newest first) or 'relevance' (best match first).
        page: Page number for pagination (default 1, max 1000 articles total).
    """
    params = {"q": q, "apikey": API_KEY, "max": max, "sortby": sortby, "page": page}

    if lang:
        params["lang"] = lang
    if country:
        params["country"] = country
    if in_:
        params["in"] = in_
    if nullable:
        params["nullable"] = nullable
    if from_date:
        params["from"] = from_date
    if to_date:
        params["to"] = to_date

    response = requests.get(f"{BASE_URL}/search", params=params)
    response.raise_for_status()
    return response.json()


@mcp.tool()
def get_top_headlines(
    category: Optional[str] = "general",
    lang: Optional[str] = None,
    country: Optional[str] = None,
    max: Optional[int] = 10,
    q: Optional[str] = None,
    nullable: Optional[str] = None,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    page: Optional[int] = 1,
) -> dict:
    """Fetch top headline news articles using the GNews API.

    Args:
        category: News category — one of: 'general', 'world', 'nation', 'business', 'technology',
                  'entertainment', 'sports', 'science', 'health'. Defaults to 'general'.
        lang: Filter by language (2-letter code, e.g. 'en', 'fr', 'es').
        country: Filter by country (2-letter code, e.g. 'us', 'gb', 'au').
        max: Number of articles to return (1-100, default 10).
        q: Optional keyword filter (max 200 characters). Supports AND, OR, NOT and exact phrases.
        nullable: Allow null values for these attributes — comma-separated: 'description', 'content', 'image'.
        from_date: Return articles published on or after this ISO 8601 date (e.g. '2024-01-01T00:00:00Z').
        to_date: Return articles published on or before this ISO 8601 date.
        page: Page number for pagination (default 1, max 1000 articles total).
    """
    params = {"apikey": API_KEY, "category": category, "max": max, "page": page}

    if lang:
        params["lang"] = lang
    if country:
        params["country"] = country
    if q:
        params["q"] = q
    if nullable:
        params["nullable"] = nullable
    if from_date:
        params["from"] = from_date
    if to_date:
        params["to"] = to_date

    response = requests.get(f"{BASE_URL}/top-headlines", params=params)
    response.raise_for_status()
    return response.json()


# if __name__ == "__main__":
#     mcp.run(transport="streamable-http")
