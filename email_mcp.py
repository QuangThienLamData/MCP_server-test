import os
from typing import Optional
from dotenv import load_dotenv
import requests
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

load_dotenv()

mcp = FastMCP(
    name="Email MCP Server",
    transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
)


@mcp.tool()
def check_api_status() -> dict:
    """
    Check the email service connectivity and API key validity by making a minimal test request.
    Returns:
        dict: A dictionary containing the status of the API connectivity and key validity.
    """
    # Placeholder implementation - replace with actual API status check logic
    return "Hello there! This is a placeholder response for the check_api_status tool. Please implement the actual API status check logic here. "

@mcp.tool()
def send_email(
    to: str,
    subject: str,
    body: str,
    from_email: Optional[str] = None,
    cc: Optional[str] = None,
    bcc: Optional[str] = None,
) -> dict:
    """
    Send an email using the email service.

    Args:
        to (str): Recipient email address.
        subject (str): Subject of the email.
        body (str): Body content of the email.
        from_email (Optional[str]): Sender email address. If not provided, a default sender will be used.
        cc (Optional[str]): CC recipient email addresses, comma-separated.
        bcc (Optional[str]): BCC recipient email addresses, comma-separated.

    Returns:
        dict: A dictionary containing the status of the email sending operation.
    """
    # Placeholder implementation - replace with actual email sending logic
    return {
        "status": "success",
        "message": f"Email sent to {to} with subject '{subject}'. This is a placeholder response. Please implement the actual email sending logic here."
    }
