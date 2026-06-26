import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    # ScaleKit Configuration
    SCALEKIT_ENVIRONMENT_URL: str = field(default_factory=lambda: os.environ.get("SCALEKIT_ENVIRONMENT_URL", ""))
    SCALEKIT_CLIENT_ID: str = field(default_factory=lambda: os.environ.get("SCALEKIT_CLIENT_ID", ""))
    SCALEKIT_CLIENT_SECRET: str = field(default_factory=lambda: os.environ.get("SCALEKIT_CLIENT_SECRET", ""))
    SCALEKIT_RESOURCE_IDENTIFIER: str = field(default_factory=lambda: os.environ.get("SCALEKIT_RESOURCE_IDENTIFIER", ""))
    SCALEKIT_RESOURCE_METADATA_URL: str = field(default_factory=lambda: os.environ.get("SCALEKIT_RESOURCE_METADATA_URL", ""))
    SCALEKIT_AUTHORIZATION_SERVERS: str = field(default_factory=lambda: os.environ.get("SCALEKIT_AUTHORIZATION_SERVERS", ""))
    SCALEKIT_AUDIENCE_NAME: str = field(default_factory=lambda: os.environ.get("SCALEKIT_AUDIENCE_NAME", ""))
    SCALEKIT_RESOURCE_NAME: str = field(default_factory=lambda: os.environ.get("SCALEKIT_RESOURCE_NAME", ""))
    SCALEKIT_RESOURCE_DOCS_URL: str = field(default_factory=lambda: os.environ.get("SCALEKIT_RESOURCE_DOCS_URL", ""))
    CLIENT_ID: str = field(default_factory=lambda: os.environ.get("CLIENT_ID", ""))
    CLIENT_SECRET: str = field(default_factory=lambda: os.environ.get("CLIENT_SECRET", ""))

    # Tavily API Key
    GNEWS_API_KEY: str = field(default_factory=lambda: os.environ.get("GNEWS_API_KEY", ""))

    # Server Port
    PORT: int = field(default_factory=lambda: int(os.environ.get("PORT", 10000)))

    def __post_init__(self):
        required = [
            "SCALEKIT_ENVIRONMENT_URL",
            "SCALEKIT_CLIENT_ID",
            "SCALEKIT_CLIENT_SECRET",
            "SCALEKIT_RESOURCE_IDENTIFIER",
            "SCALEKIT_RESOURCE_METADATA_URL",
            "SCALEKIT_AUTHORIZATION_SERVERS",
            "SCALEKIT_AUDIENCE_NAME",
            "SCALEKIT_RESOURCE_DOCS_URL",
        ]
        for name in required:
            if not getattr(self, name):
                raise ValueError(f"{name} environment variable not set")


settings = Settings()
