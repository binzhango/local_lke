"""Typed application configuration loaded from environment variables or .env."""

from functools import lru_cache

from pydantic import Field, HttpUrl, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Local LKE settings. Environment variables use the ``LKE_`` prefix."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="LKE_",
        extra="ignore",
        case_sensitive=False,
    )

    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    chat_base_url: HttpUrl = HttpUrl("http://127.0.0.1:1234/v1")
    chat_model: str = Field(default="local-model", min_length=1)
    chat_api_key: SecretStr | None = SecretStr("lm-studio")
    chat_timeout_seconds: float = Field(default=60.0, gt=0)
    chat_max_retries: int = Field(default=1, ge=0, le=10)
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        min_length=1,
    )
    default_top_k: int = Field(default=3, ge=1, le=20)

    @property
    def redacted_summary(self) -> dict[str, str | int | float]:
        """Return display-safe configuration without any API key."""
        return {
            "host": self.host,
            "port": self.port,
            "chat_base_url": str(self.chat_base_url),
            "chat_model": self.chat_model,
            "chat_timeout_seconds": self.chat_timeout_seconds,
            "chat_max_retries": self.chat_max_retries,
            "embedding_model": self.embedding_model,
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide immutable settings object."""
    return Settings()

