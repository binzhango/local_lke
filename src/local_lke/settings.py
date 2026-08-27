"""Typed application configuration loaded from environment variables or .env."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, HttpUrl, SecretStr, field_validator
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
    database_url: str = "postgresql+psycopg://localhost/local_lke"
    postgres_bin_directory: Path = Path("/opt/homebrew/opt/postgresql@18/bin")
    upload_directory: Path = Path("data/uploads")
    max_upload_bytes: int = Field(default=10 * 1024 * 1024, ge=1)
    max_batch_bytes: int = Field(default=50 * 1024 * 1024, ge=1)
    default_parser_strategy: str = "fast"
    default_chunk_strategy: str = "markdown"
    chunk_size: int = Field(default=800, ge=100, le=100_000)
    chunk_overlap: int = Field(default=120, ge=0, le=10_000)

    @field_validator("default_parser_strategy")
    @classmethod
    def validate_parser_strategy(cls, value: str) -> str:
        if value not in {"fast", "hi_res"}:
            raise ValueError("must be 'fast' or 'hi_res'")
        return value

    @field_validator("default_chunk_strategy")
    @classmethod
    def validate_chunk_strategy(cls, value: str) -> str:
        if value not in {"recursive", "markdown", "semantic"}:
            raise ValueError("must be recursive, markdown, or semantic")
        return value

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
            "database_url": _redact_database_url(self.database_url),
            "upload_directory": str(self.upload_directory),
            "max_upload_bytes": self.max_upload_bytes,
            "max_batch_bytes": self.max_batch_bytes,
            "default_parser_strategy": self.default_parser_strategy,
            "default_chunk_strategy": self.default_chunk_strategy,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide immutable settings object."""
    return Settings()


def _redact_database_url(value: str) -> str:
    """Hide credentials while retaining enough detail for local diagnostics."""
    from sqlalchemy.engine import make_url

    url = make_url(value)
    if url.password is not None:
        url = url.set(password="***")
    return url.render_as_string(hide_password=False)
