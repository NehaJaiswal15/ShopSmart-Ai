from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator


class ConfigSettings(BaseSettings):
    """Configuration Settings utilizing Pydantic for validation and type-safety.

    Loads configuration from environment variables or a local `.env` file.
    If required fields are missing, validation will raise an error at startup.
    """

    # Required keys — Pydantic validates these must be present at startup
    ASTRA_DB_API_ENDPOINT: str = Field(..., validation_alias="ASTRA_DB_API_ENDPOINT")
    ASTRA_DB_APPLICATION_TOKEN: str = Field(..., validation_alias="ASTRA_DB_APPLICATION_TOKEN")
    ASTRA_DB_KEYSPACE: str = Field(..., validation_alias="ASTRA_DB_KEYSPACE")
    GROQ_API_KEY: str = Field(..., validation_alias="GROQ_API_KEY")

    # Optional config — defaults provided
    EMBEDDING_MODEL: str = "BAAI/bge-base-en-v1.5"
    RAG_MODEL: str = "llama-3.1-8b-instant"

    @field_validator("*", mode="before")
    @classmethod
    def strip_quotes(cls, v):
        """Strip enclosing quotes from string values loaded from .env."""
        if isinstance(v, str):
            v_stripped = v.strip()
            if (v_stripped.startswith('"') and v_stripped.endswith('"')) or (
                v_stripped.startswith("'") and v_stripped.endswith("'")
            ):
                return v_stripped[1:-1]
            return v_stripped
        return v

    # Automatically load environment variables from .env
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # Ignore extra env variables in .env
    )


# Instantiate to validate immediately on import
Config = ConfigSettings()
