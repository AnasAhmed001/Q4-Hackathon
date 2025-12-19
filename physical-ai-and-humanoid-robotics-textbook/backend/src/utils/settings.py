from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class Settings(BaseSettings):
    # API Keys
    cohere_api_key: str
    groq_api_key: str  # Switched from Gemini to Groq

    # Database
    database_url: str

    # Qdrant
    qdrant_url: str
    qdrant_api_key: str

    # CORS
    allowed_origin: str

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore"  # Ignore extra fields that don't match
    )


settings = Settings()