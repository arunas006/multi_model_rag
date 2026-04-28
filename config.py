from __future__ import annotations

import logging

from pydantic import SecretStr,model_validator
from pydantic_settings import BaseSettings,SettingsConfigDict

class Settings(BaseSettings):

    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(env_file=".env", 
                                      env_file_encoding="utf-8",
                                      case_sensitive=False)

    # Storage
    storage_backend: str = "local"   # "local" | "s3"
    s3_bucket_name: str | None = None
    s3_region: str = "ap-south-1"
    s3_prefix: str = "rag-app"   # optional folder prefix


    #Backend Parser
    parser_backend: str = "ollama"  # "cloud" | "ollama"
    z_ai_api_key: SecretStr | None = None
    log_level: str = "INFO"
    output_dir: str = "./output"
    config_yaml_path: str = "config.yaml"

    #openai
    openai_api_key: SecretStr | None = None
    openai_llm_model: str = "gpt-4o"

    #Embedding
    embedding_provider: str = "openai"  # "openai" | "gemini"
    embedding_model: str = "text-embedding-3-large"
    embedding_dimensions: int = 3072
    gemini_api_key: SecretStr | None = None

     # Qdrant
    qdrant_url: str | None = None
    qdrant_api_key: SecretStr | None = None
    qdrant_collection_name: str = "documents"

    # Reranker
    reranker_backend: str = "openai"  # "jina" | "openai" | "bge" | "qwen"
    reranker_top_n: int = 5
    jina_api_key: SecretStr | None = None

    # Feature flags
    image_caption_enabled: bool = True

    # Captioning tuning
    table_max_tokens: int = 2000
    table_max_input_chars: int = 12_000
    image_max_tokens: int = 800
    table_use_vision: bool = False

    # API server
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 1

    # Logging
    log_json: bool = False

    META_FILE : str = "data/source_files.json"

    @model_validator(mode="after")
    def _validate_backend(self) -> Settings:
        if self.parser_backend == "cloud" :
            if self.z_ai_api_key is None:
                raise ValueError("z_ai_api_key must be set when parser_backend is 'cloud'")
        elif self.parser_backend == "ollama":
            if self.config_yaml_path == "config.yaml":
                self.config_yaml_path = "./ingestion/config.yaml"
        else:
            raise ValueError("parser_backend must be either 'cloud' or 'ollama'")
        
        if self.storage_backend == "s3":
            if not self.s3_bucket_name:
                raise ValueError("s3_bucket_name must be set when storage_backend='s3'")
            return self
    
_settings : Settings | None = None

def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings

def configure_logging(level: str = "INFO") -> None:
    """Configure root logger with the given level."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

if __name__ == "__main__":
    settings = get_settings()
    configure_logging(settings.log_level)
    logging.info("Settings loaded successfully.")
    print(settings.model_dump())

