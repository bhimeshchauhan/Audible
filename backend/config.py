"""
Application configuration with environment variable support.

Security-focused defaults with configurable overrides.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional

from dotenv import load_dotenv

load_dotenv()


def _parse_list(value: str) -> List[str]:
    """Parse comma-separated string into list."""
    if not value:
        return []
    return [v.strip() for v in value.split(",") if v.strip()]


@dataclass
class ServerConfig:
    """Server configuration."""

    host: str = os.getenv("SERVER_HOST", "127.0.0.1")
    port: int = int(os.getenv("SERVER_PORT", "8000"))
    reload: bool = os.getenv("SERVER_RELOAD", "false").lower() == "true"
    workers: int = int(os.getenv("SERVER_WORKERS", "1"))


@dataclass
class CORSConfig:
    """CORS configuration."""

    # Default to localhost origins for development
    # In production, set CORS_ORIGINS to your frontend domain(s)
    origins: List[str] = field(
        default_factory=lambda: _parse_list(
            os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:5173")
        )
    )
    allow_credentials: bool = os.getenv("CORS_CREDENTIALS", "true").lower() == "true"
    allow_methods: List[str] = field(
        default_factory=lambda: _parse_list(os.getenv("CORS_METHODS", "GET,POST,OPTIONS"))
    )
    allow_headers: List[str] = field(
        default_factory=lambda: _parse_list(os.getenv("CORS_HEADERS", "*"))
    )


@dataclass
class UploadConfig:
    """File upload configuration."""

    # Maximum file size in bytes (default: 50MB)
    max_size_bytes: int = int(os.getenv("UPLOAD_MAX_SIZE_MB", "50")) * 1024 * 1024
    allowed_extensions: List[str] = field(default_factory=lambda: [".pdf"])
    temp_dir: Optional[str] = os.getenv("UPLOAD_TEMP_DIR")


@dataclass
class NLPConfig:
    """NLP processing configuration."""

    # PDF
    max_pages: Optional[int] = (
        int(os.getenv("MAX_PAGES")) if os.getenv("MAX_PAGES") else None
    )

    # spaCy
    spacy_model: str = os.getenv("SPACY_MODEL", "en_core_web_sm")
    enable_coref: bool = os.getenv("ENABLE_COREF", "true").lower() == "true"
    coref_model_arch: str = os.getenv("COREF_MODEL_ARCH", "FCoref")

    # Segmentation
    min_sentence_len: int = int(os.getenv("MIN_SENTENCE_LEN", "10"))
    window_sentences: int = int(os.getenv("WINDOW_SENTENCES", "8"))
    window_stride: int = int(os.getenv("WINDOW_STRIDE", "4"))
    max_context_chars: int = int(os.getenv("MAX_CONTEXT_CHARS", "1600"))

    # Characters
    min_person_freq: int = int(os.getenv("MIN_PERSON_FREQ", "2"))
    max_people_per_window: int = int(os.getenv("MAX_PEOPLE_PER_WINDOW", "10"))

    # Candidate selection
    min_interaction_weight: int = int(os.getenv("MIN_INTERACTION_WEIGHT", "2"))
    evidence_per_pair: int = int(os.getenv("EVIDENCE_PER_PAIR", "6"))
    max_pairs_to_describe: int = int(os.getenv("MAX_PAIRS_TO_DESCRIBE", "400"))

    # Verification
    min_quote_words: int = int(os.getenv("MIN_QUOTE_WORDS", "5"))
    max_quote_words: int = int(os.getenv("MAX_QUOTE_WORDS", "40"))
    require_quotes: bool = os.getenv("REQUIRE_QUOTES", "true").lower() == "true"

    # Outputs
    out_dir: str = os.getenv("OUTPUT_DIR", "results")


@dataclass
class LLMConfig:
    """LLM configuration for relationship extraction."""

    use_local: bool = os.getenv("USE_LOCAL_LLM", "true").lower() == "true"
    local_model: str = os.getenv("LOCAL_MODEL", "llama3.2")
    local_base_url: str = os.getenv("LOCAL_BASE_URL", "http://localhost:11434/v1")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.0"))

    # Concurrency
    max_workers: int = int(os.getenv("LLM_MAX_WORKERS", "2"))
    min_request_interval_sec: float = float(
        os.getenv("LLM_MIN_REQUEST_INTERVAL_SEC", "0.1")
    )
    max_retries: int = int(os.getenv("LLM_MAX_RETRIES", "3"))


@dataclass
class Config:
    """Main application configuration."""

    server: ServerConfig = field(default_factory=ServerConfig)
    cors: CORSConfig = field(default_factory=CORSConfig)
    upload: UploadConfig = field(default_factory=UploadConfig)
    nlp: NLPConfig = field(default_factory=NLPConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)

    # Environment
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"
    environment: str = os.getenv("ENVIRONMENT", "development")


# Global config instance
config = Config()
