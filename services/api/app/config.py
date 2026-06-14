
### 5.2 Configuration System

# services/api/app/config.py

"""
WHY A CONFIG FILE?
Never hardcode API keys, URLs, or settings in your code.
- In development: values come from .env file
- In Docker: values come from environment variables
- In Kubernetes: values come from ConfigMaps and Secrets
This ONE file is the bridge between all environments.
"""

from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import Literal


class Settings(BaseSettings):
    """
    Pydantic BaseSettings automatically reads from:
    1. Environment variables (highest priority)
    2. .env file
    3. Default values (lowest priority)
    
    This means the SAME code runs locally and in production —
    only the environment changes.
    """
    
    # ── Application ──────────────────────────────────────────
    APP_NAME: str = "RAG System"
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: Literal["development", "staging", "production"] = "development"
    DEBUG: bool = False
    
    # ── ChromaDB ─────────────────────────────────────────────
    # In Docker Compose, this is the service name "chroma"
    # In Kubernetes, this is the service DNS: chroma-service.rag-namespace.svc.cluster.local
    CHROMA_HOST: str = "localhost"
    CHROMA_PORT: int = 8000
    CHROMA_COLLECTION_NAME: str = "rag_documents"
    
    # ── Embedding Model ──────────────────────────────────────
    # We use a local model so there's no API cost.
    # The model downloads once and is cached.
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384  # MiniLM produces 384-dim vectors
    
    # ── LLM Configuration ────────────────────────────────────
    LLM_PROVIDER: Literal["openai", "ollama", "anthropic", "groq"] = "ollama"
    LLM_MODEL: str = "llama3.2"         # Ollama model name
    OPENAI_API_KEY: str = ""            # Optional: OpenAI key
    ANTHROPIC_API_KEY: str = ""         # Optional: Anthropic key
    GROQ_API_KEY: str = ""              # Groq API key
    OLLAMA_BASE_URL: str = "http://ollama:11434"
    
    # ── Retrieval Settings ───────────────────────────────────
    # How many chunks to retrieve per query
    RETRIEVAL_TOP_K: int = 5
    # Minimum similarity score (0-1) to include a chunk
    RETRIEVAL_SCORE_THRESHOLD: float = 0.3
    # Enable hybrid search (vector + keyword)
    USE_HYBRID_SEARCH: bool = True
    # Weight for dense vs sparse search (1.0 = all dense)
    HYBRID_ALPHA: float = 0.7
    
    # ── Chunking Defaults ────────────────────────────────────
    DEFAULT_CHUNK_SIZE: int = 512       # tokens per chunk
    DEFAULT_CHUNK_OVERLAP: int = 50     # overlap between chunks
    
    # ── AWS ──────────────────────────────────────────────────
    AWS_REGION: str = "us-east-1"
    S3_DOCUMENTS_BUCKET: str = ""
    
    class Config:
        # Pydantic looks for a .env file in the working directory
        env_file = ".env"
        # Allows extra fields (ignores unknown env vars)
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    """
    WHY lru_cache?
    Settings are expensive to create (reads files, validates types).
    lru_cache makes this a singleton — created ONCE, reused everywhere.
    This is a standard FastAPI dependency injection pattern.
    """
    return Settings()
