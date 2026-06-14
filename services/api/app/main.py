"""
FastAPI is the HTTP interface for our RAG system.
Every HTTP endpoint maps to one RAG operation.

WHY FASTAPI?
- Automatic OpenAPI docs (visit /docs to test your API)
- Pydantic validation (bad input = automatic 422 error, not crash)
- Async support (handles many concurrent requests)
- Faster than Flask for I/O-bound tasks (LLM calls, DB queries)
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, List
from contextlib import asynccontextmanager
import logging
import structlog

from config import get_settings
from rag.engine import RAGEngine
from rag.embeddings import EmbeddingModel
from rag.retriever import VectorRetriever
from rag.chunker import DocumentChunker
from rag.generator import get_llm_client


# ── Request/Response Models ──────────────────────────────────────────

class IngestRequest(BaseModel):
    """
    Pydantic model validates incoming JSON automatically.
    If 'text' is missing or not a string → 422 error before reaching handler.
    """
    text: str = Field(..., min_length=10, description="Document text to index")
    doc_id: str = Field(..., description="Unique document identifier")
    metadata: Dict = Field(default_factory=dict, description="Filterable metadata")
    chunking_strategy: str = Field(
        default="recursive",
        pattern="^(recursive|tokens|markdown|semantic)$"  # Regex validation!
    )


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    top_k: Optional[int] = Field(default=5, ge=1, le=20)
    use_mmr: bool = False
    filter_metadata: Optional[Dict] = None
    score_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)


# ── App Lifecycle ────────────────────────────────────────────────────

# Global container for dependencies (poor man's DI container)
app_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan events: code that runs at startup and shutdown.
    
    WHY HERE?
    We create embedding model, connect to ChromaDB, etc. ONCE at startup.
    If we created them per-request, every query would reload a 90MB model.
    
    This is called "warm starting" — expensive resources loaded once,
    shared across all requests.
    """
    settings = get_settings()
    
    # Initialize all components
    app_state["embedding_model"] = EmbeddingModel(settings.EMBEDDING_MODEL_NAME)
    app_state["retriever"] = VectorRetriever(
        settings.CHROMA_HOST, settings.CHROMA_PORT, settings.CHROMA_COLLECTION_NAME
    )
    app_state["chunker"] = DocumentChunker()
    app_state["llm_client"] = get_llm_client(settings)
    app_state["engine"] = RAGEngine(
        embedding_model=app_state["embedding_model"],
        retriever=app_state["retriever"],
        llm_client=app_state["llm_client"],
        chunker=app_state["chunker"],
        settings=settings,
    )
    app_state["settings"] = settings
    
    yield  # App runs here
    
    # Cleanup (if needed)
    app_state.clear()


app = FastAPI(
    title="RAG System API",
    description="Production RAG system with multiple retrieval strategies",
    version="1.0.0",
    lifespan=lifespan,
    # Disable docs in production for security
    docs_url="/docs" if True else None,
)

# CORS: allows browser frontends to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production!
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Dependency Injection ─────────────────────────────────────────────

def get_engine() -> RAGEngine:
    """
    FastAPI Depends() pattern.
    Instead of accessing app_state directly in each route,
    we declare the dependency. FastAPI handles calling it.
    Makes routes cleaner and testable.
    """
    return app_state["engine"]


# ── Routes ──────────────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    """
    Kubernetes LIVENESS probe hits this.
    If this returns 200, Kubernetes knows the pod is alive.
    If it fails, Kubernetes restarts the pod.
    """
    return {
        "status": "healthy",
        "version": app_state.get("settings", {}).APP_VERSION if app_state else "unknown"
    }


@app.get("/ready")
async def readiness_check():
    """
    Kubernetes READINESS probe.
    Only return 200 when the app is FULLY ready (model loaded, DB connected).
    Kubernetes won't send traffic until this returns 200.
    
    This prevents the "502 Bad Gateway" you get when the pod starts
    but the model hasn't loaded yet.
    """
    try:
        stats = app_state["retriever"].get_stats()
        return {"status": "ready", "vector_store": stats}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Not ready: {str(e)}")


@app.post("/ingest")
async def ingest_document(
    request: IngestRequest,
    engine: RAGEngine = Depends(get_engine)
):
    """
    Index a document into the vector store.
    
    This is called:
    - When you add new documents to your knowledge base
    - When you update existing documents (re-index)
    """
    try:
        result = engine.ingest_document(
            text=request.text,
            doc_id=request.doc_id,
            metadata=request.metadata,
            chunking_strategy=request.chunking_strategy,
        )
        return {"success": True, **result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query")
async def query_rag(
    request: QueryRequest,
    engine: RAGEngine = Depends(get_engine)
):
    """
    Main RAG query endpoint.
    Returns answer + sources + retrieved chunks.
    """
    try:
        response = engine.query(
            question=request.question,
            top_k=request.top_k,
            filter_metadata=request.filter_metadata,
            use_mmr=request.use_mmr,
            score_threshold=request.score_threshold,
        )
        return {
            "answer": response.answer,
            "sources": response.sources,
            "chunks": response.retrieved_chunks,
            "latency_ms": response.latency_ms,
            "model": response.model_used,
            "scores": response.retrieval_scores,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_stats(engine: RAGEngine = Depends(get_engine)):
    """Useful for monitoring how many docs are in the vector store."""
    return engine.retriever.get_stats()
