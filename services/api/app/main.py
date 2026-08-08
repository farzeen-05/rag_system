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
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, List
from contextlib import asynccontextmanager
import logging
import structlog

from config import get_settings
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from fastapi import Response
import monitoring
from fastapi.responses import RedirectResponse, JSONResponse
from fastapi.security import OAuth2PasswordRequestForm
from auth import (
    USERS_DB, UserIn, Token, hash_password, verify_password,
    create_access_token, get_current_user, get_google_auth_url,
    exchange_google_code
)
from fastapi.responses import RedirectResponse, JSONResponse
from fastapi.security import OAuth2PasswordRequestForm
from auth import (
    USERS_DB, UserIn, Token, hash_password, verify_password,
    create_access_token, get_current_user, get_google_auth_url,
    exchange_google_code
)
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

@app.get("/")
async def serve_frontend():
    return FileResponse("static_index.html")

@app.get("/metrics")
async def metrics():
    """Prometheus scrape endpoint. No auth — internal cluster access only via k8s network policy in production."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)



# ── Dependency Injection ─────────────────────────────────────────────

def get_engine() -> RAGEngine:
    return app_state["engine"]

def get_user_engine(current_user: str = Depends(get_current_user)) -> RAGEngine:
    settings = get_settings()
    safe_user = "".join(c if c.isalnum() else "_" for c in current_user)[:40]
    collection_name = f"rag_{safe_user}"
    retriever = VectorRetriever(settings.CHROMA_HOST, settings.CHROMA_PORT, collection_name)
    return RAGEngine(
        embedding_model=app_state["embedding_model"],
        retriever=retriever,
        llm_client=app_state["llm_client"],
        chunker=app_state["chunker"],
        settings=settings,
    )



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
    current_user: str = Depends(get_current_user)
):
    """
    Index a document into the vector store.
    
    This is called:
    - When you add new documents to your knowledge base
    - When you update existing documents (re-index)
    """
    engine = get_user_engine(current_user)
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
    current_user: str = Depends(get_current_user)
):
    """
    Main RAG query endpoint.
    Returns answer + sources + retrieved chunks.
    """
    engine = get_user_engine(current_user)
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
            "hallucination_risk": response.hallucination_risk,
            "evidence_sufficient": response.evidence_sufficient,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_stats(engine: RAGEngine = Depends(get_engine)):
    """Useful for monitoring how many docs are in the vector store."""
    return engine.retriever.get_stats()


@app.post("/auth/register", response_model=Token)
async def register(user: UserIn):
    if user.username in USERS_DB:
        raise HTTPException(status_code=400, detail="Username already exists")
    USERS_DB[user.username] = {
        "username": user.username,
        "hashed_password": hash_password(user.password)
    }
    token = create_access_token({"sub": user.username})
    return {"access_token": token, "token_type": "bearer"}

@app.post("/auth/login", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = USERS_DB.get(form_data.username)
    if not user or not verify_password(form_data.password, user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Invalid username or password")
    token = create_access_token({"sub": form_data.username})
    return {"access_token": token, "token_type": "bearer"}

@app.get("/auth/google")
async def google_login():
    return RedirectResponse(get_google_auth_url())

@app.get("/auth/google/callback")
async def google_callback(code: str):
    user_info = await exchange_google_code(code)
    if not user_info:
        raise HTTPException(status_code=400, detail="Google auth failed")
    username = user_info.get("email")
    if username not in USERS_DB:
        USERS_DB[username] = {"username": username, "hashed_password": None}
    token = create_access_token({"sub": username})
    response = RedirectResponse(url="/")
    response.set_cookie("access_token", token, httponly=True, max_age=86400)
    return response

@app.get("/auth/me")
async def get_me(current_user: str = Depends(get_current_user)):
    return {"username": current_user}

@app.post("/auth/logout")
async def logout():
    response = JSONResponse({"message": "Logged out"})
    response.delete_cookie("access_token")
    return response
