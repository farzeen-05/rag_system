from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime
import time
import logging
import hashlib

from rag.embeddings import EmbeddingModel
from rag.chunker import DocumentChunker, Chunk
from rag.retriever import VectorRetriever, RetrievedChunk

logger = logging.getLogger(__name__)

@dataclass
class RAGResponse:
    answer: str
    sources: List[Dict]
    retrieved_chunks: List[Dict]
    latency_ms: float
    model_used: str
    retrieval_scores: List[float]
    hallucination_risk: str = "low"
    evidence_sufficient: bool = True

# Simple in-memory cache
_query_cache: Dict[str, RAGResponse] = {}

def _cache_key(question: str, top_k: int) -> str:
    return hashlib.md5(f"{question}:{top_k}".encode()).hexdigest()

class RAGEngine:
    # Minimum score threshold — below this we flag insufficient evidence
    MIN_EVIDENCE_SCORE = 0.35

    def __init__(self, embedding_model, retriever, llm_client, chunker, settings):
        self.embedder = embedding_model
        self.retriever = retriever
        self.llm = llm_client
        self.chunker = chunker
        self.settings = settings

    def ingest_document(self, text: str, doc_id: str, metadata: dict,
                        chunking_strategy: str = "recursive") -> dict:
        start = time.time()
        self.retriever.delete_document(doc_id)

        if chunking_strategy == "recursive":
            chunks = self.chunker.chunk_recursive(text, metadata={**metadata, "doc_id": doc_id})
        elif chunking_strategy == "tokens":
            chunks = self.chunker.chunk_by_tokens(text, metadata={**metadata, "doc_id": doc_id})
        elif chunking_strategy == "markdown":
            chunks = self.chunker.chunk_markdown(text, metadata={**metadata, "doc_id": doc_id})
        else:
            chunks = self.chunker.chunk_recursive(text, metadata={**metadata, "doc_id": doc_id})

        if not chunks:
            return {"doc_id": doc_id, "chunks_indexed": 0,
                    "chunking_strategy": chunking_strategy, "indexing_time_seconds": 0,
                    "avg_chunk_length": 0}

        texts = [c.text for c in chunks]
        embeddings = self.embedder.embed_texts(texts)
        self.retriever.add_chunks(chunks, embeddings)

        return {
            "doc_id": doc_id,
            "chunks_indexed": len(chunks),
            "chunking_strategy": chunking_strategy,
            "indexing_time_seconds": round(time.time() - start, 2),
            "avg_chunk_length": round(sum(len(c.text) for c in chunks) / len(chunks))
        }

    def query(self, question: str, top_k: int = 5,
              filter_metadata: Optional[Dict] = None,
              use_mmr: bool = False,
              score_threshold: Optional[float] = None) -> RAGResponse:

        start = time.time()

        # ── Check cache first ─────────────────────────────────────
        cache_key = _cache_key(question, top_k)
        if cache_key in _query_cache:
            logger.info(f"Cache hit for query: {question[:50]}")
            cached = _query_cache[cache_key]
            cached.latency_ms = 0.0
            return cached

        # ── Retrieve ──────────────────────────────────────────────
        query_embedding = self.embedder.embed_query(question)
        threshold = score_threshold if score_threshold is not None else self.MIN_EVIDENCE_SCORE
        chunks = self.retriever.hybrid_search(
            question, query_embedding, top_k=top_k * 2,
            score_threshold=threshold,
            filter_metadata=filter_metadata
        )
        chunks = self.retriever.rerank(chunks, question, top_k=top_k)

        # ── Source confidence scoring ─────────────────────────────
        scored_chunks = self._score_chunks(chunks, question)

        # ── Hallucination fallback ────────────────────────────────
        evidence_sufficient = self._check_evidence(scored_chunks)

        if not evidence_sufficient:
            response = RAGResponse(
                answer="I don't have sufficient evidence in the uploaded documents to answer this question confidently. Please upload relevant documents or rephrase your question.",
                sources=[],
                retrieved_chunks=[],
                latency_ms=round((time.time() - start) * 1000, 2),
                model_used="fallback",
                retrieval_scores=[],
                hallucination_risk="high",
                evidence_sufficient=False
            )
            return response

        # ── Build constrained prompt ──────────────────────────────
        context = self._build_context(scored_chunks)
        prompt = self._build_constrained_prompt(question, context)

        # ── Generate ──────────────────────────────────────────────
        answer = self.llm.generate(prompt, max_tokens=500)

        # ── Build response ────────────────────────────────────────
        sources = []
        for chunk in scored_chunks:
            source_entry = {
                "source": chunk.metadata.get("source", "unknown"),
                "page": chunk.metadata.get("page"),
                "relevance": round(chunk.similarity_score, 4),
                "confidence": chunk.metadata.get("confidence", "medium")
            }
            if source_entry not in sources:
                sources.append(source_entry)

        response = RAGResponse(
            answer=answer,
            sources=sources,
            retrieved_chunks=[{
                "text": c.text[:200] + "..." if len(c.text) > 200 else c.text,
                "score": round(c.similarity_score, 4),
                "metadata": c.metadata
            } for c in scored_chunks],
            latency_ms=round((time.time() - start) * 1000, 2),
            model_used=getattr(self.settings, "LLM_MODEL", "unknown"),
            retrieval_scores=[round(c.similarity_score, 4) for c in scored_chunks],
            hallucination_risk="low" if evidence_sufficient else "high",
            evidence_sufficient=evidence_sufficient
        )

        # ── Cache result ──────────────────────────────────────────
        _query_cache[cache_key] = response
        return response

    def _score_chunks(self, chunks: List[RetrievedChunk], question: str) -> List[RetrievedChunk]:
        """Add confidence scoring based on similarity score."""
        for chunk in chunks:
            score = chunk.similarity_score
            if score >= 0.7:
                chunk.metadata["confidence"] = "high"
            elif score >= 0.5:
                chunk.metadata["confidence"] = "medium"
            else:
                chunk.metadata["confidence"] = "low"
        return sorted(chunks, key=lambda x: x.similarity_score, reverse=True)

    def _check_evidence(self, chunks: List[RetrievedChunk]) -> bool:
        """Return False if evidence is too weak to answer reliably."""
        if not chunks:
            return False
        top_score = chunks[0].similarity_score if chunks else 0
        return top_score >= self.MIN_EVIDENCE_SCORE

    def _build_context(self, chunks: List[RetrievedChunk]) -> str:
        parts = []
        for i, chunk in enumerate(chunks, 1):
            source = chunk.metadata.get("source", "unknown")
            parts.append(f"[Source {i}: {source}]\n{chunk.text}")
        return "\n\n".join(parts)

    def _build_constrained_prompt(self, question: str, context: str) -> str:
        """Constrained prompt — forces model to use only provided context."""
        return f"""You are a precise document assistant. Answer the question using ONLY the information provided in the context below.

STRICT RULES:
- Use ONLY the provided context to answer
- If the context does not contain enough information, say "The provided documents do not contain sufficient information to answer this question."
- Do NOT use any outside knowledge
- Keep your answer concise and factual
- Cite which source (Source 1, Source 2, etc.) supports each key point

CONTEXT:
{context}

QUESTION: {question}

ANSWER (based only on the context above):"""
