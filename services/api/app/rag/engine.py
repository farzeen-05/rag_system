"""
THE HEART OF RAG.
This class orchestrates the entire pipeline:
  ingest documents → index → retrieve → generate

Think of it as the "director" — it tells embeddings,
chunker, retriever, and LLM what to do and when.
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import logging
import time

from rag.embeddings import EmbeddingModel
from rag.chunker import DocumentChunker, Chunk
from rag.retriever import VectorRetriever, RetrievedChunk

logger = logging.getLogger(__name__)


@dataclass
class RAGResponse:
    """
    WHY A STRUCTURED RESPONSE?
    The API shouldn't just return the answer string.
    It should return everything the caller needs:
    - The answer
    - The sources used (for citations)
    - The retrieved chunks (for debugging)
    - Performance metrics (for monitoring)
    """
    answer: str
    sources: List[Dict[str, Any]]      # Which docs contributed
    retrieved_chunks: List[Dict]        # Raw chunks (for debugging)
    query: str
    latency_ms: float
    model_used: str
    retrieval_scores: List[float]


class RAGEngine:
    """
    Core RAG orchestrator.
    
    Dependency injection pattern:
    We pass in embedding_model, retriever, llm_client instead of 
    creating them inside. This makes testing easy — swap real components
    with mock objects in tests.
    """
    
    def __init__(
        self,
        embedding_model: EmbeddingModel,
        retriever: VectorRetriever,
        llm_client,  # LLM wrapper (OpenAI, Ollama, etc.)
        chunker: DocumentChunker,
        settings,
    ):
        self.embedding_model = embedding_model
        self.retriever = retriever
        self.llm_client = llm_client
        self.chunker = chunker
        self.settings = settings
    
    # ── INDEXING ─────────────────────────────────────────────────────
    
    def ingest_document(
        self,
        text: str,
        doc_id: str,
        metadata: Dict[str, Any],
        chunking_strategy: str = "recursive"
    ) -> Dict[str, Any]:
        """
        Full indexing pipeline for one document.
        
        Steps:
        1. Chunk the document using chosen strategy
        2. Add doc_id to each chunk's metadata
        3. Embed all chunks in one batch
        4. Store in ChromaDB
        
        Returns stats about what was indexed.
        """
        start = time.time()
        
        # Step 1: Chunk
        logger.info(f"Chunking document {doc_id} with strategy: {chunking_strategy}")
        
        chunk_metadata = {
            "doc_id": doc_id,
            **metadata  # Merge caller-provided metadata
        }
        
        # Choose chunking strategy based on parameter
        if chunking_strategy == "recursive":
            chunks = self.chunker.chunk_recursive(
                text, 
                chunk_size=self.settings.DEFAULT_CHUNK_SIZE,
                chunk_overlap=self.settings.DEFAULT_CHUNK_OVERLAP,
                metadata=chunk_metadata
            )
        elif chunking_strategy == "tokens":
            chunks = self.chunker.chunk_by_tokens(text, metadata=chunk_metadata)
        elif chunking_strategy == "markdown":
            chunks = self.chunker.chunk_markdown(text, metadata=chunk_metadata)
        elif chunking_strategy == "semantic":
            sentences = text.split(". ")
            chunks = self.chunker.chunk_semantic(
                sentences, self.embedding_model, metadata=chunk_metadata
            )
        else:
            raise ValueError(f"Unknown chunking strategy: {chunking_strategy}")
        
        if not chunks:
            logger.warning(f"No chunks generated for document {doc_id}")
            return {"doc_id": doc_id, "chunks_indexed": 0}
        
        # Step 2: Embed all chunks in one batch (fast!)
        logger.info(f"Embedding {len(chunks)} chunks...")
        texts = [chunk.text for chunk in chunks]
        embeddings = self.embedding_model.embed_texts(texts)
        
        # Step 3: Delete existing chunks if updating document
        self.retriever.delete_document(doc_id)
        
        # Step 4: Store in ChromaDB
        self.retriever.add_chunks(chunks, embeddings)
        
        elapsed = time.time() - start
        logger.info(f"Indexed {len(chunks)} chunks in {elapsed:.2f}s")
        
        return {
            "doc_id": doc_id,
            "chunks_indexed": len(chunks),
            "chunking_strategy": chunking_strategy,
            "indexing_time_seconds": round(elapsed, 2),
            "avg_chunk_length": sum(len(c.text) for c in chunks) // len(chunks)
        }
    
    # ── RETRIEVAL + GENERATION ───────────────────────────────────────
    
    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        filter_metadata: Optional[Dict] = None,
        use_mmr: bool = False,
        score_threshold: Optional[float] = None,
    ) -> RAGResponse:
        """
        Full RAG pipeline: question → answer.
        
        This is the function called on every user query.
        """
        start = time.time()
        top_k = top_k or self.settings.RETRIEVAL_TOP_K
        threshold = score_threshold or self.settings.RETRIEVAL_SCORE_THRESHOLD
        
        # ── Step 1: Embed the query ───────────────────────────
        # Use the SAME model used during indexing!
        # If you index with model A and query with model B, results are garbage.
        query_embedding = self.embedding_model.embed_query(question)
        
        # ── Step 2: Retrieve relevant chunks ─────────────────
        if use_mmr:
            # MMR = diverse results (less redundancy)
            retrieved = self.retriever.mmr_search(
                query_embedding, top_k=top_k
            )
        else:
            # Standard similarity search
            retrieved = self.retriever.search(
                query_embedding,
                top_k=top_k,
                score_threshold=threshold,
                filter_metadata=filter_metadata
            )
        
        if not retrieved:
            # No relevant documents found
            return RAGResponse(
                answer="I couldn't find relevant information in the knowledge base to answer your question.",
                sources=[],
                retrieved_chunks=[],
                query=question,
                latency_ms=round((time.time() - start) * 1000, 2),
                model_used=self.settings.LLM_MODEL,
                retrieval_scores=[]
            )
        
        # ── Step 3: Build the prompt ──────────────────────────
        prompt = self._build_prompt(question, retrieved)
        
        # ── Step 4: Generate with LLM ────────────────────────
        answer = self.llm_client.generate(prompt)
        
        # ── Step 5: Build response with sources ──────────────
        sources = self._extract_sources(retrieved)
        
        return RAGResponse(
            answer=answer,
            sources=sources,
            retrieved_chunks=[
                {"text": r.text[:200] + "...", "score": r.similarity_score, 
                 "metadata": r.metadata}
                for r in retrieved
            ],
            query=question,
            latency_ms=round((time.time() - start) * 1000, 2),
            model_used=self.settings.LLM_MODEL,
            retrieval_scores=[r.similarity_score for r in retrieved]
        )
    
    def _build_prompt(self, question: str, chunks: List[RetrievedChunk]) -> str:
        """
        Assemble the final prompt sent to the LLM.
        
        PROMPT DESIGN DECISIONS:
        1. System message: constrain LLM to use ONLY context (faithfulness)
        2. Context block: label each source for transparency
        3. Question last: LLM sees context before question (better attention)
        4. Format instruction: tells LLM how to structure the answer
        
        The "do not use prior knowledge" instruction is critical.
        Without it, the LLM will mix retrieved facts with hallucinations.
        """
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            source = chunk.metadata.get("source", "Unknown")
            page = chunk.metadata.get("page", "")
            page_str = f", Page {page}" if page else ""
            score = chunk.similarity_score
            
            context_parts.append(
                f"[Source {i}: {source}{page_str} | Relevance: {score:.2f}]\n"
                f"{chunk.text}"
            )
        
        context = "\n\n---\n\n".join(context_parts)
        
        return f"""You are a precise, helpful assistant. Answer the question using ONLY the provided context.

Rules:
- Base your answer entirely on the provided context
- If the context doesn't answer the question, say: "The available documents don't contain information about this."
- Quote or reference specific sources when possible
- Be concise but complete
- Do NOT use any prior knowledge

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""
    
    def _extract_sources(self, chunks: List[RetrievedChunk]) -> List[Dict]:
        """Deduplicate sources for the response citations."""
        seen = set()
        sources = []
        for chunk in chunks:
            source = chunk.metadata.get("source", "Unknown")
            if source not in seen:
                seen.add(source)
                sources.append({
                    "source": source,
                    "page": chunk.metadata.get("page"),
                    "relevance": chunk.similarity_score
                })
        return sources
