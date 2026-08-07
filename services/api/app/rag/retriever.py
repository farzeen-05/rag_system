
"""
WHY CHROMADB?
ChromaDB is a purpose-built vector database.
It stores vectors with metadata and supports:
- Cosine/L2/dot product similarity
- Metadata filtering ("only search docs from 2024")
- Persistent storage
- Client-server mode (perfect for Docker/Kubernetes)

Alternative: Pinecone (managed, expensive), Weaviate (complex), FAISS (no persistence)
"""

import chromadb
from chromadb.config import Settings as ChromaSettings
from rank_bm25 import BM25Okapi
from typing import List, Dict, Any, Optional
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass  
class RetrievedChunk:
    """Result from vector search, enriched with score."""
    text: str
    metadata: Dict[str, Any]
    similarity_score: float
    chunk_id: str


class VectorRetriever:
    """
    Manages ChromaDB operations: store, search, delete.
    
    ChromaDB concepts:
    - Collection: like a table in SQL, or an index in Elasticsearch
    - Document: the raw text we indexed
    - Embedding: the vector representation
    - Metadata: filterable key-value pairs
    - ID: unique identifier for each entry
    """
    
    def __init__(self, host: str, port: int, collection_name: str):
        """
        Connect to ChromaDB running as a separate server.
        
        WHY HTTP CLIENT?
        In local dev: ChromaDB runs on your machine
        In Docker Compose: ChromaDB is a container named "chroma"
        In Kubernetes: ChromaDB is a Service at chroma-service:8000
        
        The HTTP client abstracts ALL of these — same code, different host.
        """
        self.client = chromadb.HttpClient(
            host=host,
            port=port,
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        
        # get_or_create_collection:
        # First run → creates the collection
        # Subsequent runs → fetches existing collection (idempotent!)
        # This is important: your app won't crash if ChromaDB already has data
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            # Distance function for similarity:
            # "cosine": angle between vectors (best for normalized embeddings)
            # "l2": euclidean distance (good for unnormalized)
            # "ip": inner product (same as cosine for unit vectors)
            metadata={"hnsw:space": "cosine"}
        )
        
        logger.info(f"Connected to ChromaDB | Collection: {collection_name} | "
                   f"Documents: {self.collection.count()}")
    
    def add_chunks(self, chunks: List, embeddings: List[List[float]]) -> None:
        """
        Store chunks + their vectors in ChromaDB.
        
        ChromaDB add() requires parallel lists:
        - ids: unique string IDs
        - embeddings: vectors
        - documents: original text
        - metadatas: dicts with filterable info
        
        WHY BATCH? ChromaDB is optimized for batch inserts.
        Single inserts create overhead (disk I/O, index updates).
        Batch everything under 5000 items at once.
        """
        batch_size = 500  # ChromaDB handles up to 5000, but 500 is safe
        
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i+batch_size]
            batch_embeddings = embeddings[i:i+batch_size]
            
            self.collection.add(
                ids=[f"{c.source_doc_id}_{c.chunk_index}" for c in batch_chunks],
                embeddings=batch_embeddings,
                documents=[c.text for c in batch_chunks],
                metadatas=[c.metadata for c in batch_chunks],
            )
        
        logger.info(f"Added {len(chunks)} chunks to ChromaDB")
    
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        score_threshold: float = 0.3,
        filter_metadata: Optional[Dict] = None
    ) -> List[RetrievedChunk]:
        """
        Perform vector similarity search.
        
        WHAT HAPPENS UNDER THE HOOD:
        ChromaDB uses HNSW (Hierarchical Navigable Small World) algorithm.
        This is an approximate nearest neighbor (ANN) algorithm.
        
        WHY NOT EXACT SEARCH?
        Exact search over 1M vectors: check all 1M, O(n) 
        HNSW: navigates a multi-layer graph, O(log n)
        For 1M vectors: exact = ~1 second, HNSW = ~1ms
        
        HNSW builds a graph where similar vectors are connected.
        Search = start from random entry point, greedily move to 
        closer neighbors layer by layer.
        
        where_filter example:
        {"source": {"$eq": "policy.pdf"}}  → only search policy docs
        {"year": {"$gte": 2023}}           → only recent docs
        {"$and": [{"source": "..."}, ...]} → combine conditions
        """
        available = self.collection.count()
        if available == 0:
            return []

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k * 2, available),  # Fetch extra for threshold filtering
            where=filter_metadata,
            include=["documents", "metadatas", "distances"]
        )
        
        retrieved = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            # ChromaDB cosine distance = 1 - cosine_similarity
            # Convert: similarity = 1 - distance
            similarity = 1 - dist
            
            if similarity >= score_threshold:
                retrieved.append(RetrievedChunk(
                    text=doc,
                    metadata=meta,
                    similarity_score=round(similarity, 4),
                    chunk_id=meta.get("chunk_id", "")
                ))
        
        # Sort by similarity descending, take top_k
        retrieved.sort(key=lambda x: x.similarity_score, reverse=True)
        return retrieved[:top_k]
    
    def mmr_search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        fetch_k: int = 20,
        lambda_mult: float = 0.5
    ) -> List[RetrievedChunk]:
        """
        MAXIMAL MARGINAL RELEVANCE (MMR) Search
        
        Problem MMR solves:
        Query: "How do I reset my password?"
        Top-5 results: all 5 chunks say same thing from same paragraph.
        This wastes context window — you sent 5 duplicates to the LLM.
        
        MMR algorithm:
        1. Fetch top-20 candidates by similarity
        2. Greedily select chunks that are:
           - Relevant to query (high similarity to query)
           - Different from already selected (low similarity to selected)
        3. Score = λ * relevance - (1-λ) * redundancy
        
        lambda_mult:
        - 1.0 = pure relevance (same as regular search)
        - 0.0 = pure diversity (maximally different results)
        - 0.5 = balanced (recommended)
        
        Result: 5 chunks that cover DIFFERENT aspects of the topic.
        """
        import numpy as np
        
        # Step 1: Fetch more candidates than we need
        candidates = self.search(query_embedding, top_k=fetch_k, score_threshold=0.0)
        
        if not candidates:
            return []
        
        # Step 2: MMR selection
        selected = []
        candidate_embeddings = []  # We'd need to fetch these from ChromaDB
        
        # Simplified MMR using similarity scores
        # (Full MMR would re-embed and compute pairwise distances)
        selected_indices = set()
        query_vec = np.array(query_embedding)
        
        for _ in range(min(top_k, len(candidates))):
            best_score = -float('inf')
            best_idx = None
            
            for i, candidate in enumerate(candidates):
                if i in selected_indices:
                    continue
                
                # Relevance to query
                relevance = candidate.similarity_score
                
                # Redundancy with already selected
                if selected:
                    redundancy = max(
                        self._text_similarity(candidate.text, s.text)
                        for s in selected
                    )
                else:
                    redundancy = 0
                
                mmr_score = lambda_mult * relevance - (1 - lambda_mult) * redundancy
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i
            
            if best_idx is not None:
                selected.append(candidates[best_idx])
                selected_indices.add(best_idx)
        
        return selected
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Simple Jaccard similarity for MMR redundancy check."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        if not words1 or not words2:
            return 0.0
        return len(words1 & words2) / len(words1 | words2)
    
    def _rebuild_bm25_index(self):
        """Rebuild in-memory BM25 index from all documents in ChromaDB.
        WHY BM25? Vector search finds semantically similar text but can miss
        exact keyword matches (e.g. product codes, names, acronyms).
        BM25 is a classic keyword-ranking algorithm — combining both gives
        better recall than either alone."""
        try:
            all_data = self.collection.get(include=["documents", "metadatas"])
            self._bm25_docs = all_data["documents"]
            self._bm25_metas = all_data["metadatas"]
            self._bm25_ids = all_data["ids"]
            if self._bm25_docs:
                tokenized = [doc.lower().split() for doc in self._bm25_docs]
                self._bm25_index = BM25Okapi(tokenized)
            else:
                self._bm25_index = None
        except Exception as e:
            logger.warning(f"BM25 index rebuild failed: {e}")
            self._bm25_index = None
            self._bm25_docs = []
            self._bm25_metas = []
            self._bm25_ids = []

    def hybrid_search(
        self,
        query_text: str,
        query_embedding: List[float],
        top_k: int = 5,
        alpha: float = 0.8,
        score_threshold: float = 0.3,
        filter_metadata: Optional[Dict] = None
    ) -> List[RetrievedChunk]:
        """Hybrid retrieval: combines dense vector search with BM25 keyword search.
        alpha: weight for vector score (1-alpha for BM25 score). 0.6 = favor semantic
        slightly while still catching exact keyword matches BM25 finds."""
        vector_results = self.search(
            query_embedding, top_k=top_k * 3,
            score_threshold=0.0, filter_metadata=filter_metadata
        )

        if not hasattr(self, '_bm25_index') or self._bm25_index is None:
            self._rebuild_bm25_index()

        bm25_scores_by_text = {}
        if self._bm25_index is not None and self._bm25_docs:
            tokenized_query = query_text.lower().split()
            bm25_scores = self._bm25_index.get_scores(tokenized_query)
            max_bm25 = max(bm25_scores) if len(bm25_scores) > 0 and max(bm25_scores) > 0 else 1.0
            for doc, score in zip(self._bm25_docs, bm25_scores):
                normalized = score / max_bm25 if max_bm25 > 0 else 0.0
                bm25_scores_by_text[doc] = max(0.0, min(1.0, normalized))  # clamp to 0-1

        combined = []
        for chunk in vector_results:
            bm25_score = bm25_scores_by_text.get(chunk.text, 0.0)
            combined_score = alpha * chunk.similarity_score + (1 - alpha) * bm25_score
            chunk.metadata["vector_score"] = chunk.similarity_score
            chunk.metadata["bm25_score"] = round(bm25_score, 4)
            chunk.similarity_score = round(combined_score, 4)
            if combined_score >= score_threshold:
                combined.append(chunk)

        combined.sort(key=lambda x: x.similarity_score, reverse=True)
        return combined[:top_k]

    def rerank(self, chunks: List[RetrievedChunk], query_text: str, top_k: int = 5) -> List[RetrievedChunk]:
        """Lightweight second-stage reranking.
        WHY TWO-STAGE? Stage 1 (ANN/BM25) is fast but approximate — good for
        narrowing 1000s of docs to ~20 candidates. Stage 2 reranking does a
        more careful, query-aware scoring pass over just those candidates,
        which is too slow to run over the whole corpus but fine for 20 items."""
        query_terms = set(query_text.lower().split())

        for chunk in chunks:
            text_terms = set(chunk.text.lower().split())
            term_overlap = len(query_terms & text_terms) / max(len(query_terms), 1)
            # Reward chunks with more exact term overlap AND high base similarity
            rerank_score = 0.7 * chunk.similarity_score + 0.3 * term_overlap
            chunk.metadata["rerank_score"] = round(rerank_score, 4)
            chunk.similarity_score = round(rerank_score, 4)

        chunks.sort(key=lambda x: x.similarity_score, reverse=True)
        return chunks[:top_k]

    def delete_document(self, doc_id: str) -> None:
        """Delete all chunks from a document (for updates/re-indexing)."""
        self.collection.delete(where={"doc_id": {"$eq": doc_id}})
        logger.info(f"Deleted document: {doc_id}")
    
    def get_stats(self) -> Dict:
        """Returns collection statistics for monitoring."""
        return {
            "total_chunks": self.collection.count(),
            "collection_name": self.collection.name,
        }
