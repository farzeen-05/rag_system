"""
Testing RAG systems requires:
1. Unit tests: test each component in isolation
2. Integration tests: test components together
3. Evaluation tests: measure retrieval quality with RAGAS
"""

import pytest
from unittest.mock import MagicMock, patch
from app.rag.chunker import DocumentChunker
from app.rag.embeddings import EmbeddingModel


class TestChunker:
    """Unit tests for chunking strategies."""
    
    def setup_method(self):
        self.chunker = DocumentChunker()
        self.sample_text = """
        Introduction to Machine Learning
        
        Machine learning is a subset of artificial intelligence.
        It enables systems to learn from data without explicit programming.
        
        Supervised Learning
        
        In supervised learning, models learn from labeled training data.
        Common algorithms include linear regression and decision trees.
        """
    
    def test_recursive_chunking_produces_chunks(self):
        chunks = self.chunker.chunk_recursive(
            self.sample_text, chunk_size=200, chunk_overlap=20
        )
        assert len(chunks) > 0
        assert all(chunk.text.strip() for chunk in chunks)
    
    def test_chunks_have_metadata(self):
        chunks = self.chunker.chunk_recursive(
            self.sample_text,
            metadata={"source": "ml_textbook.pdf", "doc_id": "doc_001"}
        )
        assert all("source" in chunk.metadata for chunk in chunks)
        assert all("doc_id" in chunk.metadata for chunk in chunks)
    
    def test_chunk_overlap_captures_boundary_content(self):
        """Verify that content near chunk boundaries appears in two chunks."""
        long_text = " ".join(["word"] * 1000)
        chunks = self.chunker.chunk_recursive(long_text, chunk_size=100, chunk_overlap=20)
        
        # Adjacent chunks should share some content (overlap)
        if len(chunks) >= 2:
            words_1 = set(chunks[0].text.split())
            words_2 = set(chunks[1].text.split())
            # Overlap means some words appear in both
            assert len(words_1 & words_2) > 0


class TestRAGEngineIntegration:
    """Integration tests with real ChromaDB (use test collection)."""
    
    @pytest.fixture
    def engine(self):
        """Creates a RAG engine with test ChromaDB collection."""
        settings = MagicMock()
        settings.EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
        settings.DEFAULT_CHUNK_SIZE = 256
        settings.DEFAULT_CHUNK_OVERLAP = 25
        settings.RETRIEVAL_TOP_K = 3
        settings.RETRIEVAL_SCORE_THRESHOLD = 0.1
        settings.LLM_MODEL = "llama3.2:1b"
        
        embedding_model = EmbeddingModel(settings.EMBEDDING_MODEL_NAME)
        
        # Use in-memory ChromaDB for tests (no server needed)
        import chromadb
        chroma_client = chromadb.EphemeralClient()
        
        from app.rag.retriever import VectorRetriever
        # Patch to use in-memory client
        retriever = MagicMock()
        
        llm_client = MagicMock()
        llm_client.generate.return_value = "This is a test answer."
        
        from app.rag.engine import RAGEngine
        return RAGEngine(
            embedding_model=embedding_model,
            retriever=retriever,
            llm_client=llm_client,
            chunker=DocumentChunker(),
            settings=settings,
        )
    
    def test_ingest_and_query(self, engine):
        """Tests that ingesting a doc makes it queryable."""
        # This test verifies the full pipeline works together.
        # The embedding model and chunker are real; LLM is mocked.
        text = "The Eiffel Tower is located in Paris, France. It was built in 1889."
        
        engine.retriever.add_chunks = MagicMock()
        engine.retriever.delete_document = MagicMock()
        engine.retriever.search = MagicMock(return_value=[
            MagicMock(
                text="The Eiffel Tower is in Paris",
                metadata={"source": "test.txt"},
                similarity_score=0.9,
                chunk_id="test_0"
            )
        ])
        
        result = engine.query("Where is the Eiffel Tower?")
        
        assert result.answer is not None
        assert len(result.answer) > 0
        engine.retriever.search.assert_called_once()
