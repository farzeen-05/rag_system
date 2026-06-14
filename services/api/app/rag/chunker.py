
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
    MarkdownHeaderTextSplitter,
)
from typing import List, Dict, Any
from dataclasses import dataclass
import re


@dataclass
class Chunk:
    """
    A chunk is NOT just text. It carries metadata.
    Metadata is crucial for:
    1. Filtering ("only search policy docs")
    2. Citations ("this comes from page 5 of policy.pdf")
    3. Debugging ("why did we retrieve this?")
    """
    text: str
    metadata: Dict[str, Any]
    chunk_index: int
    total_chunks: int
    source_doc_id: str


class DocumentChunker:
    """
    Unified interface for all chunking strategies.
    Choose strategy based on your document type.
    """
    
    def chunk_recursive(
        self,
        text: str,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        metadata: Dict = None
    ) -> List[Chunk]:
        """
        RECURSIVE CHARACTER SPLITTING — The Default Choice
        
        How it works:
        1. Try to split on ["\n\n", "\n", ". ", " ", ""]
        2. Start with \n\n (paragraph breaks) — most natural
        3. If a paragraph is still too big, split on \n
        4. If still too big, split on ". "
        5. Last resort: split mid-word on " " or even ""
        
        WHY RECURSIVE?
        It respects natural document structure as long as possible,
        only falling back to harder splits when necessary.
        
        chunk_size: measured in CHARACTERS here (not tokens)
        For 512 tokens, use ~2000 characters (1 token ≈ 4 chars)
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size * 4,      # ~4 chars per token
            chunk_overlap=chunk_overlap * 4,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
            length_function=len,
        )
        
        texts = splitter.split_text(text)
        return self._wrap_chunks(texts, metadata or {})
    
    def chunk_by_tokens(
        self,
        text: str,
        chunk_size: int = 256,
        chunk_overlap: int = 32,
        metadata: Dict = None
    ) -> List[Chunk]:
        """
        TOKEN-BASED SPLITTING — Precise for LLM Context Windows
        
        WHY TOKENS INSTEAD OF CHARACTERS?
        LLMs process tokens, not characters.
        "unbelievable" = 1 word, but could be 3-4 tokens.
        
        Uses the SAME tokenizer as the embedding model.
        This ensures each chunk fits exactly in the model's context.
        
        Best for: when you need precise token counts (e.g., fine-tuned RAG)
        """
        splitter = SentenceTransformersTokenTextSplitter(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        texts = splitter.split_text(text)
        return self._wrap_chunks(texts, metadata or {})
    
    def chunk_markdown(
        self,
        markdown_text: str,
        metadata: Dict = None
    ) -> List[Chunk]:
        """
        MARKDOWN-AWARE SPLITTING — For Structured Docs
        
        WHY HEADER-AWARE?
        Markdown headings define sections. Splitting within a section
        is fine, but losing track of WHICH section a chunk belongs to
        destroys context.
        
        This splitter:
        1. Splits on ##, ###, etc. first
        2. Preserves header info as metadata on each chunk
        
        So chunk.metadata will have:
        {"Header 1": "Installation Guide", "Header 2": "Step 3: Configure"}
        
        Now your LLM knows WHERE in the document the answer came from.
        """
        headers = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers)
        docs = splitter.split_text(markdown_text)
        
        chunks = []
        for i, doc in enumerate(docs):
            combined_metadata = {**(metadata or {}), **doc.metadata}
            chunks.append(Chunk(
                text=doc.page_content,
                metadata=combined_metadata,
                chunk_index=i,
                total_chunks=len(docs),
                source_doc_id=metadata.get("doc_id", "unknown")
            ))
        return chunks
    
    def chunk_semantic(
        self,
        sentences: List[str],
        embedding_model,
        threshold: float = 0.3,
        metadata: Dict = None
    ) -> List[Chunk]:
        """
        SEMANTIC CHUNKING — The Advanced Choice
        
        How it works:
        1. Embed every sentence
        2. Compute cosine distance between consecutive sentences
        3. Where distance SPIKES → topic has changed → split here
        
        Example:
        "The sky is blue."      → [0.2, 0.8, ...]
        "Clouds form from water."→ [0.3, 0.7, ...]  ← similar to above
        "Python was invented..."→ [-0.5, 0.1, ...] ← VERY different → SPLIT
        "It is widely used..."  → [-0.4, 0.2, ...]  ← similar to Python sentence
        
        TRADEOFF:
        ✓ Best semantic coherence
        ✗ Slow (N embedding calls during indexing)
        ✗ Non-deterministic chunk boundaries
        
        Use for: high-quality offline indexing where speed doesn't matter
        """
        if len(sentences) < 2:
            return self._wrap_chunks(sentences, metadata or {})
        
        # Embed all sentences
        embeddings = embedding_model.embed_texts(sentences)
        
        # Find topic boundaries where similarity drops below threshold
        chunks_text = []
        current_chunk = [sentences[0]]
        
        for i in range(1, len(sentences)):
            import numpy as np
            prev_vec = np.array(embeddings[i-1])
            curr_vec = np.array(embeddings[i])
            similarity = float(np.dot(prev_vec, curr_vec))
            
            if similarity < threshold:
                # Low similarity = topic changed = new chunk
                chunks_text.append(" ".join(current_chunk))
                current_chunk = [sentences[i]]
            else:
                current_chunk.append(sentences[i])
        
        if current_chunk:
            chunks_text.append(" ".join(current_chunk))
        
        return self._wrap_chunks(chunks_text, metadata or {})
    
    def _wrap_chunks(self, texts: List[str], metadata: Dict) -> List[Chunk]:
        """Helper to wrap raw text strings into Chunk objects."""
        return [
            Chunk(
                text=text,
                metadata={**metadata, "chunk_index": i},
                chunk_index=i,
                total_chunks=len(texts),
                source_doc_id=metadata.get("doc_id", "unknown")
            )
            for i, text in enumerate(texts)
            if text.strip()  # Skip empty chunks
        ]