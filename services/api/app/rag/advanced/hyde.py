"""
PROBLEM: "What are the side effects of aspirin?"
  - Query embedding: medical question vector
  - But indexed docs might phrase it as: "aspirin may cause..."
  - The ANSWER format is different from the QUESTION format.

HYDE SOLUTION:
1. Ask LLM: "Write a paragraph that would answer: {question}"
2. Embed the HYPOTHETICAL ANSWER instead of the question
3. This hypothetical answer is in the same "answer space" as your indexed docs
4. Better semantic match!

WHY IT WORKS:
The hypothetical answer uses the same vocabulary and structure
as actual documents, so it embeds closer to relevant chunks.
"""

class HyDERetriever:
    def __init__(self, llm_client, embedding_model, retriever):
        self.llm = llm_client
        self.embedder = embedding_model
        self.retriever = retriever
    
    def retrieve(self, question: str, top_k: int = 5):
        # Step 1: Generate hypothetical document
        hyde_prompt = f"""Write a short, factual paragraph that directly answers this question.
Write as if you are an expert. Be specific and informative.
Do not say "I think" or "The answer is". Just write the content.

Question: {question}

Paragraph:"""
        
        hypothetical_doc = self.llm.generate(hyde_prompt, max_tokens=200)
        
        # Step 2: Embed the hypothetical document
        hyp_embedding = self.embedder.embed_query(hypothetical_doc)
        
        # Step 3: Retrieve using hypothetical embedding
        return self.retriever.search(hyp_embedding, top_k=top_k)


### 13.2 Self-Query (Structured Filtering)

"""
PROBLEM: User asks "Show me legal documents from 2023"
  - Naive RAG: embeds whole question, hopes metadata filtering happens
  - Better: parse the query into semantic part + filter part

SELF-QUERY:
1. Ask LLM to decompose: 
   "legal documents from 2023" →
   semantic: "legal documents"
   filter: {"year": {"$eq": 2023}, "type": {"$eq": "legal"}}

2. Use semantic part for vector search
3. Use filter for ChromaDB metadata filtering
"""

def self_query(question: str, llm_client, retriever, embedder) -> List:
    decompose_prompt = f"""Extract the search query and filters from this question.
Respond in JSON only.

Question: {question}

Respond as:
{{"semantic_query": "...", "filters": {{"field": {{"$eq": "value"}}}}}}

If no filters, use: {{"semantic_query": "...", "filters": null}}"""
    
    result = llm_client.generate(decompose_prompt, max_tokens=100)
    import json
    parsed = json.loads(result)
    
    embedding = embedder.embed_query(parsed["semantic_query"])
    return retriever.search(embedding, filter_metadata=parsed.get("filters"))


### 13.3 Contextual Compression

"""
PROBLEM: You retrieve a 500-token chunk but only 50 tokens are relevant.
You're wasting 450 tokens of LLM context window.

SOLUTION: After retrieving, compress each chunk to only the relevant parts.
"""

def compress_chunk(chunk_text: str, question: str, llm_client) -> str:
    compress_prompt = f"""Extract ONLY the sentences from the context that are relevant to the question.
If nothing is relevant, respond with: "NOT RELEVANT"
Do not add any text. Only return sentences from the original.

Question: {question}

Context: {chunk_text}

Relevant sentences:"""
    
    return llm_client.generate(compress_prompt, max_tokens=300)

### 13.4 Multi-Query Retrieval
"""
PROBLEM: One query vector might miss relevant docs phrased differently.

SOLUTION: Generate 3-5 variations of the query, retrieve for each,
merge results (deduplicate), get better coverage.
"""

def multi_query_retrieve(question: str, llm_client, embedder, retriever) -> List:
    variation_prompt = f"""Generate 4 different phrasings of this question.
Each should seek the same information but use different words.
Output as JSON array of strings.

Question: {question}

Variations:"""
    
    import json
    result = llm_client.generate(variation_prompt, max_tokens=200)
    variations = json.loads(result)
    variations.append(question)  # Include original
    
    # Retrieve for each variation
    all_chunks = {}
    for query in variations:
        embedding = embedder.embed_query(query)
        chunks = retriever.search(embedding, top_k=3)
        for chunk in chunks:
            chunk_id = f"{chunk.metadata.get('doc_id')}_{chunk.metadata.get('chunk_index')}"
            if chunk_id not in all_chunks:
                all_chunks[chunk_id] = chunk
    
    # Sort by score, take top-5 unique chunks
    return sorted(all_chunks.values(), key=lambda x: x.similarity_score, reverse=True)[:5]