from typing import List
import numpy as np
from embeddings import Embedder, VectorStore
from preprocessing import Normalizer, Chunker


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

class RAGPipeline:
    def __init__(self, embedder: Embedder, store: VectorStore, chunker: Chunker, normalizer: Normalizer):
        self.embedder = embedder
        self.store = store
        self.chunker = chunker
        self.normalizer = normalizer

    def add_document(self, text: str, source: str):
        text = self.normalizer.normalize(text)
        chunks = self.chunker.chunk(text)
        embeddings = self.embedder.embed(chunks)
        # Store with metadata
        print(f"Adding document: {source} with {len(chunks)} chunks")
        self.store.add(embeddings, chunks, metadata={"source": source})

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        similarity_threshold: float = 0.3,  # Lower default threshold
        keywords: List[str] = None
    ) -> List[str]:
        # Get more candidates to ensure we find relevant chunks
        search_top_k = top_k * 3  # Search for more initially
        query_embedding = self.embedder.embed([query])[0]
        distances, indices = self.store.index.search(np.array([query_embedding], dtype=np.float32), search_top_k)
        
        retrieved = []

        # Heuristic filter: match doc source if query mentions it
        preferred_doc = None
        query_lower = query.lower()
        if "domestic" in query_lower:
            preferred_doc = "Domestic Travel Policy.pdf"
        elif "international" in query_lower:
            preferred_doc = "International Travel Policy.pdf"
        elif "leave" in query_lower:
            preferred_doc = "Leave Policy.pdf"
        elif "referral" in query_lower:
            preferred_doc = "Employee Referral Scheme.pdf"
        
        print(f"Query: '{query}'")
        print(f"Preferred document: {preferred_doc}")
        print(f"Searching {search_top_k} candidates, distances: {distances[0][:5]}...")  # Show first 5

        # Two-pass approach: collect all relevant chunks, then prioritize
        all_relevant_chunks = []
        
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            sim = 1 / (1 + dist)
            chunk_text = self.store.documents[idx]
            chunk_meta = self.store.metadatas[idx] if hasattr(self.store, "metadatas") else {}
            
            if sim < similarity_threshold:
                if i < 5:  # Only print first few skips for debugging
                    print(f"Chunk {idx}: SKIPPED - similarity {sim:.4f} < threshold {similarity_threshold}")
                continue

            # Store all relevant chunks with their metadata
            all_relevant_chunks.append({
                'sim': sim,
                'text': chunk_text,
                'source': chunk_meta.get('source', 'unknown'),
                'is_preferred': preferred_doc and chunk_meta.get('source') == preferred_doc
            })

        # If we have preferred document filtering, prioritize those chunks
        if preferred_doc and all_relevant_chunks:
            # Separate preferred and other chunks
            preferred_chunks = [chunk for chunk in all_relevant_chunks if chunk['is_preferred']]
            other_chunks = [chunk for chunk in all_relevant_chunks if not chunk['is_preferred']]
            
            # Sort both lists by similarity (highest first)
            preferred_chunks.sort(key=lambda x: x['sim'], reverse=True)
            other_chunks.sort(key=lambda x: x['sim'], reverse=True)
            
            # Take top preferred chunks first, then top other chunks
            max_preferred = min(top_k, len(preferred_chunks))
            max_other = top_k - max_preferred
            
            for chunk in preferred_chunks[:max_preferred]:
                retrieved.append(chunk['text'])
                print(f"✓ PRIORITIZED: {chunk['source']} (sim: {chunk['sim']:.4f})")
                
            for chunk in other_chunks[:max_other]:
                retrieved.append(chunk['text'])
                print(f"→ INCLUDED: {chunk['source']} (sim: {chunk['sim']:.4f})")
                
        else:
            # No preferred document, just take top by similarity
            all_relevant_chunks.sort(key=lambda x: x['sim'], reverse=True)
            for chunk in all_relevant_chunks[:top_k]:
                retrieved.append(chunk['text'])
                print(f"→ ACCEPTED: {chunk['source']} (sim: {chunk['sim']:.4f})")

        # Apply keyword filtering if specified
        if keywords and retrieved:
            print(f"Applying keyword filtering for: {keywords}")
            keyword_filtered = []
            for chunk_text in retrieved:
                sentences = chunk_text.split(". ")
                matched = []
                for i, sent in enumerate(sentences):
                    if any(kw.lower() in sent.lower() for kw in keywords):
                        start = max(0, i - 1)
                        end = min(len(sentences), i + 2)
                        matched.append(". ".join(sentences[start:end]))
                if matched:
                    keyword_filtered.extend(matched)
            retrieved = keyword_filtered

        print(f"Retrieved {len(retrieved)} chunks matching criteria.")
        return retrieved[:top_k]

class Chatbot:
    def __init__(self, rag: RAGPipeline):
        self.rag = rag

    def answer(
        self,
        query: str,
        top_k: int = 5,
        similarity_threshold: float = 0.5,
        keywords: List[str] = None
    ) -> str:
        chunks = self.rag.retrieve(
            query,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            keywords=keywords
        )
        if not chunks:
            return "❌ Sorry, no relevant information found in the documents."
        return "\n".join(chunks)
