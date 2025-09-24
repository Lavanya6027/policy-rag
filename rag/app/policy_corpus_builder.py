from pathlib import Path
from typing import List, Protocol, Optional
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from app.document_loaders import JSONChunkLoader


# -----------------------------
# Custom Exceptions
# -----------------------------
class VectorStoreError(Exception):
    """Base exception for vectorstore errors."""
    pass

class DocumentLoadError(VectorStoreError):
    """Raised when documents cannot be loaded properly."""
    pass

class EmbeddingProviderError(VectorStoreError):
    """Raised when embeddings cannot be initialized or retrieved."""
    pass

class VectorStorePersistenceError(VectorStoreError):
    """Raised when the vectorstore fails to persist or access."""
    pass


# -----------------------------
# Interface Segregation + DIP
# -----------------------------
class EmbeddingProvider(Protocol):
    def embed(self, texts: List[str]):
        ...


# -----------------------------
# Concrete Implementation
# -----------------------------
class HuggingFaceEmbeddingProvider:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        try:
            print(f"[INFO] Initializing HuggingFace embeddings with model: {model_name}")
            self._embeddings = HuggingFaceEmbeddings(model_name=model_name)
        except Exception as e:
            raise EmbeddingProviderError(f"Failed to initialize HuggingFace embeddings: {e}")

    def get(self):
        try:
            print("[INFO] Returning embedding object")
            return self._embeddings
        except Exception as e:
            raise EmbeddingProviderError(f"Failed to retrieve embeddings: {e}")


# -----------------------------
# High-level Corpus Loader
# -----------------------------
class PolicyCorpusBuilder:
    def __init__(self, 
                 doc_loader,
                 embedding_provider: HuggingFaceEmbeddingProvider,
                 persist_dir: str = "./chroma_store"):
        self._doc_loader = doc_loader
        self._embedding_provider = embedding_provider
        self._persist_dir = persist_dir
        self._vectorstore: Optional[Chroma] = None

    def load_or_create_vectorstore(self):
        persist_path = Path(self._persist_dir)
        embeddings = self._embedding_provider.get()

        # Check if persisted Chroma store exists
        if persist_path.exists() and any(persist_path.iterdir()):
            try:
                print(f"[INFO] Persisted vectorstore found at {self._persist_dir}. Loading...")
                self._vectorstore = Chroma(persist_directory=str(self._persist_dir), embedding_function=embeddings)
                print("[INFO] Vectorstore successfully loaded from disk!")
                return self._vectorstore
            except Exception as e:
                raise VectorStorePersistenceError(f"Failed to load existing vectorstore: {e}")

        # If not exists, create new vectorstore
        try:
            print("[INFO] No existing vectorstore found. Creating new one...")
            docs = self._doc_loader.load()
            print(f"[INFO] Loaded {len(docs)} documents")

            ids = [d.metadata.get("id") if d.metadata else None for d in docs]
            self._vectorstore = Chroma.from_documents(
                documents=docs,
                embedding=embeddings,
                ids=ids,
                persist_directory=self._persist_dir
            )
            self._vectorstore.persist()
            print("[INFO] New vectorstore created and persisted successfully!")
            return self._vectorstore
        except DocumentLoadError as e:
            raise e
        except Exception as e:
            raise VectorStorePersistenceError(f"Failed to create/persist vectorstore: {e}")

    @property
    def vectorstore(self) -> Chroma:
        if not self._vectorstore:
            raise VectorStoreError("[ERROR] Vectorstore not built yet. Call load_or_create_vectorstore() first.")
        print("[INFO] Returning vectorstore object")
        return self._vectorstore


# -----------------------------
# High-level function to build/load vectorstore from JSON
# -----------------------------
def build_vectorstore_from_json(
    json_path: str,
    persist_dir: str = "./chroma_store",
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
) -> Chroma:
    try:
        print(f"[INFO] Initializing vectorstore from JSON: {json_path}")

        loader = JSONChunkLoader(json_path=Path(json_path))
        embedding_provider = HuggingFaceEmbeddingProvider(model_name=embedding_model)

        corpus_builder = PolicyCorpusBuilder(
            doc_loader=loader,
            embedding_provider=embedding_provider,
            persist_dir=persist_dir
        )

        return corpus_builder.load_or_create_vectorstore()

    except (DocumentLoadError, EmbeddingProviderError, VectorStorePersistenceError, VectorStoreError) as e:
        print(f"[ERROR] {e}")
        raise e
    except Exception as e:
        print(f"[ERROR] Unexpected error building vectorstore: {e}")
        raise VectorStoreError(f"Unexpected error: {e}")


if "__main__" == __name__:
    # Example usage
    vectorstore = build_vectorstore_from_json(
        json_path=r"C:\Users\lavanya.e\Downloads\processed_chunks_with_ids.json",
        persist_dir="./chroma_store",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2"
    )
    print(vectorstore.similarity_search("What is a data policy?", k=2))

    hyrid_retriever = init_policy_retriever(vector_store=vectorstore, reranker=None)

    doc_loader = JSONChunkLoader(json_path=Path(r"C:\Users\lavanya.e\Downloads\processed_chunks_with_ids.json"))
    docs = doc_loader.load()

    context, chunks = hyrid_retriever.retrieve_context(
        "tell me about domestic travel policy",
        docs,
        initial_k=5,
        final_k=3
    )
    question = "category of trainees?"
    query = f"""
You are an AI assistant specialized in answering company policy and HR-related queries.
- Always use the provided context to answer.
- If the answer requires reasoning (e.g., calculating allowance), explain your steps clearly.
- If context is insufficient, say "I could not find this information in the documents."
- Be concise but cover key details.
- Provide references to source documents when possible.
 
Context:
{context}
 
Question:
{question}
""" 

    # generate
    ollama = OllamaAIClient()
    ai_service = AIAnswerService(ollama)
    response = ai_service.get_answer(query=query)
    print(response)






