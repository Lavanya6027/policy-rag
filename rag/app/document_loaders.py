# -----------------------------
# Document Loader / JSONChunkLoader
# -----------------------------
import json
from pathlib import Path
from typing import List, Protocol
from langchain.schema import Document


class DocumentLoader(Protocol):
    def load(self) -> List[Document]:
        ...


class JSONChunkLoader:
    """Concrete strategy for loading docs from a single JSON chunk file."""

    def __init__(self, json_path: Path):
        self._json_path = json_path

    def load(self) -> List[Document]:
        chunks = self._load_chunks_from_file(self._json_path)
        return self._chunks_to_documents(chunks)

    @staticmethod
    def _load_chunks_from_file(json_path: Path):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict) and "chunks" in data:
            data = data["chunks"]

        if not isinstance(data, list):
            raise ValueError(f"Unexpected format in {json_path}")
        return data

    @staticmethod
    def _chunks_to_documents(chunks) -> List[Document]:
        docs = []
        for c in chunks:
            metadata = {
                "source_name": c.get("source_name", ""),
                "title": c.get("metadata", {}).get("title", "")
            }

            # Include id only if present
            if "id" in c:
                metadata["id"] = c["id"]

            topics = c.get("metadata", {}).get("topics", "")
            if isinstance(topics, list):
                topics = ", ".join(topics)
            metadata["topics"] = topics

            docs.append(Document(page_content=c["text"], metadata=metadata))
        return docs

