# document_loader.py

import os
import logging
from typing import List, Any
from langchain_core.documents import Document
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader, UnstructuredWordDocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Handles loading and initial preparation (splitting) of documents."""

    def __init__(self, content_path: str):
        self.content_path = content_path

    def _get_loader(self, file_path: str) -> Any:
        """Returns the appropriate LangChain document loader based on file extension."""
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".pdf":
            return PyMuPDFLoader(file_path)
        elif ext == ".txt":
            return TextLoader(file_path, encoding="utf-8")
        elif ext in [".docx", ".doc"]:
            return UnstructuredWordDocumentLoader(file_path)
        else:
            logger.warning(f"Skipping unsupported file format: {file_path}")
            return None

    def load_raw_documents(self) -> List[Document]:
        """Traverses the content directory and loads documents, adding source metadata."""
        all_docs = []
        logger.info(f"Starting content extraction from directory: {self.content_path}")
        
        if not os.path.exists(self.content_path):
            logger.error(f"Content directory not found: {self.content_path}")
            raise FileNotFoundError(f"Content path not found: {self.content_path}")

        for root, _, files in os.walk(self.content_path):
            for file_name in files:
                file_path = os.path.join(root, file_name)
                loader = self._get_loader(file_path)
                
                if loader:
                    try:
                        docs = loader.load()
                        for doc in docs:
                            doc.metadata['source'] = file_path
                            doc.metadata['file_name'] = file_name
                        all_docs.extend(docs)
                        logger.info(f"Successfully loaded {len(docs)} pages/parts from: {file_name}")
                    except Exception as e:
                        logger.error(f"Failed to load content from {file_name}: {e}")

        logger.info(f"Total documents loaded: {len(all_docs)}")
        return all_docs
    
    def chunk_documents(self, documents: List[Document], chunk_size: int, chunk_overlap: int) -> List[Document]:
        """Splits loaded documents into smaller, manageable chunks."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
        chunks = splitter.split_documents(documents)
        logger.info(f"Total chunks created: {len(chunks)}")
        return chunks