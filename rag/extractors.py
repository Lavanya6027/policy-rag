import os
import fitz
import docx
from abc import ABC, abstractmethod

class DocumentExtractor(ABC):
    @abstractmethod
    def extract(self, path: str) -> str:
        pass

class PDFExtractor(DocumentExtractor):
    def extract(self, path: str) -> str:
        with fitz.open(path) as doc:
            return "\n".join(page.get_text("text") for page in doc)

class DocxExtractor(DocumentExtractor):
    def extract(self, path: str) -> str:
        d = docx.Document(path)
        return "\n".join(p.text for p in d.paragraphs)

class TxtExtractor(DocumentExtractor):
    def extract(self, path: str) -> str:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

class ExtractorDispatcher:
    _registry = {}

    @classmethod
    def register(cls, ext: str, extractor: DocumentExtractor):
        cls._registry[ext.lower()] = extractor

    @classmethod
    def extract(cls, path: str) -> str:
        ext = os.path.splitext(path)[-1].lower()
        if ext not in cls._registry:
            raise ValueError(f"Unsupported file type: {ext}")
        return cls._registry[ext].extract(path)
