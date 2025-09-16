import re
from typing import List

class Normalizer:
    def normalize(self, text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

class Chunker:
    def chunk(self, text: str, chunk_size=300, overlap=50) -> List[str]:
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunks.append(" ".join(words[i:i+chunk_size]))
        return chunks
