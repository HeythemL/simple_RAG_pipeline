import os
import unicodedata
from typing import List

import pypdf

from interface.base_datastore import DataItem
from interface.base_indexer import BaseIndexer

CHUNK_SIZE = 500


class Indexer(BaseIndexer):

    def index(self, document_paths: List[str]) -> List[DataItem]:
        items: List[DataItem] = []

        for path in document_paths:
            ext = os.path.splitext(path)[-1].lower()

            if ext == ".pdf":
                chunks = self._simple_pdf_chunker(path)
            else:
                raise ValueError(f"Unsupported file type '{ext}' for: {path}")

            for chunk in chunks:
                items.append(DataItem(content=chunk, source=path))

        return items


    def _simple_pdf_chunker(self, pdf_path: str) -> List[str]:
        """Extract text from a PDF and split it into fixed-size character chunks."""
        print(f"Attempting to read PDF at: {pdf_path}")
        reader = pypdf.PdfReader(pdf_path)

        full_text = ""
        for page in reader.pages:
            page_text = page.extract_text() or ""
            full_text += page_text + " "

        # NFKC normalization expands ligatures (fi, fl, ffi, ff -> standard chars)
        # and other compatibility characters, making text readable by LLMs
        full_text = unicodedata.normalize("NFKC", full_text)
        full_text = " ".join(full_text.split())

        chunks = [
            full_text[i : i + CHUNK_SIZE]
            for i in range(0, len(full_text), CHUNK_SIZE)
            if full_text[i : i + CHUNK_SIZE].strip()
        ]

        return chunks
