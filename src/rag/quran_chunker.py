from typing import Annotated
from annotated_types import Ge, Le
import re

from quran_chunk import QuranChunk


class QuranChunker:
    threshold_similarity: Annotated[float, Ge(0.0), Le(1.0)] = 0.8
    threshold_title: int = 10

    def chunking_strategy(self, sequence_words: list[str]):
        pass

    def clean_str(self, content: str) -> str:
        return content.strip()

    def isheader(self, text: str) -> bool:
        text = text.strip()

        if not text:
            return False

        # Markdown headers
        if re.match(r"^#+\s+", text):
            return True

        # Numbered sections (e.g. "1 Introduction", "2.3 Methods")
        if re.match(r"^\d+(\.\d+)*\s+", text):
            return True

        # Short titles in uppercase or Title Case
        if len(text.split()) <= self.threshold_title and not text.endswith("."):
            if text.isupper() or text.istitle():
                return True

        return False

    def _create_chunk(
        self,
        chunk_id: int,
        surah_n: int,
        ayah_n: int,
        ayah_ar: str,
        ayah_en: str,
        tafsir_text: str = "",
        header: str | None = None,
    ) -> QuranChunk:
        """
        Helper method to initialize a QuranChunk with shared metadata.
        """
        chunk = QuranChunk()
        chunk.chunk_id = chunk_id
        chunk.surah_n = surah_n
        chunk.ayah_n = ayah_n
        chunk.ayah_ar = ayah_ar
        chunk.ayah_en = ayah_en
        chunk.tafsir_chunk = tafsir_text
        chunk.header = header  # Optional field if your QuranChunk supports it
        return chunk

    def split_by_header_paragraphs(
        self,
        ayah_n: int,
        surah_n: int,
        ayah_ar: str,
        ayah_en: str,
        content: str,
    ) -> list[QuranChunk]:
        """
        Split tafsir text into chunks whenever a header is encountered.

        Each chunk contains:
        - Shared ayah metadata
        - Optional header title
        - Concatenated tafsir text
        """
        if not content:
            return []

        # Split into non-empty paragraphs
        paragraphs = [
            self.clean_str(p)
            for p in content.split("\n")
            if self.clean_str(p)
        ]

        if not paragraphs:
            return []

        chunks: list[QuranChunk] = []
        chunk_id = 0

        # Start with an empty chunk
        current_chunk = self._create_chunk(
            chunk_id=chunk_id,
            surah_n=surah_n,
            ayah_n=ayah_n,
            ayah_ar=ayah_ar,
            ayah_en=ayah_en,
        )

        for paragraph in paragraphs:
            # If paragraph is a header, close the current chunk first
            if self.isheader(paragraph):
                # Save current chunk only if it contains text
                if current_chunk.tafsir_chunk.strip():
                    chunks.append(current_chunk)
                    chunk_id += 1

                # Start a new chunk and store the header separately
                current_chunk = self._create_chunk(
                    chunk_id=chunk_id,
                    surah_n=surah_n,
                    ayah_n=ayah_n,
                    ayah_ar=ayah_ar,
                    ayah_en=ayah_en,
                    header=paragraph,
                )
                continue

            # Append paragraph to the current chunk
            if current_chunk.tafsir_chunk:
                current_chunk.tafsir_chunk += "\n\n"
            current_chunk.tafsir_chunk += paragraph

        # Append the final chunk if it contains text
        if current_chunk.tafsir_chunk.strip():
            chunks.append(current_chunk)

        return chunks