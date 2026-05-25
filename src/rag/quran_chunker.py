from typing import Annotated
from annotated_types import Ge, Le
import re
from FlagEmbedding import BGEM3FlagModel
import numpy as np
from rag.quran_chunk import QuranChunk
from rag.quran_doc import QuranDoc
from rag.index_chunk import IndexedChunk


class QuranChunker:
    threshold_similarity: Annotated[float, Ge(0.0), Le(1.0)] = 0.8
    threshold_title: int = 10
    _embedding_model = BGEM3FlagModel("BAAI/bge-m3", use_bf16=True)

    def clean_str(self, content: str) -> str:
        return content.strip()

    def isheader(self, text: str) -> bool:
        text = text.strip()

        if not text:
            return False

        if re.match(r"^#+\s+", text):
            return True

        if re.match(r"^\d+(\.\d+)*\s+", text):
            return True

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
        chunk = QuranChunk(
                chunk_id=chunk_id,
                parent_id=f"{surah_n}:{ayah_n}",
                surah_n=surah_n,
                ayah_n=ayah_n,
                ayah_ar=ayah_ar,
                ayah_en=ayah_en,
                tafsir_chunk=tafsir_text,
                chunk_index=0,      # updated later
                content="",         # filled later
                header=header or "",
            )
        return chunk

    def split_by_header_paragraphs(
        self,
        ayah_n: int,
        surah_n: int,
        ayah_ar: str,
        ayah_en: str,
        content: str,
        max_tokens: int = 500,
        similarity_threshold: float = 0.85,
    ) -> QuranDoc:
        if not content:
            return []

        paragraphs = [
            self.clean_str(p)
            for p in content.split("\n")
            if self.clean_str(p)
        ]

        if not paragraphs:
            return []

        structured_chunks: list[QuranChunk] = []
        chunk_index = 0

        current_chunk = self._create_chunk(
            chunk_id=f"{surah_n}:{ayah_n}:{chunk_index}",
            surah_n=surah_n,
            ayah_n=ayah_n,
            ayah_ar=ayah_ar,
            ayah_en=ayah_en,
        )

        for paragraph in paragraphs:
            if self.isheader(paragraph):
                if current_chunk.tafsir_chunk.strip():
                    structured_chunks.append(current_chunk)
                    chunk_index += 1

                current_chunk = self._create_chunk(
                    chunk_id=f"{surah_n}:{ayah_n}:{chunk_index}",
                    surah_n=surah_n,
                    ayah_n=ayah_n,
                    ayah_ar=ayah_ar,
                    ayah_en=ayah_en,
                    header=self.clean_paragraph(paragraph),
                )
                continue

            cleaned = self.clean_paragraph(paragraph)

            if current_chunk.tafsir_chunk:
                current_chunk.tafsir_chunk += "\n\n"

            current_chunk.tafsir_chunk += cleaned

        if current_chunk.tafsir_chunk.strip():
            structured_chunks.append(current_chunk)

        tokenizer = self._embedding_model.tokenizer

        def count_tokens(text: str) -> int:
            return len(tokenizer.encode(text, add_special_tokens=False))

        final_chunks: list[QuranChunk] = []
        final_chunk_index = 0

        for structured_chunk in structured_chunks:
            tafsir_text = structured_chunk.tafsir_chunk
            n_tokens = count_tokens(tafsir_text)

            if n_tokens <= max_tokens:
                structured_chunk.chunk_index = final_chunk_index
                structured_chunk.chunk_id = f"{surah_n}:{ayah_n}:{final_chunk_index}"
                structured_chunk.content = (
                    f"Surah {surah_n}, Ayah {ayah_n}\n\n"
                    f"Arabic:\n{ayah_ar}\n\n"
                    f"English:\n{ayah_en}\n\n"
                    f"Tafsir:\n{tafsir_text}"
                )
                final_chunks.append(structured_chunk)
                final_chunk_index += 1
                continue

            sub_chunks = self.semantic_chunks(
                paragraph=tafsir_text,
                max_tokens=max_tokens,
                similarity_threshold=similarity_threshold,
            )

            for sub_text in sub_chunks:
                new_chunk = self._create_chunk(
                    chunk_id=f"{surah_n}:{ayah_n}:{final_chunk_index}",
                    surah_n=surah_n,
                    ayah_n=ayah_n,
                    ayah_ar=ayah_ar,
                    ayah_en=ayah_en,
                    tafsir_text=sub_text,
                    header=getattr(structured_chunk, "header", None),
                )
                new_chunk.chunk_index = final_chunk_index
                new_chunk.content = (
                    f"Surah {surah_n}, Ayah {ayah_n}\n\n"
                    f"Arabic:\n{ayah_ar}\n\n"
                    f"English:\n{ayah_en}\n\n"
                    f"Tafsir:\n{sub_text}"
                )
                final_chunks.append(new_chunk)
                final_chunk_index += 1

        chunk_doc = QuranDoc()
        chunk_doc.ayah_ar = ayah_ar
        chunk_doc.ayah_en = ayah_en
        chunk_doc.ayah_n = ayah_n
        chunk_doc.surah_n = surah_n
        chunk_doc.parent_id = str(surah_n) + ":" + str(ayah_n)
        chunk_doc.tafsir_full = content
        chunk_doc.tafsir_chunks = final_chunks

        return chunk_doc

    def semantic_chunks(
        self,
        paragraph: str,
        max_tokens: int = 500,
        similarity_threshold: float = 0.85,
    ) -> list[str]:
        sentences = [s.strip() for s in paragraph.split(".") if s.strip()]
        if not sentences:
            return []

        embeddings = []
        for sentence in sentences:
            result = self._embedding_model.encode(
                sentence,
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False,
            )
            embeddings.append(result["dense_vecs"])

        similarities = []
        for i in range(len(embeddings) - 1):
            v1 = np.asarray(embeddings[i]).flatten()
            v2 = np.asarray(embeddings[i + 1]).flatten()
            sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            similarities.append(sim)

        tokenizer = self._embedding_model.tokenizer

        def count_tokens(text: str) -> int:
            return len(tokenizer.encode(text, add_special_tokens=False))

        chunks = []
        current_chunk = sentences[0]
        current_tokens = count_tokens(current_chunk)

        for i in range(1, len(sentences)):
            sentence = sentences[i]
            sentence_tokens = count_tokens(sentence)
            sim = similarities[i - 1]

            if current_tokens + sentence_tokens > max_tokens or sim < similarity_threshold:
                chunks.append(current_chunk)
                current_chunk = sentence
                current_tokens = sentence_tokens
            else:
                current_chunk += ". " + sentence
                current_tokens += sentence_tokens

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def clean_paragraph(self, txt: str) -> str:
        txt = txt.replace("\\n", " ")
        txt = txt.replace("\n", " ")
        txt = txt.replace("\\-", " ")
        txt = txt.replace("\\`", " ")
        txt = re.sub(r"\s+", " ", txt)
        txt = re.sub(r"\.\s*([A-Z])", " ", txt)
        return txt.replace("\n\n", " ").strip()

    def indexing_chunk(self, chunk: QuranChunk) -> IndexedChunk:
        text = chunk.content if getattr(chunk, "content", "") else chunk.tafsir_chunk

        result = self._embedding_model.encode(
            text,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )

        embedding = np.asarray(result["dense_vecs"], dtype=np.float32).flatten()
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        indexed_chunk = IndexedChunk(
        chunk_id = chunk.chunk_id,
        surah_n = chunk.surah_n,
        ayah_n = chunk.ayah_n,
        text = text,
        parent_id=chunk.parent_id,
        embedding = embedding
        )
        return indexed_chunk