from __future__ import annotations
import hashlib, re, uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
import numpy as np

# ── Chunk schema ──────────────────────────────────────────────
@dataclass
class Chunk:
    text: str
    chunk_id: str
    doc_id: str
    parent_chunk_id: Optional[str] = None
    section_header: str = ""
    token_count: int = 0
    source_type: str = "general"
    embedding_model: str = ""
    indexed_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    file_hash: str = ""
    chunk_index: int = 0

# ── Main chunker ───────────────────────────────────────────────
class SemanticChunker:
    def __init__(
        self,
        embed_fn,                    # callable: str → np.array
        child_size: int = 256,      # tokens for retrieval chunks
        parent_size: int = 1024,    # tokens for generation chunks
        similarity_threshold: float = 0.75,  # split below this
        overlap: int = 32,          # token overlap between chunks
    ):
        self.embed_fn = embed_fn
        self.child_size = child_size
        self.parent_size = parent_size
        self.threshold = similarity_threshold
        self.overlap = overlap

    def chunk_document(
        self,
        text: str,
        doc_id: str,
        source_type: str = "general",
        file_hash: str = "",
        embedding_model: str = "text-embedding-3-large",
    ) -> tuple[list[Chunk], list[Chunk]]:
        """
        Returns (parent_chunks, child_chunks).
        Index child_chunks in the vector DB.
        Pass parent_chunks to the LLM after retrieval.
        """
        sentences = self._split_sentences(text)
        boundaries = self._find_semantic_boundaries(sentences)
        parent_chunks = self._build_parent_chunks(
            sentences, boundaries, doc_id,
            source_type, file_hash, embedding_model
        )
        child_chunks = self._build_child_chunks(parent_chunks)
        return parent_chunks, child_chunks

    def _split_sentences(self, text: str) -> list[str]:
        # Respect headings as hard boundaries
        text = re.sub(r'(#{1,6}\s[^\n]+)', r'\n\1\n', text)
        # Split on sentence-ending punctuation
        sentences = re.split(
            r'(?<=[.!?])\s+(?=[A-Z])|(?<=\n)\n+', text
        )
        return [s.strip() for s in sentences if s.strip()]

    def _find_semantic_boundaries(
        self, sentences: list[str]
    ) -> list[int]:
        """
        Embed sentences in one batched call.
        Split where cosine similarity between
        adjacent sentences drops below threshold.
        """
        if len(sentences) <= 1:
            return []
        embeddings = np.array([
            self.embed_fn(s) for s in sentences
        ])
        # Cosine similarity between adjacent sentences
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normed = embeddings / np.maximum(norms, 1e-8)
        similarities = np.sum(
            normed[:-1] * normed[1:], axis=1
        )
        boundaries = [
            i + 1
            for i, sim in enumerate(similarities)
            if sim < self.threshold
        ]
        return boundaries

    def _build_parent_chunks(
        self, sentences, boundaries, doc_id,
        source_type, file_hash, embedding_model
    ) -> list[Chunk]:
        segments = self._group_by_boundaries(sentences, boundaries)
        chunks = []
        current_header = ""
        for i, seg in enumerate(segments):
            # Detect and carry section headers forward
            header_match = re.match(r'^#{1,6}\s(.+)', seg[0])
            if header_match:
                current_header = header_match.group(1).strip()
            # Prepend header to chunk text for context
            body = " ".join(seg)
            text = (
                f"[{current_header}] {body}"
                if current_header else body
            )
            chunk_id = str(uuid.uuid4())
            chunks.append(Chunk(
                text=text,
                chunk_id=chunk_id,
                doc_id=doc_id,
                section_header=current_header,
                token_count=len(text.split()),
                source_type=source_type,
                embedding_model=embedding_model,
                file_hash=file_hash,
                chunk_index=i,
            ))
        return chunks

    def _build_child_chunks(
        self, parents: list[Chunk]
    ) -> list[Chunk]:
        """
        Split each parent into smaller child chunks.
        Children reference their parent via parent_chunk_id.
        Index children in the vector DB for precision retrieval.
        Return parent on hit for rich generation context.
        """
        children = []
        for parent in parents:
            words = parent.text.split()
            step = self.child_size - self.overlap
            for j, start in enumerate(
                range(0, len(words), step)
            ):
                child_words = words[start : start + self.child_size]
                if not child_words:
                    continue
                # Always prefix with section header
                prefix = (
                    f"[{parent.section_header}] "
                    if parent.section_header else ""
                )
                child_text = prefix + " ".join(child_words)
                children.append(Chunk(
                    text=child_text,
                    chunk_id=str(uuid.uuid4()),
                    doc_id=parent.doc_id,
                    parent_chunk_id=parent.chunk_id,
                    section_header=parent.section_header,
                    token_count=len(child_words),
                    source_type=parent.source_type,
                    embedding_model=parent.embedding_model,
                    file_hash=parent.file_hash,
                    chunk_index=j,
                ))
        return children

    def _group_by_boundaries(
        self, sentences, boundaries
    ) -> list[list[str]]:
        groups, current = [], []
        for i, s in enumerate(sentences):
            if i in boundaries and current:
                groups.append(current)
                current = []
            current.append(s)
        if current:
            groups.append(current)
        return groups