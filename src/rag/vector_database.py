import numpy as np
import chromadb
from FlagEmbedding import BGEM3FlagModel


class VectorDatabase:
    def __init__(
        self,
        collection_name: str,
        path: str = "../../vector_database",
        model=None
    ):
        """
        Vector DB wrapper for Quran RAG system
        """

        # -----------------------------
        # 1. Embedding model
        # -----------------------------
        if model is None:
            model = BGEM3FlagModel("BAAI/bge-m3", use_bf16=True)

        self.model = model

        # -----------------------------
        # 2. Chroma client
        # -----------------------------
        self.client = chromadb.PersistentClient(path=path)

        # -----------------------------
        # 3. Collection init
        # -----------------------------
        self.collection = self.client.get_or_create_collection(
            name=collection_name
        )

    # ------------------------------------------------------------
    # SEARCH (semantic retrieval)
    # ------------------------------------------------------------
    def search(self, query: str, top_k: int = 5):
        """
        Perform semantic search over stored Quran chunks
        """

        # Encode query into dense embedding
        q_emb = self.model.encode(
            query,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False
        )["dense_vecs"]

        # Convert to numpy and normalize (cosine similarity)
        q_emb = np.asarray(q_emb, dtype=np.float32)
        q_emb = q_emb / np.linalg.norm(q_emb)

        # Query vector database
        results = self.collection.query(
            query_embeddings=[q_emb.tolist()],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        return results

    # ------------------------------------------------------------
    # INSERT (index chunks into vector DB)
    # ------------------------------------------------------------
    def add_chunks(self, indexed_chunks):
        """
        Add pre-embedded chunks into Chroma vector database
        """

        ids = []
        documents = []
        embeddings = []
        metadatas = []

        for c in indexed_chunks:

            # Unique chunk ID
            ids.append(c.chunk_id)

            # Text stored for retrieval display
            documents.append(c.text)

            # Normalize embeddings before storage
            emb = np.asarray(c.embedding, dtype=np.float32)
            emb = emb / np.linalg.norm(emb)

            embeddings.append(emb.tolist())

            # Store metadata for filtering/debugging
            metadatas.append({
                "surah_n": c.surah_n,
                "ayah_n": c.ayah_n
            })

        # Insert into Chroma collection
        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )

    # ------------------------------------------------------------
    # DEBUG UTIL: count stored vectors
    # ------------------------------------------------------------
    def count(self):
        """
        Return number of stored chunks in vector DB
        """
        return self.collection.count()