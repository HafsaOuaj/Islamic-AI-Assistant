import pickle
import traceback
from collections import defaultdict

from rag.quran_doc import QuranDoc
from rag.quran_chunk import QuranChunk


def retrieval(
    docs_path,
    query,
    vector_db,
    top_k=15,
    final_k=3,
    max_chunks_per_parent=1,
    return_parent_ids=False,
):
    """
    Retrieval pipeline:
    1. Dense retrieval
    2. Diversity filtering
    3. (Future) reranking
    4. Final context construction
    """

    print("=" * 80)
    print("STEP 1: Loading QuranDoc store")
    print("=" * 80)
    print(f"Docs path: {docs_path}")

    # ------------------------------------------------------------
    # Load QuranDoc store
    # ------------------------------------------------------------
    try:
        with open(docs_path, "rb") as f:
            docs = pickle.load(f)

        print(f"Successfully loaded {len(docs)} QuranDoc objects.")

    except Exception as e:
        print("ERROR while loading pickle file:")
        print(str(e))
        traceback.print_exc()
        raise

    print()

    # ------------------------------------------------------------
    # Dense retrieval
    # ------------------------------------------------------------
    print("=" * 80)
    print("STEP 2: Dense semantic retrieval")
    print("=" * 80)

    print(f"Query: {query}")
    print(f"Top-K retrieval: {top_k}")

    try:
        results = vector_db.search(
            query=query,
            top_k=top_k,
        )

        print("Vector search completed successfully.")

    except Exception as e:
        print("ERROR during vector search:")
        print(str(e))
        traceback.print_exc()
        raise

    print()

    # ------------------------------------------------------------
    # Extract retrieval outputs
    # ------------------------------------------------------------
    ids = results["ids"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    # ------------------------------------------------------------
    # Build retrieved chunk objects
    # ------------------------------------------------------------
    print("=" * 80)
    print("STEP 3: Building retrieved chunk candidates")
    print("=" * 80)

    retrieved_chunks = []

    for rank, (chunk_id, doc_text, metadata, distance) in enumerate(
        zip(ids, documents, metadatas, distances),
        start=1,
    ):

        parent_id = f"{metadata['surah_n']}:{metadata['ayah_n']}"

        chunk_data = {
            "rank": rank,
            "chunk_id": chunk_id,
            "parent_id": parent_id,
            "text": doc_text,
            "metadata": metadata,
            "distance": distance,
        }

        retrieved_chunks.append(chunk_data)

        print(f"[{rank}]")
        print(f"Chunk ID : {chunk_id}")
        print(f"Parent ID: {parent_id}")
        print(f"Distance : {distance}")
        print()

    # ------------------------------------------------------------
    # Diversity filtering
    # ------------------------------------------------------------
    print("=" * 80)
    print("STEP 4: Diversity filtering")
    print("=" * 80)

    parent_counter = defaultdict(int)

    diversified_chunks = []

    for chunk in retrieved_chunks:

        parent_id = chunk["parent_id"]

        if parent_counter[parent_id] >= max_chunks_per_parent:
            continue

        diversified_chunks.append(chunk)

        parent_counter[parent_id] += 1

    print(f"Chunks after diversity filtering: {len(diversified_chunks)}")

    for chunk in diversified_chunks:
        print(
            f"Kept chunk {chunk['chunk_id']} "
            f"(parent={chunk['parent_id']})"
        )

    print()

    # ------------------------------------------------------------
    # FUTURE RERANKING PLACEHOLDER
    # ------------------------------------------------------------
    print("=" * 80)
    print("STEP 5: Reranking")
    print("=" * 80)

    print("Currently using dense retrieval ordering.")
    print("Future reranker will reorder diversified chunks here.")

    reranked_chunks = diversified_chunks

    print()

    # ------------------------------------------------------------
    # Keep final top chunks
    # ------------------------------------------------------------
    print("=" * 80)
    print("STEP 6: Selecting final chunks")
    print("=" * 80)

    final_chunks = reranked_chunks[:final_k]

    print(f"Final chunk count: {len(final_chunks)}")

    for i, chunk in enumerate(final_chunks, start=1):
        print(f"[FINAL {i}]")
        print(f"Chunk ID : {chunk['chunk_id']}")
        print(f"Parent ID: {chunk['parent_id']}")
        print()

    # ------------------------------------------------------------
    # Build final context
    # ------------------------------------------------------------
    print("=" * 80)
    print("STEP 7: Building final context")
    print("=" * 80)

    context_parts = []

    parent_ids = []

    for chunk in final_chunks:

        parent_ids.append(chunk["parent_id"])

        context_parts.append(chunk["text"])

    final_context = "\n\n---\n\n".join(context_parts)

    print(f"Final context length: {len(final_context)} characters")

    print("=" * 80)
    print("RETRIEVAL COMPLETED")
    print("=" * 80)

    # ------------------------------------------------------------
    # Return outputs
    # ------------------------------------------------------------
    if return_parent_ids:
        return final_context, parent_ids

    return final_context