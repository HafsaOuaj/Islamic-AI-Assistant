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
    reranker=None,
):
    """
    Multi-stage retrieval pipeline

    Steps:
    1. Dense retrieval
    2. Cross-encoder reranking
    3. Diversity filtering
    4. Final context construction
    """

    
    # STEP 1 — LOAD DOCUMENT STORE
    
    print("=" * 80)
    print("STEP 1: Loading QuranDoc store")
    print("=" * 80)
    print(f"Docs path: {docs_path}")

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

    
    # STEP 2 — DENSE RETRIEVAL
    
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

    
    # EXTRACT SEARCH OUTPUTS
    
    ids = results["ids"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    
    # STEP 3 — BUILD RETRIEVED CHUNK OBJECTS
    
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
            "original_rank": rank,
            "chunk_id": chunk_id,
            "parent_id": parent_id,
            "text": doc_text,
            "metadata": metadata,
            "distance": distance,
            "rerank_score": None,
        }

        retrieved_chunks.append(chunk_data)

        print(f"[{rank}]")
        print(f"Chunk ID      : {chunk_id}")
        print(f"Parent ID     : {parent_id}")
        print(f"Distance      : {distance:.4f}")
        print()

    print()

    # STEP 4 — RERANKING
    
    print("=" * 80)
    print("STEP 4: Cross-encoder reranking")
    print("=" * 80)

    if reranker is not None:

        print("Reranker detected.")
        print("Computing rerank scores...")

        reranked_chunks = []

        for chunk in retrieved_chunks:

            try:
                score = reranker.compute_score(
                    [[query, chunk["text"]]],
                    normalize=True,
                )[0]

            except Exception as e:
                print("ERROR during reranking:")
                print(str(e))
                traceback.print_exc()
                raise

            chunk["rerank_score"] = score

            reranked_chunks.append(chunk)

            print(
                f"Chunk {chunk['chunk_id']} | "
                f"Original Rank: {chunk['original_rank']} | "
                f"Rerank Score: {score:.4f}"
            )

        
        # Sort by rerank score DESCENDING
        
        reranked_chunks = sorted(
            reranked_chunks,
            key=lambda x: x["rerank_score"],
            reverse=True,
        )

        print()
        print("Reranking completed.")
        print()

        print("NEW ORDER AFTER RERANKING:")
        print("-" * 80)

        for new_rank, chunk in enumerate(reranked_chunks, start=1):

            print(
                f"[NEW RANK {new_rank}] "
                f"{chunk['chunk_id']} | "
                f"Original Rank={chunk['original_rank']} | "
                f"Score={chunk['rerank_score']:.4f}"
            )

    else:

        print("No reranker provided.")
        print("Using dense retrieval ordering.")

        reranked_chunks = retrieved_chunks

    print()

    
    # STEP 5 — DIVERSITY FILTERING
    
    print("=" * 80)
    print("STEP 5: Diversity filtering")
    print("=" * 80)

    parent_counter = defaultdict(int)

    diversified_chunks = []

    for chunk in reranked_chunks:

        parent_id = chunk["parent_id"]

        # Keep only N chunks per parent
        if parent_counter[parent_id] >= max_chunks_per_parent:
            continue

        diversified_chunks.append(chunk)

        parent_counter[parent_id] += 1

    print(
        f"Chunks after diversity filtering: "
        f"{len(diversified_chunks)}"
    )

    print()

    for chunk in diversified_chunks:

        print(
            f"Kept chunk {chunk['chunk_id']} | "
            f"Parent={chunk['parent_id']}"
        )

    print()

    
    # STEP 6 — FINAL TOP-K SELECTION
    
    print("=" * 80)
    print("STEP 6: Selecting final chunks")
    print("=" * 80)

    final_chunks = diversified_chunks[:final_k]

    print(f"Final chunk count: {len(final_chunks)}")
    print()

    for i, chunk in enumerate(final_chunks, start=1):

        print(f"[FINAL {i}]")
        print(f"Chunk ID      : {chunk['chunk_id']}")
        print(f"Parent ID     : {chunk['parent_id']}")
        print(f"Original Rank : {chunk['original_rank']}")

        if chunk["rerank_score"] is not None:
            print(f"Rerank Score  : {chunk['rerank_score']:.4f}")

        print()

    
    # STEP 7 — BUILD FINAL CONTEXT
    
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

    print()

    
    # FINAL SUMMARY
    
    print("=" * 80)
    print("RETRIEVAL COMPLETED")
    print("=" * 80)

    print(f"Query                 : {query}")
    print(f"Initial retrieved     : {len(retrieved_chunks)}")
    print(f"After reranking       : {len(reranked_chunks)}")
    print(f"After diversity       : {len(diversified_chunks)}")
    print(f"Final selected chunks : {len(final_chunks)}")

    print("=" * 80)

    
    # RETURN
    
    if return_parent_ids:
        return final_context, parent_ids

    return final_context