import sys
import pickle
import traceback
from rag.quran_doc import QuranDoc
from rag.quran_chunk import QuranChunk

def retrieval(docs_path, query, vector_db, top_k=5,return_parents_ids=False):
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
    print("=" * 80)
    print("STEP 2: Performing semantic search")
    print("=" * 80)
    print(f"Query: {query}")
    print(f"Top K: {top_k}")

    try:
        results = vector_db.search(query=query, top_k=top_k)
        print("Search completed successfully.")
    except Exception as e:
        print("ERROR during vector search:")
        print(str(e))
        traceback.print_exc()
        raise

    print()
    print("=" * 80)
    print("STEP 3: Inspecting raw search results")
    print("=" * 80)

    ids = results["ids"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    print(f"Number of retrieved chunks: {len(ids)}")
    print()

    for i, (chunk_id, metadata, distance) in enumerate(
        zip(ids, metadatas, distances), start=1
    ):
        print(f"Result {i}")
        print(f"  Chunk ID : {chunk_id}")
        print(f"  Surah    : {metadata['surah_n']}")
        print(f"  Ayah     : {metadata['ayah_n']}")
        print(f"  Distance : {distance}")
        print()

    print("=" * 80)
    print("STEP 4: Extracting parent IDs")
    print("=" * 80)
    list_parent_ids =[f'{m["surah_n"]}:{m["ayah_n"]}'
            for m in metadatas]
    parent_ids = list(
        set(list_parent_ids)
           
    )

    print(f"Unique parent IDs ({len(parent_ids)}):")
    for parent_id in parent_ids:
        print(f"  - {parent_id}")

    print()
    print("=" * 80)
    print("STEP 5: Building final context")
    print("=" * 80)

    content_parts = []

    for parent_id in parent_ids:
        print(f"Processing parent_id: {parent_id}")

        if parent_id not in docs:
            print("  -> WARNING: parent_id not found in doc store.")
            continue

        doc = docs[parent_id]

        print(f"  -> Found Surah {doc.surah_n}, Ayah {doc.ayah_n}")
        print(f"  -> Tafsir length: {len(doc.tafsir_full)} characters")

        formatted_doc = (
            f"Surah {doc.surah_n}, Ayah {doc.ayah_n}\n"
            f"Arabic: {doc.ayah_ar}\n"
            f"English: {doc.ayah_en}\n\n"
            f"Tafsir:\n{doc.tafsir_full}\n"
        )


        content_parts.append(formatted_doc)

    print()
    print("=" * 80)
    print("STEP 6: Final summary")
    print("=" * 80)
    print(f"Documents included in context: {len(content_parts)}")

    final_context = "\n\n---\n\n".join(content_parts)

    print(f"Final context length: {len(final_context)} characters")
    print("=" * 80)
    if return_parents_ids: return final_context,list_parent_ids

    return final_context
