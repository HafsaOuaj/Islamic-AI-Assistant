import pandas as pd
from quran_chunker import QuranChunker
from vector_database import VectorDatabase
import pandas as pd
import pickle
import traceback
import os
def build_index(dataset_path,output_path):
    print("=" * 80)
    print("STEP 1: Loading dataset")
    print("=" * 80)

    # Load only one row for debugging
    data = pd.read_json(dataset_path, lines=True, nrows=1)
    print(f"Dataset loaded successfully.")
    print(f"Number of rows: {len(data)}")
    print(f"Columns: {list(data.columns)}")
    print()

    print("=" * 80)
    print("STEP 2: Initializing Vector Database")
    print("=" * 80)

    vector_db = VectorDatabase(
        collection_name="quran_vdb",
        path=output_path,
    )
    print("Vector database initialized.")
    print()

    print("=" * 80)
    print("STEP 3: Initializing QuranChunker")
    print("=" * 80)

    chunker = QuranChunker()
    print("Chunker initialized.")
    print()

    all_indexed_chunks = []
    doc_store = {}

    print("=" * 80)
    print("STEP 4: Processing dataset rows")
    print("=" * 80)

    for idx, row in data.iterrows():
        print(f"\nProcessing row {idx + 1}/{len(data)}")
        print(f"Surah: {row['surah_n']}, Ayah: {row['ayah_n']}")
        print(f"Parent ID: {row['surah_n']}:{row['ayah_n']}")

        try:
            print("  -> Chunking tafsir...")
            qdoc = chunker.split_by_header_paragraphs(
                ayah_n=row["ayah_n"],
                surah_n=row["surah_n"],
                ayah_ar=row["ayah_text_ar"],
                ayah_en=row["ayah_text"],
                content=row["ayah_tafsir"],
                max_tokens=500,
                similarity_threshold=0.85,
            )

            print("  -> Chunking completed.")

            if not qdoc:
                print("  -> WARNING: qdoc is None. Skipping.")
                continue

            if not qdoc.tafsir_chunks:
                print("  -> WARNING: No chunks generated. Skipping.")
                continue

            print(f"  -> Generated {len(qdoc.tafsir_chunks)} chunks.")

            # Save full QuranDoc
            doc_store[qdoc.parent_id] = qdoc
            print(f"  -> Stored QuranDoc with key: {qdoc.parent_id}")

            # Embed each chunk
            for chunk_idx, chunk in enumerate(qdoc.tafsir_chunks):
                print(
                    f"    -> Embedding chunk {chunk_idx + 1}/{len(qdoc.tafsir_chunks)} "
                    f"(ID: {chunk.chunk_id})"
                )

                indexed_chunk = chunker.indexing_chunk(chunk)
                all_indexed_chunks.append(indexed_chunk)

                print(
                    f"       Embedding shape: {indexed_chunk.embedding.shape}"
                )

        except Exception as e:
            print(f"  -> ERROR while processing row {idx}:")
            print(str(e))
            traceback.print_exc()
            raise

    print()
    print("=" * 80)
    print("STEP 5: Adding chunks to Vector Database")
    print("=" * 80)

    print(f"Total indexed chunks to add: {len(all_indexed_chunks)}")

    if all_indexed_chunks:
        try:
            vector_db.add_chunks(all_indexed_chunks)
            print("Chunks successfully added to vector database.")
        except Exception as e:
            print("ERROR while adding chunks to vector database:")
            print(str(e))
            traceback.print_exc()
            raise
    else:
        print("No chunks to add.")

    print()
    print("=" * 80)
    print("STEP 6: Checking database statistics")
    print("=" * 80)

    try:
        db_count = vector_db.count()
        print(f"Indexed chunks: {len(all_indexed_chunks)}")
        print(f"Vector database count: {db_count}")
    except Exception as e:
        print("ERROR while counting database entries:")
        print(str(e))
        traceback.print_exc()
        raise

    print()
    print("=" * 80)
    print("STEP 7: Saving QuranDoc store")
    print("=" * 80)

    output_path = os.path.join(output_path,"quran_docs.pkl")
    print(f"Saving {len(doc_store)} QuranDoc objects to: {output_path}")

    try:
        with open(output_path, "wb") as f:
            pickle.dump(doc_store, f)
        print("Pickle file saved successfully.")
    except Exception as e:
        print("ERROR while saving pickle file:")
        print(str(e))
        traceback.print_exc()
        raise

    print()
    print("=" * 80)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"Total QuranDoc objects saved: {len(doc_store)}")
    print(f"Total chunks indexed: {len(all_indexed_chunks)}")
    print(f"Final vector database count: {vector_db.count()}")


if __name__ == "__main__":
    dataset_path = "../../data/silver/tafsir_dataset.json"
    output_path= "../../data/gold"
    build_index(dataset_path=dataset_path,output_path=output_path)