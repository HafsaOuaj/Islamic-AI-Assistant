class QuranChunk:

    # --- Identification ---
    chunk_id: str          # "1:2:0"
    parent_id: str         # "1:2"

    # --- Quran metadata ---
    surah_n: int
    ayah_n: int

    # --- Core content ---
    ayah_ar: str
    ayah_en: str
    tafsir_chunk: str

    # --- Structured info ---
    chunk_index: int       # order inside ayah

    # --- Embedding content ---
    content: str           # formatted text used for embedding
