from rag.quran_chunk import QuranChunk
from dataclasses import dataclass, field

class QuranDoc:
    # --- Identification ---
    parent_id: str   # "1:2"

    # --- Quran metadata ---
    surah_n: int
    ayah_n: int
    surah_ar: str
    surah_eng: str

    # --- Core content ---
    ayah_ar: str
    ayah_en: str
    tafsir_full: str   # FULL tafsir (important)

    # --- Children ---
    tafsir_chunks: list[QuranChunk]