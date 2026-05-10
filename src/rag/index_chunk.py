from dataclasses import dataclass, field
import numpy as np
@dataclass
class IndexedChunk:
    chunk_id: str
    surah_n: int
    ayah_n: int
    text: str
    embedding: np.ndarray  # normalized dense embedding

