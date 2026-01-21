from dataclasses import asdict, dataclass
from typing import List, Dict, Any, Optional

from src.config.datatype import ChunkType

# ===== Structural Chunk Classes =====

@dataclass
class StructuralChunk:
    content: str
    chunk_type: ChunkType
    level: int
    section_hierarchy: List[str]
    metadata: Dict[str, Any]
    token_count: int
    is_oversized: bool = False
    parent_chunk_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["chunk_type"] = self.chunk_type.value
        return data


# ===== Vector Store Data Models =====

@dataclass
class VectorPoint:
    """Represents a vector point for storage in Qdrant"""
    id: str
    vector: List[float]  # Dense vector
    payload: Dict[str, Any]
    sparse_indices: Optional[List[int]] = None  # Sparse vector indices
    sparse_values: Optional[List[float]] = None  # Sparse vector values


@dataclass
class SparseVector:
    """Represents a sparse vector with indices and values"""
    indices: List[int]
    values: List[float]


@dataclass
class SearchResult:
    """Represents a search result from vector store"""
    id: str
    score: float
    payload: Dict[str, Any]
