from enum import Enum
from dataclasses import asdict, dataclass
from typing import List, Dict, Any, Optional


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"

# Define chunking types for document chunking
class ChunkType(Enum):
    DOCUMENT = "document"
    SECTION = "section"
    SUBSECTION = "subsection" 
    PARAGRAPH = "paragraph"
    TABLE = "table"
    RECURSIVE_SPLIT = "recursive_split"

# Define structural chunk representation
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
