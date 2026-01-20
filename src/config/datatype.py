from enum import Enum

class RoleType(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class ChunkType(Enum):
    DOCUMENT = "document"
    SECTION = "section"
    SUBSECTION = "subsection" 
    PARAGRAPH = "paragraph"
    TABLE = "table"
    RECURSIVE_SPLIT = "recursive_split"