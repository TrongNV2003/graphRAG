import re
from loguru import logger
from transformers import AutoTokenizer
from typing import List, Dict, Optional, Tuple
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config.setting import llm_config
from src.config.datatype import ChunkType
from src.config.dataclass import StructuralChunk


class TwoPhaseDocumentChunker:
    """
    Chunking text using Rule-based methods from a long context.
    Phase 1: Document-Structured Chunking (e.g.: "Chapter 1", "Chương 1", "I." v.v.).
    Phase 2: Fix-sized Chunking: Chunking with fixed sizes (e.g.: 2048 tokens).

    Args:
        input (str): Whole documents text.

    Returns:
        output (list): List of chunking summerized context.
    """
    
    def __init__(
        self, 
        chunk_size: int = 2048,
        chunk_overlap: int = 0,
        tokenize_model: Optional[str] = None,
        verbose: bool = False,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.verbose = verbose
        
        model_name = tokenize_model or llm_config.llm_model
        
        if model_name.lower().startswith(("gpt-", "o1-", "openai/")):
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
            logger.info(f"Using gpt2 tokenizer as proxy for OpenAI model: {model_name}")
        else:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            except Exception as e:
                logger.warning(f"Failed to load tokenizer for {model_name}: {e}. Falling back to gpt2.")
                self.tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)

        # Markdown heading patterns
        self.heading_patterns = {
            1: r'^#\s+(.+)$',
            2: r'^##\s+(.+)$', 
            3: r'^###\s+(.+)$',
            4: r'^####\s+(.+)$',
            5: r'^#####\s+(.+)$',
            6: r'^######\s+(.+)$',
            7: r'^#######\s+(.+)$',
            8: r'^########\s+(.+)$',
            9: r'^#########\s+(.+)$'
        }
        
        # Rule-based heading patterns
        self.heading_patterns_rulebased = {
            'phan': [
                r'^\s*[Pp][Hh][Ầầ][Nn]\s+([IVX]+)\.\s+(.+)$',   # PHẦN I. Title, Phần II. Title
                r'^\s*[Pp][Hh][Ầầ][Nn]\s+([IVX]+)\)\s+(.+)$',   # PHẦN I) Title, Phần II) Title
            ],
            'chuong': [
                r'^\s*[Cc][Hh][Ưư][Ơơ][Nn][Gg]\s+([IVX]+)\.\s+(.+)$',  # CHƯƠNG I. Title, Chương II. Title
                r'^\s*[Cc][Hh][Ưư][Ơơ][Nn][Gg]\s+([IVX]+)\)\s+(.+)$',  # CHƯƠNG I) Title, Chương II) Title
            ],
            'uppercase_alphabetical': [
                r'^\s*([A-HJ-UW-Z])\.\s+(.+)$',     # A. B. C. ... (exclude I,V,X)
                r'^\s*([A-HJ-UW-Z])\)\s+(.+)$',     # A) B) C) ... (exclude I,V,X)
            ],
            'roman': [
                r'^\s*([IVX]+)\.\s+(.+)$',           # I. Title, II. Title (fallback, if no "CHƯƠNG")
                r'^\s*([IVX]+)\)\s+(.+)$',           # I) Title, II) Title
            ],
            'muc': [
                r'^\s*[Mm][ỤỤụu][Cc]\s+(\d+)\.\s+(.+)$',     # MỤC 1. Title, MỤC 2. Title
                r'^\s*[Mm][ỤỤụu][Cc]\s+(\d+)\)\s+(.+)$',     # MỤC 1) Title, MỤC 2) Title
            ],
            # Move sub_decimal and decimal BEFORE arabic to prioritize specific matches
            'sub_decimal': [
                r'^\s*(\d+\.\d+\.\d+)\.\s+(.+)$',  # 1.1.1. Title
                r'^\s*(\d+\.\d+\.\d+)\)\s+(.+)$',  # 1.1.1) Title
            ],
            'decimal': [
                r'^\s*(\d+\.\d+)\.\s+(.+)$',  # 1.1. Title, 1.2. Title
                r'^\s*(\d+\.\d+)\)\s+(.+)$',  # 1.1) Title, 1.2) Title
            ],
            'arabic': [
                r'^\s*(\d+)\.\s+(.+)$',           # 1. Title, 2. Title (fallback, if no "MỤC")
                r'^\s*(\d+)\)\s+(.+)$',           # 1) Title, 2) Title
            ],
            'alphabetical': [
                r'^\s*([a-z])\.\s+(.+)$',     # a. Title, b. Title
                r'^\s*([a-z])\)\s+(.+)$',     # a) Title, b) Title
            ]
        }
        
        # Appendix and footnotes patterns
        self.special_section_patterns = {
            'appendix': [
                r'^\s*[Pp][Hh][Ụụ]\s+[Ll][Ụụ][Cc]\s*$',           # "Phụ lục"
                r'^\s*[Pp][Hh][Ụụ]\s+[Ll][Ụụ][Cc]\s*\d*\s*$',      # "Phụ lục 1"
                r'^\s*[Pp][Hh][Ụụ]\s+[Ll][Ụụ][Cc]\s+([IVX]+)\s*$', # "Phụ lục I"
                r'^\s*[Aa][Pp][Pp][Ee][Nn][Dd][Ii][Xx]\s*$',          # "APPENDIX"
                r'^\s*[Aa][Pp][Pp][Ee][Nn][Dd][Ii][Xx]\s+[A-Z0-9]+\s*$',  # "APPENDIX A", "APPENDIX 1"
            ],
            'footnote': [
                r'^\s*[Cc][Hh][Úú]\s+[Tt][Hh][Íí][Cc][Hh]\s*$',   # "Chú thích"
                r'^\s*[Nn][Oo][Tt][Ee][Ss]?\s*$',                      # "NOTES", "NOTE"
            ],
            'reference': [
                r'^\s*[Tt][Àà][Ii]\s+[Ll][Ii][Ệệ][Uu]\s+[Tt][Hh][Aa][Mm]\s+[Kk][Hh][Ảả][Oo]\s*$',  # "Tài liệu tham khảo"
                r'^\s*[Rr][Ee][Ff][Ee][Rr][Ee][Nn][Cc][Ee][Ss]?\s*$',  # "REFERENCES", "REFERENCE"
            ]
        }
        
        self.structure_markers = {
            'table_row': r'^\s*\|.*\|\s*$',
        }
        
        # Define order of pattern type
        self.pattern_hierarchy = ['phan', 'chuong', 'uppercase_alphabetical', 'roman', 'muc', 'arabic', 'decimal', 'sub_decimal', 'alphabetical']

    def chunk_document(self, document: str, max_new_chunk_size: Optional[int] = None) -> List[StructuralChunk]:
        logger.info("Phase 1: Structure-based chunking.")
        structural_chunks = self._structural_chunking(document, chunk_size=max_new_chunk_size)

        logger.info("Phase 2: Recursive chunking for oversized chunks.")
        final_chunks = self._recursive_chunking(structural_chunks)
        
        return final_chunks

    def export_to_schema(
        self,
        chunks: List[StructuralChunk], 
        show_structure: Optional[bool] = False,
    ) -> Dict:
        analysis_data = {
            "chunks": [],
            "total_chunks": len(chunks),
        }

        for i, chunk in enumerate(chunks):
            chunk_data = {
                "chunk_id": i,
                "level": chunk.level,
                "chunk_type": chunk.chunk_type.value,
                "section_hierarchy": chunk.section_hierarchy,
                "is_oversized": chunk.is_oversized,
                "parent_chunk_id": chunk.parent_chunk_id,
                "content": chunk.content,
            }
            analysis_data["chunks"].append(chunk_data)
            
        if show_structure:
            analysis_data["document_structure"] = self._analyze_document_structure(chunks)
        
        return analysis_data

    def _analyze_document_structure(self, chunks: List[StructuralChunk]) -> Dict:
        """Analyze the overall document structure"""
        structure_tree = {}
        section_depths = []
        
        for chunk in chunks:
            if chunk.section_hierarchy:
                current_level = structure_tree
                for section in chunk.section_hierarchy:
                    if section not in current_level:
                        current_level[section] = {
                            "_chunks": 0,
                            "_subsections": {}
                        }
                    current_level[section]["_chunks"] += 1
                    current_level = current_level[section]["_subsections"]
                
                section_depths.append(len(chunk.section_hierarchy))
        
        return {
            "structure_tree": structure_tree,
            "max_depth": max(section_depths) if section_depths else 0,
        }

    def _structural_chunking(self, document: str, chunk_size: Optional[int] = None) -> List[StructuralChunk]:
        lines = document.split('\n')
        structural_elements = self._parse_document_structure(lines)
        structural_elements = self._normalize_heading_levels(structural_elements)
        chunks = self._create_structural_chunks(structural_elements)
        
        if chunk_size is None:
            chunk_size = self.chunk_size
        
        for chunk in chunks:
            if not chunk.content:
                continue
            
            chunk.token_count = len(self.tokenizer.tokenize(chunk.content))
            chunk.is_oversized = chunk.token_count > chunk_size
        
        if self.verbose:
            logger.info(f"Created {len(chunks)} structural chunks")
            oversized_count = sum(1 for c in chunks if c.is_oversized)
            logger.warning(f"{oversized_count} chunks exceed {chunk_size} tokens")

        return chunks

    def _parse_document_structure(self, lines: List[str]) -> List[Dict]:
        elements = []
        current_section_stack = []
        special_section_active = None
        i = 0
        
        while i < len(lines):
            line = lines[i].rstrip()
            
            if not line.strip():
                elements.append({
                    'type': 'empty_line',
                    'content': line,
                    'line_index': i,
                    'section_stack': current_section_stack.copy(),
                    'special_section': special_section_active
                })
                i += 1
                continue
            
            # Check for special sections (Phụ lục, Chú thích)
            special_section_type = self._detect_special_section(line)
            
            if special_section_type:
                special_section_active = special_section_type
                
                current_section_stack = []
                elements.append({
                    'type': 'special_section_header',
                    'special_section_type': special_section_type,
                    'content': line,
                    'line_index': i,
                    'section_stack': []
                })
                i += 1
                continue
            
            pattern_type = self._detect_heading_level(line)
            
            if pattern_type:
                heading_text = self._extract_heading_text(line, pattern_type)
                elements.append({
                    'type': 'heading',
                    'pattern_type': pattern_type,
                    'content': line,
                    'heading_text': heading_text,
                    'line_index': i,
                    'section_stack': current_section_stack.copy(),
                    'special_section': special_section_active
                })
                i += 1
                
            # Check for tables
            elif re.match(self.structure_markers['table_row'], line):
                table_content, end_index = self._extract_table(lines, i)
                elements.append({
                    'type': 'table',
                    'content': table_content,
                    'line_index': i,
                    'end_index': end_index,
                    'section_stack': [],
                    'special_section': special_section_active
                })
                i = end_index + 1
                
            # Check for paragraphs
            else:
                paragraph, end_index = self._extract_paragraph(lines, i)
                elements.append({
                    'type': 'paragraph',
                    'content': paragraph,
                    'line_index': i,
                    'end_index': end_index,
                    'section_stack': [],
                    'special_section': special_section_active
                })
                i = end_index + 1
        
        return elements
    
    def _normalize_heading_levels(self, elements: List[Dict]) -> List[Dict]:
        """
        Normalize heading levels to ensure proper hierarchy based on document structure.
        This method dynamically assigns levels based on the actual pattern types found in the document.
        """
        heading_elements = [e for e in elements if e['type'] == 'heading']
        
        if not heading_elements:
            return elements
        
        pattern_types_found = []
        seen_patterns = set()
        for element in heading_elements:
            pattern_type = element['pattern_type']
            if pattern_type not in seen_patterns:
                pattern_types_found.append(pattern_type)
                seen_patterns.add(pattern_type)
        
        level_mapping = self._create_dynamic_level_mapping(pattern_types_found)
        
        print(f"Dynamic level mapping based on document structure:") if self.verbose else None
        for pattern_type, level in level_mapping.items():
            print(f"   {pattern_type} → level {level}") if self.verbose else None
        
        normalized_elements = []
        current_section_stack = []
        
        active_special_section = None
        
        for element in elements:
            if element.get('special_section') != active_special_section:
                current_section_stack = []
                active_special_section = element.get('special_section')
                
            if element['type'] == 'heading':
                pattern_type = element['pattern_type']
                new_level = level_mapping.get(pattern_type)
                
                if new_level is None:
                    normalized_elements.append(element)
                    continue
                
                element_copy = element.copy()
                element_copy['level'] = new_level
                
                # Rebuild section stack with normalized level
                heading_text = element_copy['heading_text']
                current_section_stack = current_section_stack[:new_level-1]
                current_section_stack.append(heading_text)
                element_copy['section_stack'] = current_section_stack.copy()
                
                normalized_elements.append(element_copy)

                print(f"   '{heading_text}' ({pattern_type}) → level {new_level}") if self.verbose else None
            else:
                # Update section stack for non-heading elements
                element_copy = element.copy()
                element_copy['section_stack'] = current_section_stack.copy()
                normalized_elements.append(element_copy)
        
        return normalized_elements

    def _create_dynamic_level_mapping(self, pattern_types_found: List[str]) -> Dict[str, int]:
        """
        Create a dynamic level mapping based on the pattern types found in the document.
        This ensures that the highest-level pattern in the document becomes level 1.
        """
        level_mapping = {}
        
        markdown_patterns = [p for p in pattern_types_found if p.startswith('markdown_')]
        rulebased_patterns = [p for p in pattern_types_found if not p.startswith('markdown_')]
        
        # Handle markdown patterns (these already have explicit levels)
        for pattern in markdown_patterns:
            level = int(pattern.split('_')[1])
            level_mapping[pattern] = level
        
        # Handle rule-based patterns dynamically
        if rulebased_patterns:
            # Find the order based on hierarchy priority
            ordered_patterns = []
            for hierarchy_type in self.pattern_hierarchy:
                if hierarchy_type in rulebased_patterns:
                    ordered_patterns.append(hierarchy_type)
            
            # Assign levels starting from the next available level after markdown patterns
            start_level = max([int(p.split('_')[1]) for p in markdown_patterns], default=0) + 1
            
            # If no markdown patterns exist, start from level 1
            if not markdown_patterns:
                start_level = 1
            
            for i, pattern_type in enumerate(ordered_patterns):
                level_mapping[pattern_type] = start_level + i
        
        return level_mapping

    def _create_structural_chunks(self, elements: List[Dict]) -> List[StructuralChunk]:
        chunks = []
        current_content_elements = []
        current_heading_context = None
        current_special_section = None
        
        for element in elements:
            # Handle special section headers (Phụ lục, Chú thích)
            if element['type'] == 'special_section_header':
                if current_content_elements:
                    content_chunk = self._build_content_chunk_from_elements(
                        current_content_elements, 
                        current_heading_context
                    )
                    if content_chunk:
                        chunks.append(content_chunk)
                    current_content_elements = []
                
                special_section_type = element['special_section_type']
                current_special_section = special_section_type
                
                special_chunk = self._build_special_section_chunk(element)
                if special_chunk:
                    chunks.append(special_chunk)
                
                # Reset heading context (special sections are independent)
                current_heading_context = None
                
            elif element['type'] == 'heading':
                # First, create a chunk for any accumulated content under previous heading
                if current_content_elements:
                    content_chunk = self._build_content_chunk_from_elements(
                        current_content_elements, 
                        current_heading_context,
                        special_section=current_special_section
                    )
                    if content_chunk:
                        chunks.append(content_chunk)
                    current_content_elements = []
                
                # Create a separate chunk for the heading itself
                heading_chunk = self._build_heading_chunk_from_element(
                    element,
                    special_section=current_special_section
                )
                if heading_chunk:
                    chunks.append(heading_chunk)
                
                # Update heading context for subsequent content
                current_heading_context = element
                
            else:
                if element['type'] != 'empty_line' or element['content'].strip():
                    current_content_elements.append(element)
        
        # Process any remaining content at the end
        if current_content_elements:
            content_chunk = self._build_content_chunk_from_elements(
                current_content_elements, 
                current_heading_context,
                special_section=current_special_section
            )
            if content_chunk:
                chunks.append(content_chunk)
        
        return chunks

    def _build_special_section_chunk(self, element: Dict) -> Optional[StructuralChunk]:
        """Build a chunk for special section headers like Phụ lục, Chú thích"""
        special_section_type = element['special_section_type']
        
        type_mapping = {
            'appendix': 'Phụ lục',
            'footnote': 'Chú thích',
            'reference': 'Tài liệu tham khảo'
        }
        
        section_name = type_mapping.get(special_section_type, special_section_type.title())
        
        metadata = {
            'element_count': 1,
            'contains_table': False,
            'contains_list': False,
            'heading_only': True,
            'special_section_type': special_section_type,
            'special_section_name': section_name
        }
        
        return StructuralChunk(
            content=None,
            chunk_type=ChunkType.SECTION,
            level=1,
            section_hierarchy=[section_name],
            metadata=metadata,
            token_count=0,
        )

    def _build_heading_chunk_from_element(
        self, 
        heading_element: Dict,
        special_section: Optional[str] = None
    ) -> Optional[StructuralChunk]:
        if heading_element['type'] != 'heading':
            return None
            
        content = None
        level = heading_element['level']
        section_hierarchy = heading_element.get('section_stack', [])
        
        if special_section:
            type_mapping = {
                'appendix': 'Phụ lục',
                'footnote': 'Chú thích',
                'reference': 'Tài liệu tham khảo'
            }
            special_name = type_mapping.get(special_section, special_section.title())
            section_hierarchy = [special_name] + section_hierarchy
        
        metadata = {
            'element_count': 1,
            'contains_table': False,
            'contains_list': False,
            'heading_only': True,
            'pattern_type': heading_element.get('pattern_type', 'unknown'),
            'heading_text': heading_element['content'].strip(),
            'special_section': special_section
        }
        
        return StructuralChunk(
            content=content,
            chunk_type=ChunkType.SECTION if level == 1 else ChunkType.SUBSECTION,
            level=level,
            section_hierarchy=section_hierarchy,
            metadata=metadata,
            token_count=0,
        )
    
    def _build_content_chunk_from_elements(
        self, 
        content_elements: List[Dict], 
        heading_context: Optional[Dict],
        special_section: Optional[str] = None
    ) -> Optional[StructuralChunk]:
        """Build a chunk from content elements with level based on heading context."""
        if not content_elements:
            return None
            
        if not any(e['type'] != 'empty_line' for e in content_elements):
            return None
        
        content = self._elements_to_content(content_elements)
        
        if heading_context:
            content_level = heading_context['level'] + 1
            section_hierarchy = heading_context.get('section_stack', [])
        else:
            content_level = 1
            section_hierarchy = []
        
        if special_section:
            type_mapping = {
                'appendix': 'Phụ lục',
                'footnote': 'Chú thích',
                'reference': 'Tài liệu tham khảo'
            }
            special_name = type_mapping.get(special_section, special_section.title())
            section_hierarchy = [special_name] + section_hierarchy
        
        chunk_type = self._determine_content_chunk_type(content_elements)
        
        metadata = {
            'element_count': len(content_elements),
            'contains_table': any(e['type'] == 'table' for e in content_elements),
            'heading_only': False,
            'parent_heading': heading_context['heading_text'] if heading_context else None,
            'special_section': special_section
        }
        
        return StructuralChunk(
            content=content,
            chunk_type=chunk_type,
            level=content_level,
            section_hierarchy=section_hierarchy,
            metadata=metadata,
            token_count=0,
        )
    
    def _determine_content_chunk_type(self, elements: List[Dict]) -> ChunkType:
        type_counts = {}
        
        for element in elements:
            elem_type = element['type']
            type_counts[elem_type] = type_counts.get(elem_type, 0) + 1
        
        if 'table' in type_counts:
            return ChunkType.TABLE
        else:
            return ChunkType.PARAGRAPH

    def _recursive_chunking(self, structural_chunks: List[StructuralChunk]) -> List[StructuralChunk]:
        final_chunks = []
        
        for chunk in structural_chunks:
            if not chunk.is_oversized:
                final_chunks.append(chunk)
            else:
                sub_chunks = self._recursive_split_chunk(chunk)
                final_chunks.extend(sub_chunks)
        
        return final_chunks

    def _recursive_split_chunk(self, chunk: StructuralChunk) -> List[StructuralChunk]:
        sub_chunks = []
        
        split_texts = self._recursive_chunking(
            text=chunk.content,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )

        for i, text_chunk in enumerate(split_texts):
            if not text_chunk.strip():
                continue
            
            sub_chunk = self._create_sub_chunk(
                content=text_chunk,
                parent_chunk=chunk,
                chunk_type=ChunkType.RECURSIVE_SPLIT,
                chunk_index=i
            )

            if sub_chunk.token_count > self.chunk_size:
                print(f"   Sub-chunk still oversized ({sub_chunk.token_count} tokens), splitting further...")
                further_splits = self._recursive_split_chunk(sub_chunk)
                sub_chunks.extend(further_splits)
            else:
                sub_chunks.append(sub_chunk)
        
        return sub_chunks

    def _recursive_chunking(self, text: str, chunk_size: int = 512, chunk_overlap: int = 0) -> List[str]:
        if not text:
            return []
            
        separators = ["\n\n", "\n", ". ", "!", "?", ",", " ", ""]
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=lambda x: len(self.tokenizer.tokenize(x)),
            separators=separators
        )
        return text_splitter.split_text(text)
    
    def _create_sub_chunk(self, content: str, parent_chunk: StructuralChunk, 
                         chunk_type: ChunkType, chunk_index: int = 0) -> StructuralChunk:
        token_count = len(self.tokenizer.tokenize(content))
        
        parent_id = parent_chunk.parent_chunk_id or f"{parent_chunk.section_hierarchy[-1] if parent_chunk.section_hierarchy else 'root'}"
        
        return StructuralChunk(
            content=content,
            chunk_type=chunk_type,
            level=parent_chunk.level + 1,
            section_hierarchy=parent_chunk.section_hierarchy,
            metadata={
                **parent_chunk.metadata,
                'split_from_parent': True,
                'parent_type': parent_chunk.chunk_type.value,
                'chunk_index_in_parent': chunk_index,
                'original_parent_tokens': parent_chunk.token_count
            },
            token_count=token_count,
            is_oversized=token_count > self.chunk_size,
            parent_chunk_id=f"{parent_id}_{chunk_index}"
        )

    # Helper methods
    def _roman_to_int(self, roman: str) -> int:
        """Convert Roman numerals to integer"""
        roman_values = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
        total = 0
        prev_value = 0
        
        for char in reversed(roman.upper()):
            value = roman_values.get(char, 0)
            if value < prev_value:
                total -= value
            else:
                total += value
            prev_value = value
        
        return total
    
    def _is_valid_roman(self, roman: str) -> bool:
        if not roman:
            return False
        
        pattern = r'^[IVX]+$'
        if not re.match(pattern, roman.upper()):
            return False
        
        try:
            value = self._roman_to_int(roman)
            return 1 <= value <= 100
        except:
            return False
    
    def _detect_heading_level(self, line: str) -> Optional[str]:
        """Detect heading pattern type"""
        cleaned_line = re.sub(r'^[^#]*?(#+)\s*', r'\1 ', line.strip())
        for level, pattern in self.heading_patterns.items():
            if re.match(pattern, cleaned_line):
                return f"markdown_{level}"
        
        # Try original line for markdown patterns (in case cleaning broke something)
        for level, pattern in self.heading_patterns.items():
            if re.match(pattern, line.strip()):
                return f"markdown_{level}"
        
        # Rule-based patterns
        for pattern_type, patterns in self.heading_patterns_rulebased.items():
            for pattern in patterns:
                match = re.match(pattern, line.strip())
                if match:
                    if pattern_type in ['phan', 'chuong', 'roman'] and len(match.groups()) >= 1:
                        potential_roman = match.group(1)
                        if self._is_valid_roman(potential_roman):
                            return pattern_type
                    elif pattern_type not in ['phan', 'chuong', 'roman']:
                        return pattern_type
        
        return None

    def _extract_heading_text(self, line: str, pattern_type: str) -> str:
        """Extract heading text"""
        original_line = line.strip()
        
        # Markdown heading
        if pattern_type.startswith('markdown_'):
            level = int(pattern_type.split('_')[1])
            cleaned_line = re.sub(r'^[^#]*?(#+)\s*', r'\1 ', original_line)
            
            pattern = self.heading_patterns.get(level)
            if pattern:
                match = re.match(pattern, cleaned_line)
                if match:
                    return match.group(1)

                match = re.match(pattern, original_line)
                if match:
                    return match.group(1)

        else:
            patterns = self.heading_patterns_rulebased.get(pattern_type, [])
            for pattern in patterns:
                match = re.match(pattern, original_line)
                if match:
                    if len(match.groups()) >= 2:
                        return f"{match.group(1)}. {match.group(2)}" if not original_line.endswith('.') else original_line
                    else:
                        return original_line
        
        return original_line

    def _detect_special_section(self, line: str) -> Optional[str]:
        """
        Detect if a line marks the beginning of a special section like:
        - Appendix (Phụ lục)
        - Footnote (Chú thích)
        - Reference (Tài liệu tham khảo)
        
        Returns:
            str: Type of special section ('appendix', 'footnote', 'reference') or None
        """
        line_stripped = line.strip()
        
        # Check appendix patterns
        for pattern in self.special_section_patterns['appendix']:
            if re.match(pattern, line_stripped, re.IGNORECASE):
                logger.debug(f"Detected appendix section: {line_stripped}")
                return 'appendix'
        
        # Check footnote patterns
        for pattern in self.special_section_patterns['footnote']:
            if re.match(pattern, line_stripped, re.IGNORECASE):
                logger.debug(f"Detected footnote section: {line_stripped}")
                return 'footnote'
        
        # Check reference patterns
        for pattern in self.special_section_patterns['reference']:
            if re.match(pattern, line_stripped, re.IGNORECASE):
                logger.debug(f"Detected reference section: {line_stripped}")
                return 'reference'
        
        return None

    def _extract_table(self, lines: List[str], start_index: int) -> Tuple[str, int]:
        content_lines = []
        i = start_index
        
        while i < len(lines):
            line = lines[i]
            if re.match(self.structure_markers['table_row'], line):
                content_lines.append(line)
                i += 1
            elif not line.strip():
                i += 1
            else:
                break
        
        return '\n'.join(content_lines), i - 1

    def _extract_paragraph(self, lines: List[str], start_index: int) -> Tuple[str, int]:
        content_lines = []
        i = start_index
        
        while i < len(lines):
            line = lines[i]
            
            if (self._detect_heading_level(line) or
                self._detect_special_section(line) or
                re.match(self.structure_markers['table_row'], line)):
                break
            
            if not line.strip():
                if i + 1 < len(lines) and not lines[i + 1].strip():
                    break
            
            content_lines.append(line)
            i += 1
        
        return '\n'.join(content_lines), i - 1

    def _elements_to_content(self, elements: List[Dict]) -> str:
        content_parts = []
        for element in elements:
            if element['type'] != 'empty_line':
                content_parts.append(element['content'])
        return '\n\n'.join(content_parts)

if __name__ == "__main__":
    chunker = TwoPhaseDocumentChunker(chunk_size=4096)
    file_path = "data-processed/515.signed.md"
    with open(file_path, 'r', encoding='utf-8') as f:
        document_text = f.read()
    
    chunks = chunker.chunk_document(document_text)
    analysis = chunker.export_to_schema(chunks, show_structure=True)
    import json
    print(json.dumps(analysis, ensure_ascii=False, indent=2))
    
    with open("data-processed/515.signed.chunks.analysis.json", 'w', encoding='utf-8') as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)