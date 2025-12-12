import re
from typing import List, Union


class EntityPostprocessor:
    def __call__(self, text: Union[str, List[str]]) -> Union[str, List[str]]:
        """Make the class callable for easy usage."""
        if isinstance(text, list):
            return [self.clean_llm_output(t) for t in text]
        return self.clean_llm_output(text)

    def clean_llm_output(self, text: str) -> str:
        """Cleans a string extracted by an LLM by removing common artifacts."""
        if not isinstance(text, str):
            return ""

        cleaned_text = text.strip()
        
        # Remove leading junk characters and patterns
        leading_junk_pattern = r'^\s*(\d+[\.\)]\s*|[\*\-•\+]\s*|\([a-zA-Z\d]+\)\s*)*'
        cleaned_text = re.sub(leading_junk_pattern, '', cleaned_text)
        
        # Remove trailing parenthetical remarks: "ABC (XYZ)" -> "ABC"
        trailing_parenthesis_pattern = r'\s+\([^)]*\)$'
        cleaned_text = re.sub(trailing_parenthesis_pattern, '', cleaned_text)
        
        # Remove surrounding junk characters iteratively
        surrounding_junk_pattern = r'^[\s_#$!@+\-*/()\[\]{}%"\']+|[\s_#$!@+\-*/()\[\]{}%"\']+$'
        old_text = None
        while old_text != cleaned_text:
            old_text = cleaned_text
            cleaned_text = re.sub(surrounding_junk_pattern, '', cleaned_text)

        # Remove trailing ellipses
        trailing_ellipsis_pattern = r'\s*\.{3,}$'
        cleaned_text = re.sub(trailing_ellipsis_pattern, '', cleaned_text)
        
        cleaned_text = cleaned_text.strip()
        
        return cleaned_text
