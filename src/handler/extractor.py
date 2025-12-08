from openai import OpenAI
from typing import Optional

from src.utils.utils import parse_json
from src.config.setting import llm_config
from src.prompt.ner import EXTRACT_SYSTEM_PROMPT, EXTRACT_PROMPT

class GraphExtractorLLM:
    def __init__(self, llm: Optional[OpenAI] = None):
        if llm is None:
            llm = OpenAI(api_key=llm_config.api_key, base_url=llm_config.base_url)

        self.llm = llm
        self.prompt_template = EXTRACT_PROMPT

    def _inject_prompt(self, text: str) -> str:
        prompt_str = self.prompt_template.format(
            text=text,
        )
        return prompt_str

    def call(self, doc: str) -> dict:
        nodes = []
        relationships = []
        
        prompt_str = self._inject_prompt(doc)
        
        response = self.llm.chat.completions.create(
            seed=llm_config.seed,
            temperature=llm_config.temperature,
            top_p=llm_config.top_p,
            model=llm_config.llm_model,
            messages=[
                {"role": "system", "content": EXTRACT_SYSTEM_PROMPT},
                {"role": "user", "content": prompt_str},
            ],
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        output_structure = parse_json(content)
        print(output_structure)

        nodes.extend(output_structure.get("nodes", []))
        relationships.extend(output_structure.get("relationships", []))


        return {"nodes": nodes, "relationships": relationships}
