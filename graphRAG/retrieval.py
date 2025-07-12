import argparse
from openai import OpenAI
from typing import Optional

from graphRAG.utils.utils import parse_json
from graphRAG.services.queries import GraphRetriever
from graphRAG.config.setting import llm_config, neo4j_config
from graphRAG.prompt.prompts import ANSWERING_SYSTEM_PROMPT, ANSWERING_PROMPT


class GraphQuerying:
    def __init__(self, llm: Optional[OpenAI] = None):
        if llm is None:
            llm = OpenAI(api_key=llm_config.api_key, base_url=llm_config.base_url)

        self.llm = llm
        self.retriever = GraphRetriever(
            url=neo4j_config.url,
            username=neo4j_config.username,
            password=neo4j_config.password
        )
        self.prompt_template = ANSWERING_PROMPT
        
    def _inject_prompt(self, query: str) -> str:
        prompt_str = self.prompt_template.format(
            schemas=self.retriever.retrieve(query),
            query=query,
        )
        return prompt_str

    def response(self, query: str) -> dict:
        prompt_str = self._inject_prompt(query)
        
        response = self.llm.chat.completions.create(
            seed=llm_config.seed,
            temperature=llm_config.temperature,
            top_p=llm_config.top_p,
            model=llm_config.model,
            messages=[
                {"role": "system", "content": ANSWERING_SYSTEM_PROMPT},
                {"role": "user", "content": prompt_str},
            ],
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        output_structure = parse_json(content)

        return output_structure
    
if __name__ == "__main__":
    queries = GraphQuerying()

    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, default="Elizabeth I")
    args = parser.parse_args()

    response = queries.response(query=args.query)
    print(f"Query: {args.query}")
    print(response)