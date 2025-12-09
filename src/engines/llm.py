import json
from loguru import logger
from typing import Optional, Any
from abc import ABC, abstractmethod

from openai import OpenAI

from src.config.schemas import Role
from src.config.setting import api_config, llm_config


class BaseLLM(ABC):
    def __init__(
        self,
        client: OpenAI,
        prompt_template: str,
        system_prompt: Optional[str] = None,
    ):
        if client is None:
            raise ValueError("OpenAI Client must be provided.")
        if prompt_template is None:
            raise ValueError("Prompt template is not defined for this LLM instance.")
        
        self.client = client
        self.system_prompt = system_prompt
        self.prompt_template = prompt_template

    def _inject_prompt(self, **kwargs: Any) -> str:
        """
        Injects the provided keyword arguments into the prompt template.
        Args:
            **kwargs: Keyword arguments to be injected into the prompt template.
            e.g. kwargs = {"entities": [...], "text": "..."}
        Returns:
            str: The formatted prompt string.
        """
        
        prompt_str = self.prompt_template.format(**kwargs)
        return prompt_str

    @abstractmethod
    def call(self, **kwargs: Any) -> dict:
        pass


class EntityExtractionLLM(BaseLLM):
    def __init__(
        self,
        prompt_template: str,
        client: Optional[OpenAI] = None,
        system_prompt: Optional[str] = None,
        json_schema: Optional[dict] = None,
    ) -> None:
        if client is None:
            client = OpenAI(api_key=api_config.api_key, base_url=api_config.base_url)

        super().__init__(client, prompt_template, system_prompt)
        
        self.json_schema = json_schema

    def call(self, **kwargs: Any) -> dict:
        nodes = []
        relationships = []
        
        prompt_str = self._inject_prompt(**kwargs)
        
        try:
            extraction_params = llm_config.extraction.model_dump()
            
            response = self.client.chat.completions.create(
                seed=llm_config.seed,
                stop=llm_config.stop_tokens,
                model=llm_config.llm_model,
                messages=[
                    {"role": Role.SYSTEM.value, "content": self.system_prompt},
                    {"role": Role.USER.value, "content": prompt_str},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": self.json_schema
                },
                **extraction_params
            )
            content = response.choices[0].message.content
            
            try:
                payload = json.loads(content)
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {content}")
                logger.error(f"JSON decode error: {e}")
                return {"nodes": [], "relationships": []}

            nodes = payload.get("nodes", []) if isinstance(payload, dict) else []
            relationships = payload.get("relationships", []) if isinstance(payload, dict) else []
            return {"nodes": nodes, "relationships": relationships}
        
        except Exception as e:
            logger.error(f"OpenAI API call failed: {str(e)}")
            raise e


class GenerationResponseLLM(BaseLLM):
    def __init__(
        self,
        prompt_template: str,
        client: Optional[OpenAI] = None,
        system_prompt: Optional[str] = None,
    ) -> None:
        if client is None:
            client = OpenAI(api_key=api_config.api_key, base_url=api_config.base_url)

        super().__init__(client, prompt_template, system_prompt)

    def call(self, **kwargs: Any) -> dict:
        prompt_str = self._inject_prompt(**kwargs)
        
        try:
            generation_params = llm_config.generation.model_dump()
            
            response = self.client.chat.completions.create(
                seed=llm_config.seed,
                stop=llm_config.stop_tokens,
                model=llm_config.llm_model,
                messages=[
                    {"role": Role.SYSTEM.value, "content": self.system_prompt},
                    {"role": Role.USER.value, "content": prompt_str},
                ],
                response_format={"type": "json_object"},
                **generation_params
            )
            content = response.choices[0].message.content
            return content
        
        except Exception as e:
            logger.error(f"OpenAI API call failed: {str(e)}")
            raise e