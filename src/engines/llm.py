import json
from loguru import logger
from jinja2 import Template
from typing import Optional, Any
from abc import ABC, abstractmethod

from openai import OpenAI

from src.config.datatype import RoleType
from src.config.setting import api_config, llm_config


class PromptMixin:
    """
    Mixin class to handle Jinja2 prompt injection.
    Requires the consumer class to have a 'prompt_template' attribute.
    """
    def inject_prompt(self, **kwargs: Any) -> str:
        """
        Injects the provided keyword arguments into the prompt template using Jinja2.
        Args:
            **kwargs: Keyword arguments to be injected into the prompt template.
            e.g. kwargs = {"entities": [...], "text": "..."}
        Returns:
            str: The formatted prompt string.
        """
        if not hasattr(self, 'prompt_template') or self.prompt_template is None:
            raise AttributeError(f"{self.__class__.__name__} must define 'prompt_template' to use PromptMixin.")
        
        template = Template(self.prompt_template)
        prompt_str = template.render(**kwargs)
        return prompt_str


class BaseLLM(PromptMixin, ABC):
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

    @abstractmethod
    def call(self, **kwargs: Any) -> Any:
        pass


class EntityExtractionLLM(BaseLLM):
    def __init__(
        self,
        prompt_template: str,
        system_prompt: Optional[str] = None,
        json_schema: Optional[dict] = None,
        client: Optional[OpenAI] = None,
    ) -> None:
        if client is None:
            is_openai_model = llm_config.llm_model.lower().startswith(("gpt-", "o1-", "openai/"))
            
            if is_openai_model and api_config.openai_api_key:
                logger.info(f"Using OpenAI API for model: {llm_config.llm_model}")
                client = OpenAI(api_key=api_config.openai_api_key)
            else:
                logger.info(f"Using Custom API for model: {llm_config.llm_model}")
                client = OpenAI(
                    api_key=api_config.api_key or "EMPTY", 
                    base_url=api_config.base_url
                )

        super().__init__(client, prompt_template, system_prompt)
        
        self.json_schema = json_schema

    def call(self, **kwargs: Any) -> dict:
        nodes = []
        relationships = []
        
        prompt_str = self.inject_prompt(**kwargs)
        
        try:
            extraction_params = llm_config.extraction.model_dump()
            
            response = self.client.chat.completions.create(
                seed=llm_config.seed,
                stop=llm_config.stop_tokens,
                model=llm_config.llm_model,
                messages=[
                    {"role": RoleType.SYSTEM.value, "content": self.system_prompt},
                    {"role": RoleType.USER.value, "content": prompt_str},
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
                raise

            nodes = payload.get("nodes", []) if isinstance(payload, dict) else []
            relationships = payload.get("relationships", []) if isinstance(payload, dict) else []
            return {"nodes": nodes, "relationships": relationships}
        
        except Exception as e:
            logger.error(f"OpenAI API call failed: {str(e)}")
            raise


class GenerationResponseLLM(BaseLLM):
    def __init__(
        self,
        prompt_template: str,
        system_prompt: Optional[str] = None,
        client: Optional[OpenAI] = None,
    ) -> None:
        if client is None:
            is_openai_model = llm_config.llm_model.lower().startswith(("gpt-", "o1-", "openai/"))
            
            if is_openai_model and api_config.openai_api_key:
                logger.info(f"Using OpenAI API for model: {llm_config.llm_model}")
                client = OpenAI(api_key=api_config.openai_api_key)
            else:
                logger.info(f"Using Custom API for model: {llm_config.llm_model}")
                client = OpenAI(
                    api_key=api_config.api_key or "EMPTY", 
                    base_url=api_config.base_url
                )

        super().__init__(client, prompt_template, system_prompt)

    def call(self, **kwargs: Any) -> dict:
        prompt_str = self.inject_prompt(**kwargs)
        
        try:
            generation_params = llm_config.generation.model_dump()
            
            response = self.client.chat.completions.create(
                seed=llm_config.seed,
                stop=llm_config.stop_tokens,
                model=llm_config.llm_model,
                messages=[
                    {"role": RoleType.SYSTEM.value, "content": self.system_prompt},
                    {"role": RoleType.USER.value, "content": prompt_str},
                ],
                response_format={"type": "json_object"},
                **generation_params
            )
            content = response.choices[0].message.content
            
            if not content:
                raise ValueError("OpenAI returned empty content.")
            
            return json.loads(content)
        
        except json.JSONDecodeError:
            logger.error(f"Failed to parse JSON from OpenAI response: {content}")
            raise
        except Exception as e:
            logger.error(f"OpenAI API call failed: {str(e)}")
            raise 
        
class AnalysisQueryLLM(BaseLLM):
    def __init__(
        self,
        prompt_template: str,
        system_prompt: Optional[str] = None,
        client: Optional[OpenAI] = None,
        json_schema: Optional[dict] = None,
    ) -> None:
        if client is None:
            is_openai_model = llm_config.llm_model.lower().startswith(("gpt-", "o1-", "openai/"))
            
            if is_openai_model and api_config.openai_api_key:
                logger.info(f"Using OpenAI API for model: {llm_config.llm_model}")
                client = OpenAI(api_key=api_config.openai_api_key)
            else:
                logger.info(f"Using Custom API for model: {llm_config.llm_model}")
                client = OpenAI(
                    api_key=api_config.api_key or "EMPTY", 
                    base_url=api_config.base_url
                )

        super().__init__(client, prompt_template, system_prompt)
        self.json_schema = json_schema

    def call(self, **kwargs: Any) -> dict:
        prompt_str = self.inject_prompt(**kwargs)
        
        try:
            generation_params = llm_config.generation.model_dump()
            
            response = self.client.chat.completions.create(
                seed=llm_config.seed,
                stop=llm_config.stop_tokens,
                model=llm_config.llm_model,
                messages=[
                    {"role": RoleType.SYSTEM.value, "content": self.system_prompt},
                    {"role": RoleType.USER.value, "content": prompt_str},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": self.json_schema
                },
                **generation_params
            )
            content = response.choices[0].message.content

            if not content:
                raise ValueError("OpenAI returned empty content.")
            
            return json.loads(content)

        except json.JSONDecodeError:
            logger.error(f"Failed to parse JSON from OpenAI response: {content}")
            raise
        
        except Exception as e:
            logger.error(f"OpenAI API call failed: {str(e)}")
            raise e