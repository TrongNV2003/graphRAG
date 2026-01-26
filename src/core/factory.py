from typing import Optional
from openai import OpenAI, AsyncOpenAI

from src.config.setting import api_config, llm_config


class LLMClientFactory:
    """
    Factory for creating LLM clients.
    Encapsulates the logic for selecting providers (OpenAI, Custom, etc).
    """
    @staticmethod
    def create_sync_client(model_name: Optional[str] = None) -> OpenAI:
        """Create a synchronous OpenAI client based on configuration (for LLM)."""
        model = model_name or llm_config.llm_model
        
        if LLMClientFactory._is_openai_native(model):
            return OpenAI(api_key=api_config.openai_api_key)
        else:
            return OpenAI(
                api_key=api_config.api_key or "EMPTY",
                base_url=api_config.base_url
            )

    @staticmethod
    def create_async_client(model_name: Optional[str] = None) -> AsyncOpenAI:
        """Create an asynchronous OpenAI client based on configuration (for LLM)."""
        model = model_name or llm_config.llm_model
        
        if LLMClientFactory._is_openai_native(model):
            return AsyncOpenAI(api_key=api_config.openai_api_key)
        else:
            return AsyncOpenAI(
                api_key=api_config.api_key or "EMPTY",
                base_url=api_config.base_url
            )

    @staticmethod
    def _is_openai_native(model_name: str) -> bool:
        """Determine if the model is an official OpenAI model."""
        return model_name.lower().startswith(("gpt-", "o1-", "openai/"))
