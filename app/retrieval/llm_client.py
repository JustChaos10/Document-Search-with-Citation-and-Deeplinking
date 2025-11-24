from __future__ import annotations

import logging
from typing import Any, Optional, TYPE_CHECKING

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages.ai import AIMessage

from app.config import AppConfig

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from openai import OpenAI as OpenAIClient
else:
    OpenAIClient = Any

try:
    from openai import OpenAI as OpenAIActual
except ImportError as exc:  # pragma: no cover - runtime availability
    OpenAIActual = None  # type: ignore
    _OPENAI_IMPORT_ERROR = exc
else:
    _OPENAI_IMPORT_ERROR = None


class GroqChatModel:
    """Minimal wrapper around Groq's Responses API that matches LangChain expectations."""

    def __init__(
        self,
        model_name: str,
        api_key: str,
        base_url: str,
        temperature: float = 0.1,
        top_p: float = 0.95,
        max_output_tokens: int = 4096,
        client: Optional[OpenAIClient] = None,
    ) -> None:
        if client is None:
            if OpenAIActual is None:
                raise ImportError(
                    "The openai package is required to communicate with the Groq API."
                ) from _OPENAI_IMPORT_ERROR
            client = OpenAIActual(api_key=api_key, base_url=base_url)
        self._client = client
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.max_output_tokens = max_output_tokens

    def invoke(self, prompt: str) -> AIMessage:
        response = self._client.responses.create(
            model=self.model_name,
            input=prompt,
            temperature=self.temperature,
            top_p=self.top_p,
            max_output_tokens=self.max_output_tokens,
        )
        content = self._extract_text(response)
        logger.debug("llm.groq_response", extra={"model": self.model_name, "text_length": len(content)})
        return AIMessage(content=content or "", additional_kwargs={"response": response})

    def _extract_text(self, response: Any) -> str:
        text = getattr(response, "output_text", None)
        if text:
            return str(text)
        output = getattr(response, "output", None)
        if isinstance(output, list):
            fragments: list[str] = []
            for item in output:
                if isinstance(item, dict):
                    content = item.get("content")
                else:
                    content = None
                if isinstance(content, str):
                    fragments.append(content)
                    continue
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, str):
                            fragments.append(block)
                        elif isinstance(block, dict) and "text" in block:
                            fragments.append(str(block["text"]))
                elif hasattr(item, "text"):
                    fragments.append(str(getattr(item, "text")))
            if fragments:
                return "".join(fragments)
        return ""


def build_chat_model(config: AppConfig) -> ChatGoogleGenerativeAI | GroqChatModel:
    if config.llm_provider == "groq":
        return GroqChatModel(
            model_name=config.groq_model_name,
            api_key=config.groq_api_key,
            base_url=config.groq_api_base_url,
        )
    return ChatGoogleGenerativeAI(
        model=config.gemini_model_name,
        google_api_key=config.gemini_api_key,
        temperature=0.1,
        top_p=0.95,
        max_output_tokens=4096,
        response_mime_type="application/json",
    )
