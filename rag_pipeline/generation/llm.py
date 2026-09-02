"""Phase 6: LLM client interface + Groq implementation.

The API key is read at call time from the environment variable *named* by
``settings.llm_api_key_env`` (default ``GROQ-KEY``). We ``load_dotenv()`` so a
hyphenated key in .env resolves via ``os.environ``. The key value is never
logged or returned.
"""

from __future__ import annotations

import os
from typing import Protocol

from rag_pipeline.config import LLMProvider, Settings, get_settings


class LLMClient(Protocol):
    def complete(self, system: str, user: str) -> str:
        """Return the model's reply to a system + user message."""


def _resolve_api_key(env_name: str) -> str:
    try:
        from dotenv import load_dotenv

        load_dotenv()  # populate os.environ from .env (contents never inspected here)
    except Exception:  # noqa: BLE001
        pass
    key = os.environ.get(env_name)
    if not key:
        raise RuntimeError(
            f"LLM API key not found in environment variable {env_name!r}. "
            f"Add it to your .env as `{env_name}=...`."
        )
    return key


class GroqClient:
    """Groq chat-completions client (OpenAI-compatible)."""

    def __init__(
        self,
        *,
        model: str,
        api_key_env: str,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        base_url: str = "",
    ) -> None:
        self.model = model
        self.api_key_env = api_key_env
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.base_url = base_url or None

    def complete(self, system: str, user: str) -> str:
        from groq import Groq  # lazy: needs '.[llm-groq]'

        kwargs = {"api_key": _resolve_api_key(self.api_key_env)}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        client = Groq(**kwargs)
        try:
            resp = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        except Exception as exc:  # noqa: BLE001 - surface a clean message, not a traceback
            status = getattr(exc, "status_code", None)
            message = getattr(exc, "message", None) or str(exc)
            suffix = f" (status {status})" if status else ""
            raise RuntimeError(
                f"Groq request failed for model {self.model!r}{suffix}: {message}"
            ) from exc
        return (resp.choices[0].message.content or "").strip()


def get_llm(settings: Settings | None = None) -> LLMClient:
    settings = settings or get_settings()
    if settings.llm_provider is LLMProvider.groq:
        return GroqClient(
            model=settings.llm_model,
            api_key_env=settings.llm_api_key_env,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            base_url=settings.llm_base_url,
        )
    raise NotImplementedError(
        f"LLM provider {settings.llm_provider!r} not implemented yet (default: groq)."
    )
