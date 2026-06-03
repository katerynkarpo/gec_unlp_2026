from __future__ import annotations

import os
from typing import Any

import litellm
from litellm.router import Router


ENV_PREFIX = "env:"
_PATCHED_ASYNC_HANDLER_DEL = False


def _resolve_env_placeholders(value: Any) -> Any:
    """Resolve strings like 'env:OPENAI_API_KEY' recursively inside dicts/lists."""
    if isinstance(value, str) and value.startswith(ENV_PREFIX):
        env_var = value[len(ENV_PREFIX):]
        return os.getenv(env_var)
    if isinstance(value, dict):
        return {k: _resolve_env_placeholders(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_env_placeholders(v) for v in value]
    return value


def create_router(config: dict[str, Any]) -> Router:
    """Create LiteLLM router from config dictionary."""
    # Automatically drop params unsupported by a given provider (e.g. reasoning_effort,
    # top_p on Anthropic, etc.) so the same config works across different models.
    litellm.drop_params = True
    # Avoid asyncio cleanup warnings from async HTTP handlers in multithreaded runs.
    litellm.use_aiohttp_transport = False
    litellm.disable_aiohttp_transport = True
    _patch_litellm_async_handler_del()
    resolved = _resolve_env_placeholders(config)
    return Router(**resolved)


def _patch_litellm_async_handler_del() -> None:
    """
    Disable LiteLLM AsyncHTTPHandler.__del__ create_task cleanup.

    LiteLLM's destructor schedules an async close task, which may remain pending
    when worker threads/event loops are torn down, causing:
    "Task was destroyed but it is pending!" warnings.
    We rely on explicit process-level cleanup instead.
    """
    global _PATCHED_ASYNC_HANDLER_DEL
    if _PATCHED_ASYNC_HANDLER_DEL:
        return
    try:
        from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
    except Exception:
        return

    def _no_op_del(self) -> None:  # pragma: no cover - defensive runtime patch
        return

    AsyncHTTPHandler.__del__ = _no_op_del
    _PATCHED_ASYNC_HANDLER_DEL = True
