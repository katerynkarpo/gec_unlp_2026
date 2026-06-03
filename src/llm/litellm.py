import os

import litellm
from litellm import Router

litellm.drop_params = True

model_list = [
    # ── Anthropic ──
    {
        "model_name": "claude-sonnet-4.6",
        "litellm_params": {
            "model": "claude-sonnet-4-6",
            "api_key": os.environ.get("ANTHROPIC_API_KEY", ""),
        },
    },
    {
        "model_name": "claude-opus-4.6",
        "litellm_params": {
            "model": "claude-opus-4-6",
            "api_key": os.environ.get("ANTHROPIC_API_KEY", ""),
        },
    },
    # ── OpenAI ──
    {
        "model_name": "gpt-4o",
        "litellm_params": {
            "model": "gpt-4o",
            "api_key": os.environ.get("OPENAI_API_KEY", ""),
        },
    },
    {
        "model_name": "gpt-4.1-mini",
        "litellm_params": {
            "model": "gpt-4.1-mini",
            "api_key": os.environ.get("OPENAI_API_KEY", ""),
        },
    },
    {
        "model_name": "gpt-5-mini",
        "litellm_params": {
            "model": "gpt-5-mini",
            "api_key": os.environ.get("OPENAI_API_KEY", ""),
        },
    },
    {
        "model_name": "gpt-4.1",
        "litellm_params": {
            "model": "gpt-4.1",
            "api_key": os.environ.get("OPENAI_API_KEY", ""),
        },
    },
    {
        "model_name": "gpt-5.1",
        "litellm_params": {
            "model": "gpt-5.1",
            "api_key": os.environ.get("OPENAI_API_KEY", ""),
        },
    },
    {
        "model_name": "gpt-5.2",
        "litellm_params": {
            "model": "gpt-5.2",
            "api_key": os.environ.get("OPENAI_API_KEY", ""),
        },
    },
    {
        "model_name": "gpt-5.4",
        "litellm_params": {
            "model": "gpt-5.4",
            "api_key": os.environ.get("OPENAI_API_KEY", ""),
        },
    },
    # ── Google ──
    {
        "model_name": "gemini-3-flash",
        "litellm_params": {
            "model": "gemini/gemini-3-flash-preview",
            "api_key": os.environ.get("GEMINI_API_KEY", ""),
        },
    },
    # { # DEPRECATED
    #     "model_name": "gemini-3-pro",
    #     "litellm_params": {
    #         "model": "gemini/gemini-3-pro-preview",
    #         "api_key": os.environ.get("GEMINI_API_KEY", ""),
    #     },
    # },
    {
        "model_name": "gemini-3.1-flash-lite",
        "litellm_params": {
            "model": "gemini/gemini-3.1-flash-lite-preview",
            "api_key": os.environ.get("GEMINI_API_KEY", ""),
        },
    },
    {
        "model_name": "gemini-3.1-pro",
        "litellm_params": {
            "model": "gemini/gemini-3.1-pro-preview",
            "api_key": os.environ.get("GEMINI_API_KEY", ""),
        },
    },
    {
        "model_name": "gemma-4-26b",
        "litellm_params": {
            "model": "gemini/gemma-4-26b-a4b-it",
            "api_key": os.environ.get("GEMINI_API_KEY", ""),
        },
    },
    {
        "model_name": "gemma-4-31b",
        "litellm_params": {
            "model": "gemini/gemma-4-31b-it",
            "api_key": os.environ.get("GEMINI_API_KEY", ""),
        },
    },
    # ── Moonshot ──
    {
        "model_name": "kimi-k2.5",
        "litellm_params": {
            "model": "moonshot/kimi-k2.5",
            "api_key": os.environ.get("MOONSHOT_API_KEYa", ""),
            "api_base": "https://api.moonshot.ai/v1",
        },
    },
    {
        "model_name": "kimi-k2-0905-preview",
        "litellm_params": {
            "model": "moonshot/kimi-k2-0905-preview",
            "api_key": os.environ.get("MOONSHOT_API_KEYa", ""),
            "api_base": "https://api.moonshot.ai/v1",
        },
    },
]

router = Router(
    model_list=model_list,
    # num_retries=3,
    # retry_after=5,
    enable_pre_call_checks=True,
    cache_responses=True,
)