from abc import ABC, abstractmethod
import json
import re
from typing import Any, Optional, Type
from pydantic import BaseModel


class _SafeFormatDict(dict):
    """Keep unknown format placeholders unchanged."""

    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


class BaseAgent(ABC):
    """Base class for all agents in the system."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        top_p: Optional[float] = None,
        request_timeout: float = 120.0,
        prompt_name: Optional[str] = None,
        llm_router: Any = None,
        reasoning_effort: Optional[str] = None,
    ):
        """
        Initialize the base agent.

        Args:
            model: Model group/name resolved by LiteLLM Router
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            request_timeout: Per-request timeout (seconds) passed to LiteLLM
            prompt_name: Optional prompt selector for agent subclasses
            llm_router: Optional LiteLLM Router instance
            reasoning_effort: Optional reasoning effort level (low/medium/high) for supported models
        """
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.request_timeout = request_timeout
        self.prompt_name = prompt_name
        self.llm_router = llm_router
        self.reasoning_effort = reasoning_effort

        if self.llm_router is None:
            raise ValueError("llm_router must be provided.")

    @abstractmethod
    def get_system_prompt(self) -> str:
        """Return the system prompt for this agent."""
        pass

    @abstractmethod
    def get_response_model(self) -> Type[BaseModel]:
        """Return the Pydantic model for structured output."""
        pass

    def execute(self, input_text: str, **kwargs) -> BaseModel:
        """
        Execute the agent on input text.

        Args:
            input_text: Input text to process
            **kwargs: Additional parameters for the API call

        Returns:
            Structured response according to the agent's response model
        """
        return self.execute_with_model(input_text=input_text, response_model=self.get_response_model(), **kwargs)

    def execute_with_model(
        self,
        input_text: str,
        response_model: Type[BaseModel],
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> BaseModel:
        """Execute using a custom structured response model."""
        raw_system_prompt = system_prompt if system_prompt is not None else self.get_system_prompt()
        system_prompt = self._render_system_prompt(raw_system_prompt, input_text)

        router_messages = [
            {"role": "system", "content": system_prompt},
        ]
        # Anthropic via LiteLLM requires at least one non-system message.
        router_user_content = input_text if str(input_text).strip() else "Please continue."
        router_messages.append({"role": "user", "content": router_user_content})

        router_kwargs = {
            "model": self.model,
            "messages": router_messages,
            "temperature": self.temperature,
            "response_format": self._build_router_response_format(response_model),
            **kwargs,
        }
        if self.top_p is not None:
            router_kwargs["top_p"] = self.top_p
        if self.reasoning_effort is not None:
            router_kwargs["reasoning_effort"] = self.reasoning_effort
        if "timeout" not in router_kwargs:
            router_kwargs["timeout"] = self.request_timeout

        completion = self._router_completion_with_fallback(router_kwargs)
        content = completion.choices[0].message.content
        if isinstance(content, dict):
            return response_model.model_validate(content)
        if isinstance(content, list):
            content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        try:
            parsed_payload = self._extract_json_payload(content)
            return response_model.model_validate(parsed_payload)
        except Exception:
            # Fallback for single-field sentence responses when model returns malformed JSON.
            if set(response_model.model_fields.keys()) == {"corrected_sentence"}:
                repaired = self._recover_corrected_sentence(content)
                return response_model.model_validate({"corrected_sentence": repaired})
            raise

    def _router_completion_with_fallback(self, router_kwargs: dict[str, Any]) -> Any:
        """Retry router completion with response_format downgrade fallback.

        Most unsupported params (top_p, temperature, reasoning_effort, etc.) are
        handled globally by ``litellm.drop_params = True``. This method only
        retries when the provider rejects the ``json_schema`` response format,
        downgrading it to ``json_object``.
        """
        try:
            return self.llm_router.completion(**router_kwargs)
        except Exception as exc:
            message = str(exc)
            if ("response_format" in message or "json_schema" in message) and "response_format" in router_kwargs:
                fallback_kwargs = dict(router_kwargs)
                fallback_kwargs["response_format"] = {"type": "json_object"}
                return self.llm_router.completion(**fallback_kwargs)
            raise

    @staticmethod
    def _build_router_response_format(response_model: Type[BaseModel]) -> dict[str, Any]:
        """Build provider-agnostic schema config for LiteLLM structured output."""
        schema = BaseAgent._enforce_closed_object_schema(response_model.model_json_schema())
        return {
            "type": "json_schema",
            "json_schema": {
                "name": response_model.__name__,
                "schema": schema,
                "strict": True,
            },
        }

    @staticmethod
    def _enforce_closed_object_schema(node: Any) -> Any:
        """Ensure all JSON Schema object nodes declare additionalProperties=false."""
        if isinstance(node, dict):
            is_object = node.get("type") == "object" or "properties" in node
            if is_object and "additionalProperties" not in node:
                node["additionalProperties"] = False
            for key, value in list(node.items()):
                node[key] = BaseAgent._enforce_closed_object_schema(value)
            return node
        if isinstance(node, list):
            return [BaseAgent._enforce_closed_object_schema(item) for item in node]
        return node

    @staticmethod
    def _render_system_prompt(system_prompt: str, input_text: str) -> str:
        """Inject input text into prompt templates that use {input_text}."""
        if "{input_text}" in system_prompt:
            return system_prompt.format_map(_SafeFormatDict(input_text=input_text))
        return system_prompt

    @staticmethod
    def _extract_json_payload(content: str) -> dict[str, Any]:
        """Extract a JSON object from model output."""
        if not content:
            raise ValueError("Model returned empty response while JSON was expected.")

        text = content.strip()
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.DOTALL)
        if fenced:
            return json.loads(fenced.group(1))

        first_brace = text.find("{")
        last_brace = text.rfind("}")
        if first_brace != -1 and last_brace > first_brace:
            return json.loads(text[first_brace:last_brace + 1])

        raise ValueError(f"Could not parse JSON object from model response: {text}")

    @staticmethod
    def _recover_corrected_sentence(content: str) -> str:
        """Best-effort recovery for malformed single-field JSON outputs."""
        text = (content or "").strip()
        if not text:
            return ""

        # Common case: malformed JSON containing quoted chunks.
        quoted = re.findall(r'"([^"\\]*(?:\\.[^"\\]*)*)"', text, flags=re.DOTALL)
        if quoted:
            if quoted[0] == "corrected_sentence":
                parts = [p for p in quoted[1:] if p.strip()]
                if parts:
                    return "".join(parts).strip()
            return quoted[-1].strip()

        # Fallback to plain text cleanup.
        cleaned = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
        cleaned = cleaned.replace("{", "").replace("}", "")
        cleaned = cleaned.replace("corrected_sentence", "")
        cleaned = cleaned.replace(":", " ").strip(" \n\t\"',")
        return cleaned.strip()
