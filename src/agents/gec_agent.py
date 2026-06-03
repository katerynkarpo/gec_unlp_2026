from typing import Type
from pydantic import BaseModel
from .base import BaseAgent
from ..models.gec import GECResponse
from .prompts.base import get_gec_prompt


class SinglePromptGECAgent(BaseAgent):
    """Agent for Grammatical Error Correction using zero-shot instruction."""

    def get_system_prompt(self) -> str:
        """Return the configured instruction prompt for GEC."""
        return get_gec_prompt(self.prompt_name or "base_en")

    def get_response_model(self) -> Type[BaseModel]:
        """Return the GEC response model."""
        return GECResponse
