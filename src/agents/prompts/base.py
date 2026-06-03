from .prompt_01_zero_shot import EN_BASE_GEC_PROMPT, UA_BASE_GEC_PROMPT
from .prompt_02_few_shot import EN_BASE_GEC_PROMPT_FEW_SHOT, UA_BASE_FEW_SHOT
from .prompt_03_minimal_edits import (
    UA_MINIMAL_EDITS_PROMPT,
    UA_MINIMAL_EDITS_PROMPT_FEW_SHOT,
)
from .prompt_04_optimised_minimal_edits import OPTIMISED_GPT_FAMILY_PROMPT


GEC_PROMPTS = {
    "zero_shot_en": EN_BASE_GEC_PROMPT,
    "zero_shot_ua": UA_BASE_GEC_PROMPT,
    "few_shot_en": EN_BASE_GEC_PROMPT_FEW_SHOT,
    "few_shot_ua": UA_BASE_FEW_SHOT,
    "minimal_edits_ua": UA_MINIMAL_EDITS_PROMPT,
    "minimal_edits_few_shot_ua": UA_MINIMAL_EDITS_PROMPT_FEW_SHOT,
    "optimised_minimal_edits_ua": OPTIMISED_GPT_FAMILY_PROMPT,
}


def get_gec_prompt(name: str) -> str:
    """Return a named GEC system prompt."""
    if name not in GEC_PROMPTS:
        available = ", ".join(sorted(GEC_PROMPTS.keys()))
        raise ValueError(f"Unknown prompt '{name}'. Available prompts: {available}")
    return GEC_PROMPTS[name]
