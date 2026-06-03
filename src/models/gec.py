from pydantic import BaseModel, Field


class GECResponse(BaseModel):
    """Response model for Grammatical Error Correction."""
    corrected_sentence: str = Field(..., description="Corrected version of the input sentence")
