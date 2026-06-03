from pydantic import BaseModel, Field


class AgentRequest(BaseModel):
    """Base request model for agent inputs."""
    input_text: str = Field(..., description="Input text to process")


class AgentResponse(BaseModel):
    """Base response model for agent outputs."""
    output_text: str = Field(..., description="Processed output text")
