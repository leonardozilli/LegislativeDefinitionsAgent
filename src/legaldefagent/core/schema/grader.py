from pydantic import BaseModel, Field


class DefinitionRelevance(BaseModel):
    """Binary score for relevance check on retrieved definitions."""

    binary_score: str = Field(
        description="Definitions are relevant to the question, 'yes' or 'no'"
    )
