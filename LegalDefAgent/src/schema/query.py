from typing import Optional
from datetime import datetime
from pydantic import BaseModel, Field


class Query(BaseModel):
    """Query schema for definition retrieval."""

    definendum: str = Field(description="The term to define")
    legislation_filter: Optional[str] = Field(None, description="The legislation to search for the term in")
    time_filter: Optional[tuple] = Field(None, description="Tuple representing the date range to search for the term in")
