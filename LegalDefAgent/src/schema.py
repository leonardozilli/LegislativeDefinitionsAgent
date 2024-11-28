from pydantic import BaseModel, Field
from typing import Annotated, Literal, Sequence, List, Any, Dict

class DefinitionMetadata(BaseModel):
    id: int = Field(description="the unique identifier of the definition")
    dataset: str = Field(description="the dataset the definition is from")
    document_id: str = Field(description="the document id the definition is from")
    references: List[str] = Field(description="the references mentioned the definition.")

class Definition(BaseModel):
    metadata: DefinitionMetadata
    definition_text: str = Field(description="the full text of the definition")  # Changed from 'definition'

class DefinitionsList(BaseModel):
    relevant_definitions: List[Definition] = Field(description="a list of relevant definitions")