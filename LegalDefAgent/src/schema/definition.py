from pydantic import BaseModel, Field
from typing import List


class DefinitionMetadata(BaseModel):
    id: int = Field(description="the identifier of the definition in the database")
    dataset: str = Field(description="the dataset the definition is from")
    document_id: str = Field(description="the document id the definition is from")
    def_n: str = Field(description="the identifier of the definition in the document")
    references: List[str] = Field(description="the references mentioned the definition.")

class Definition(BaseModel):
    definition_text: str = Field(description="the full text of the definition")
    metadata: DefinitionMetadata

class DefinitionsList(BaseModel):
    relevant_definitions: List[Definition] = Field(description="a dict of relevant definitions")

