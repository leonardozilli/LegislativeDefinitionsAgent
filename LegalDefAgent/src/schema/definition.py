from pydantic import BaseModel, Field
from typing import List


class DefinitionMetadata(BaseModel):
    id: int = Field(description="the identifier of the definition in the database")
    dataset: str = Field(description="the dataset the definition is from")
    document_id: str = Field(description="the document id the definition is from")
    definendum_label: str = Field(description="the definendum label as it appears in the document")
    def_n: str = Field(description="the identifier of the definition in the document")
    #references: List[str] = Field(description="the references mentioned the definition.")
    frbr_work: str = Field(description="the frbr work uri of the definition")

class Definition(BaseModel):
    definition_text: str = Field(description="the full text of the definition")
    metadata: DefinitionMetadata

class DefinitionsList(BaseModel):
    relevant_definitions: List[Definition] = Field(description="a dict of relevant definitions")

class RelevantDefinitionsIDList(BaseModel):
    relevant_definitions: List[int] = Field(description="a list containing the ids of relevant definitions")

class GeneratedDefinition(BaseModel):
    generated_definition: str = Field(description="the generated definition")

class AnswerDefinition(BaseModel):
    most_relevant_definition: dict = Field(description="a containing the single, most relevant definition and its metadata")