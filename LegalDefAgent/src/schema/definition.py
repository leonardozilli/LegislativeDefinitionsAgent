from typing import List

from pydantic import BaseModel, Field


class DefinitionMetadata(BaseModel):
    id: int = Field(
        description="the identifier of the definition in the database")
    dataset: str = Field(description="the dataset the definition is from")
    document_id: str = Field(
        description="the document id the definition is from")
    definendum_label: str = Field(
        description="the definendum label as it appears in the document")
    def_n: str = Field(
        description="the identifier of the definition in the document")
    frbr_work: str = Field(description="the frbr work uri of the definition")


class Definition(BaseModel):
    definition_text: str = Field(description="the full text of the definition")
    metadata: DefinitionMetadata


class DefinitionsList(BaseModel):
    relevant_definitions: List[Definition] = Field(
        description="a dict of relevant definitions")


class RelevantDefinitionsIDList(BaseModel):
    relevant_definitions: List[int] = Field(
        description="a list containing the ids of relevant definitions")


class GeneratedDefinition(BaseModel):
    generated_definition: str = Field(description="the generated definition")


class PickedDefinition(BaseModel):
    picked_definition_id: int | None = Field(
        description="the id of the definition chosen to answer the query. None if no definition was chosen")
    timeline_id: int | None = Field(
        description="the number next to the timeline entry the picked definition is from. None if no definition was chosen")
