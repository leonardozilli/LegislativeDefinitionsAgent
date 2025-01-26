from typing import Annotated, Literal, Sequence, Dict, Any
import logging
import datetime

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import ToolMessage, trim_messages, filter_messages
from langgraph.types import Command
from langgraph.prebuilt import InjectedState
from langchain_core.tools.base import InjectedToolCallId

from .. import utils
from ..vectorstore import retriever as retriever_
from ..settings import settings
from ..schema.task_data import Task
from ..schema.definition import DefinitionsList, Definition, RelevantDefinitionsIDList
from ..llm import get_model


logger = logging.getLogger(__name__)


retriever = retriever_.setup_retriever(k=10)
existdb_handler = utils.setup_existdb_handler()


def query_vectorstore(query: str):
    retrieved_definitions = retriever.invoke(query)

    return utils.docs_list_to_json_list(retrieved_definitions)


def get_relevant_definitions_id(config: RunnableConfig, retrieved_definitions: list[Dict], question: str) -> int:
    model = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))

    parser = JsonOutputParser(pydantic_object=RelevantDefinitionsIDList)
    prompt = PromptTemplate(
        template="""
        You are an AI legal expert tasked with assessing the relevance of multilingul legal definitions to a user's question. Your primary goal is to filter a list of provided definitions, keeping only those that are relevant to the user's query.
        You will be provided with a dictionary containing definitions that were automatically retrieved. Your task is to perform the following steps:

            1. Carefully read and understand the user's question.
            2. Review each definition provided in the dictionary.
            3. For each definition, determine its relevance to the user's question by considering:
                a. Whether the definition contains keywords from the question.
                b. Whether the semantic meaning of the definition relates to the question's topic, even if in another language from the user's question.
            4. Keep relevant definitions and discard irrelevant ones.
            5. Format the relevant definitions following the formatting instructions provided.
        
        ### VERY IMPORTANT NOTES:
            - The retrieved definition dictionary can contain definitions in English and Italian. You must consider both languages when filtering the definitions.
            - Your final output must be valid, directly parsable JSON.
            - If no relevant definitions are found, output an empty array: "relevant_definitions": []

        Here are the formatting instructions: {format_instructions}

        Here are the retrieved definitions you need to analyze: \n{context}\n

        Here is the question asked by the user: {question}

        Please proceed with your analysis and provide the filtered list of relevant definitions.
        """,
        input_variables=["context", "question"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    chain = prompt | model | parser
    response = chain.invoke({"context": retrieved_definitions, "question": question})

    return response['relevant_definitions']


def filter_definitions_by_legislation(definitions, legislation_filter):
    legislation_dataset_map = {
        'EU': ['EurLex'],
        'IT': ['Normattiva', 'PDL'],
    }
    
    return [definition for definition in definitions if definition['metadata']['dataset'] in legislation_dataset_map[legislation_filter]]


def get_definition_timeline(definition):
    logger.info(f"Getting definition timeline for definition: {definition['metadata']['id']}")
    cons = existdb_handler.find_consolidated(definition['metadata'])
    if cons:
        earliest_entries = {}
        for item in cons:
            definition_text = item['definition'].strip()
            current_entry = earliest_entries.get(definition_text)
            
            if current_entry is None or item['date'] < current_entry['date']:
                earliest_entries[definition_text] = item
                
        ordered_definitions = sorted(earliest_entries.values(), key=lambda x: x['date'])
        for entry in ordered_definitions:
            entry['date'] = entry['date'].strftime('%Y-%m-%d')

        return ordered_definitions

    static = existdb_handler.extract_definition_from_exist(definition['metadata'])
    if static:
        for entry in static:
            entry['date'] = entry['date'].strftime('%Y-%m-%d')
        return static
    return None


def get_definition_eurovocs(definition):
    logger.info(f"Getting eurovocs for definition: {definition['metadata']['id']}")
    eurovocs = existdb_handler.get_work_eurovocs(definition['metadata'])

    return eurovocs


def filter_definitions_by_date(documents, date_filters):
    """
    Filter legal definitions based on their timeline dates and given date filters.
    
    Args:
        documents: List of document dictionaries containing timeline information
        date_filters: Tuple of (from_date, to_date) as strings
        
    Returns:
        List of filtered documents that fall within the date range
    """
    parsed_filters = utils.parse_date_filters(date_filters)
    
    # If single date filter
    if isinstance(parsed_filters, datetime.date):
        target_date = parsed_filters
        return [
            doc for doc in documents
            if any(
                utils.parse_date(entry['date']) == target_date
                for entry in doc['timeline']
            )
        ]
    
    # If date range filter
    from_date, to_date = parsed_filters
    return [
        doc for doc in documents
        if any(
            from_date <= utils.parse_date(entry['date']) <= to_date
            for entry in doc['timeline']
        )
    ]


async def definition_search(state: Annotated[dict, InjectedState], tool_call_id: Annotated[str, InjectedToolCallId], config: RunnableConfig, question: str, definendum: str, legislation: str | None = None, date_filters: tuple | None = None) -> Command[Literal['pick_definition', 'generate_definition']]:
    """
    Searches and retrieves the most similar definitions to the given query in a vector DB.

    Args:
        question: The whole user's query.
        definendum: The term to be defined, as extracted from the user's query.
        legislation: The legislation to search in. Possible values: "EU", "IT", None
        date_filters: Date filters in the form of a tuple (from_date, to_date) to apply to the search. e.g. ("2021-01-01", "2021-12-31")
    """

    # clear state
    state['messages'] = filter_messages(state['messages'], exclude_types="tool")

    await Task("Retrieve definition").start(data={"query": {'question': question, 'definendum': definendum, 'legislation': legislation, 'date_filters': date_filters}}, config=config)

    # Retrieve definitions from the vector store
    await Task("Query vector store").start(data={"input": definendum}, config=config)
    retrieved_definitions = query_vectorstore(definendum)
    logger.info(f"Retrieved definitions: {len(retrieved_definitions)}")
    await Task("Query vector store").finish(result="success", data={"retrieved_definitions": retrieved_definitions}, config=config)

    # Ask the llm to filter them
    stripped_definitions = [{'id': x['metadata']['id'], 'definition': x['definition_text']} for x in retrieved_definitions]
    await Task("Semantic filtering").start(data={"retrieved_definitions": stripped_definitions}, config=config)
    relevant_definitions_ids = get_relevant_definitions_id(config, retrieved_definitions, question)
    logger.info(f"Relevant definitions IDs: {relevant_definitions_ids}")
    relevant_definitions = [definition for definition in retrieved_definitions if definition['metadata']['id'] in relevant_definitions_ids]
    logger.info(f"Relevant definitions: {relevant_definitions}")
    await Task("Semantic filtering").finish(result="success", data={"relevant_definitions": relevant_definitions}, config=config)

    # If a legislation filter is provided, filter the definitions
    if legislation:
        await Task("Filter by legislation").start(data={"input": relevant_definitions}, config=config)
        relevant_definitions = filter_definitions_by_legislation(relevant_definitions, legislation)
        await Task("Filter by legislation").finish(result="success", data={"output": relevant_definitions}, config=config)

    if len(relevant_definitions) == 0:
        examples = [definition.get('definition_text') for definition in retrieved_definitions]
        #generated_definition = await generate_definition(config, definendum, question, relevant_definitions=examples)
        #return {"generated_definition": generated_definition}
        await Task("Retrieve definition").finish(result="success", data={"output": ""}, config=config)
        return Command(
            update={
                "query": {'question': question, 'definendum': definendum, 'legislation': legislation, 'date_filters': date_filters},
                "retrieved_definitions": '\n'.join(examples),
                "messages": [ToolMessage(f"I couldn't find an appropriate definition for '{definendum}'", tool_call_id=tool_call_id)],
            },
            goto="generate_definition"
        )


    await Task("Retrieving Definition Timeline...").start(data={"input": relevant_definitions}, config=config)
    for definition in relevant_definitions:
        definition['timeline'] = get_definition_timeline(definition)
    await Task("Retrieving Definition Timeline...").finish(result="success", data={"output": relevant_definitions}, config=config)
    
    # If date filters are provided, filter the definitions
    if date_filters:
        await Task("Filter by date").start(data={"input": relevant_definitions}, config=config)
        relevant_definitions = filter_definitions_by_date(relevant_definitions, date_filters)
        await Task("Filter by date").finish(result="success", data={"output": relevant_definitions}, config=config)

    if len(relevant_definitions) == 0:
        examples = [definition.get('definition_text') for definition in retrieved_definitions]
        await Task("Retrieve definition").finish(result="success", data={"output": ""}, config=config)

        #return {"generated_definition": generated_definition}
        return Command(
            update={
                "query": {'question': question, 'definendum': definendum, 'legislation': legislation, 'date_filters': date_filters},
                "retrieved_definitions": '\n'.join(examples),
                "messages": [ToolMessage(f"I couldn't find an appropriate definition for '{definendum}'", tool_call_id=tool_call_id)],
            },
            goto="generate_definition"
        )

    #await Task("Retrieving Eurovocs...").start(data={"input": relevant_definitions}, config=config)
    for definition in relevant_definitions:
        definition['eurovocs'] = get_definition_eurovocs(definition)
    
    for definition in relevant_definitions:
        definition['metadata']['celex_id'] = definition['metadata'].pop('document_id').split('.')[0]
        del definition['metadata']['definendum_label']
        del definition['metadata']['frbr_expression']
        del definition['metadata']['frbr_work']
        del definition['definition_text']
        #del definition['eurovocs'] # this can be removed if pick_definition is implemented

    #await Task("Retrieving Eurovocs...").finish(result="success", data={"output": relevant_definitions}, config=config)

    await Task("Retrieve definition").finish(result="success", data={"output": ""}, config=config)

    return Command(
        update={
            "query": {'question': question, 'definendum': definendum, 'legislation': legislation, 'date_filters': date_filters},
            "retrieved_definitions": relevant_definitions,
            "messages": [ToolMessage("Successfully retrieved relevant definitions", tool_call_id=tool_call_id)],
        },
        goto="pick_definition"
    )
    #return {"relevant_definitions": relevant_definitions}  