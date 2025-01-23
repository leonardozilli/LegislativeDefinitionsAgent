from langchain_core.tools import tool
from langgraph.graph import START, StateGraph
from typing import TypedDict
from typing import Annotated, Literal, Sequence, Dict, Any
from typing_extensions import TypedDict
import uuid

from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langgraph.graph.message import add_messages
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.callbacks import dispatch_custom_event
from langchain_core.runnables import RunnableConfig
import logging
from langchain_core.messages import ToolMessage, trim_messages, filter_messages
from langgraph.types import Command
from langgraph.prebuilt import InjectedState
from langchain_core.tools.base import InjectedToolCallId
from langchain_core.tools import tool

import datetime


from . import llm, tools, utils
from .retriever import vector_store
from .settings import settings
from .schema.task_data import Task
from .schema.definition import DefinitionsList, Definition, RelevantDefinitionsIDList
from .schema.grader import DefinitionRelevance

from .schema.task_data import Task

from .llm import get_model
import LegalDefAgent.src.utils 


logger = logging.getLogger(__name__)


#vectorstore = vector_store.setup_vectorstore()
#retriever = vectorstore.as_retriever(search_kwargs={"k": 7})
retriever = vector_store.setup_retriever()
existdb_handler = utils.setup_existdb_handler()


async def generate_definition(config: RunnableConfig, definendum: str, question: str, relevant_definitions: list):
    """
    Generate a definition for a given term.

    Args:
        config (RunnableConfig): The configuration of the agent
        definendum (str): The term to define
        question (str): The question asked by the user
    
    Returns:
        str: The generated definition
    """

    model = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))

    examples = "\n".join([f"{definition}" for definition in relevant_definitions])
    await Task("Generating definition").start(data={"examples": examples}, config=config)

    prompt = PromptTemplate(
    template="""
        You are a legal expert specialized in legal definitions. Your job is to draft a legal definition for a specific term.
        Provide a definition for the term "{definendum}" to answer the user's question provided below following the style and formatting of the definitions provided as examples.
        User Question: {question}
        Example definitions: 
        \n{examples}\n

        ### IMPORTANT NOTES:
        * If you use explicit text from the examples, you should provide the source of the definition used.
        * Output the generated definition in the following format: {{ "generated_definition": "..." }}
        """,
        input_variables=["question", "definendum", "examples"]
    )

    chain = prompt | model
    response = chain.invoke({"question": question, "definendum": definendum, "examples": examples})
    await Task("Generating definition").finish(result="success", data={"done": "done"}, config=config)

    return {response.content}


def query_vectorstore(query: str):
    retrieved_definitions = retriever.invoke(query)

    return utils.docs_list_to_json_list(retrieved_definitions)


def semantic_filtering(config: RunnableConfig, retrieved_definitions: list[Dict], question: str) -> int:
    model = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))

    parser = JsonOutputParser(pydantic_object=RelevantDefinitionsIDList)
    prompt = PromptTemplate(
        template="""
        You are an AI legal expert tasked with assessing the relevance of legal definitions to a user's question. Your primary goal is to filter a list of provided definitions, keeping only those that are relevant to the user's query.
        You will be provided with a dictionary containing definitions that were automatically retrieved. Your task is to perform the following steps:

            1. Carefully read and understand the user's question.
            2. Review each definition provided in the dictionary.
            3. For each definition, determine its relevance to the user's question by considering:
                a. Whether the definition contains keywords from the question.
                b. Whether the semantic meaning of the definition relates to the question's topic.
            4. Keep relevant definitions and discard irrelevant ones.
            5. Format the relevant definitions following the formatting instructions provided.
        
        For each  the text of the definition contains keyword(s) or semantic meaning related to the user's question, keep it. Otherwise discard it.
        Output only the relevant definitions using the formatting instructions provided.

        ### VERY IMPORTANT NOTES:
            - Your final output must be valid, directly parsable JSON.
            - If no relevant definitions are found, output an empty array: "relevant_definitions": []

        Here are the formatting instructions: {format_instructions}

        Here are the retrieved definitions you need to analyze: {context}

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


def filter_documents_by_date(documents, date_filters):
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
                entry['date'].date() == target_date
                for entry in doc['timeline']
            )
        ]
    
    # If date range filter
    from_date, to_date = parsed_filters
    return [
        doc for doc in documents
        if any(
            from_date <= entry['date'].date() <= to_date
            for entry in doc['timeline']
        )
    ]


async def definition_search(state: Annotated[dict, InjectedState], tool_call_id: Annotated[str, InjectedToolCallId], config: RunnableConfig, question: str, definendum: str, legislation: str | None = None, date_filters: tuple | None = None) -> Command[Literal['answer', 'generate_definition']]:
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
    await Task("Query vector store").finish(result="success", data={"output": retrieved_definitions}, config=config)

    # Ask the llm to filter them
    await Task("Semantic filtering").start(data={"input": retrieved_definitions}, config=config)
    relevant_definitions_ids = semantic_filtering(config, retrieved_definitions, question)
    logger.info(f"Relevant definitions IDs: {relevant_definitions_ids}")
    relevant_definitions = [definition for definition in retrieved_definitions if definition['metadata']['id'] in relevant_definitions_ids]
    logger.info(f"Relevant definitions: {relevant_definitions}")
    await Task("Semantic filtering").finish(result="success", data={"output": relevant_definitions}, config=config)

    # If a legislation filter is provided, filter the definitions
    if legislation:
        await Task("Filter by legislation").start(data={"input": relevant_definitions}, config=config)
        relevant_definitions = filter_definitions_by_legislation(relevant_definitions, legislation)
        await Task("Filter by legislation").finish(result="success", data={"output": relevant_definitions}, config=config)

    if len(relevant_definitions) == 0:
        examples = [definition.get('definition_text') for definition in retrieved_definitions]
        generated_definition = await generate_definition(config, definendum, question, relevant_definitions=examples)
        return {"generated_definition": generated_definition}
        return Command(
            update={
                "relevant_definitions": relevant_definitions,
                "messages": [ToolMessage(f"I couldn't find an appropriate definition for '{definendum}'", tool_call_id=tool_call_id)],
            },
            goto="answer"
        )


    await Task("Retrieving Definition Timeline...").start(data={"input": relevant_definitions}, config=config)
    for definition in relevant_definitions:
        definition['timeline'] = get_definition_timeline(definition)
    await Task("Retrieving Definition Timeline...").finish(result="success", data={"output": relevant_definitions}, config=config)
    
    # If date filters are provided, filter the definitions
    if date_filters:
        await Task("Filter by date").start(data={"input": relevant_definitions}, config=config)
        relevant_definitions = filter_documents_by_date(relevant_definitions, legislation)
        await Task("Filter by date").finish(result="success", data={"output": relevant_definitions}, config=config)

    #we dont' neeed all the metadata for generation
    if len(relevant_definitions) == 0:
        examples = [definition.get('definition_text') for definition in retrieved_definitions]
        generated_definition = await generate_definition(config, definendum, question, relevant_definitions=examples)
        return {"generated_definition": generated_definition}
        return Command(
            update={
                "relevant_definitions": relevant_definitions,
                "messages": [ToolMessage(f"I couldn't find an appropriate definition for '{definendum}'", tool_call_id=tool_call_id)],
            },
            goto="answer"
        )

    await Task("Retrieving Eurovocs...").start(data={"input": relevant_definitions}, config=config)
    for definition in relevant_definitions:
        definition['eurovocs'] = get_definition_eurovocs(definition)
    await Task("Retrieving Eurovocs...").finish(result="success", data={"output": relevant_definitions}, config=config)
    
    for definition in relevant_definitions:
        definition['metadata']['celex_id'] = definition['metadata'].pop('document_id').split('.')[0]
        del definition['metadata']['definendum_label']
        del definition['metadata']['frbr_expression']
        del definition['metadata']['frbr_work']
        del definition['definition_text']
        #del definition['eurovocs'] # this can be removed if pick_definition is implemented

#    return Command(
        #update={
            #"relevant_definitions": relevant_definitions,
            #"definendum": definendum,
            #"question": question,
            #"messages": [ToolMessage("Successfully retrieved relevant definitions", tool_call_id=tool_call_id)],
        #},
        #goto="answer"
    #)
    return {"relevant_definitions": relevant_definitions}  


async def definition_search2(config: RunnableConfig, question: str, definendum: str, legislation: str | None = None, date_filters: tuple | None = None) -> str:
    """
    Searches and retrieves the most similar definitions to the given query in a vector DB.

    Args:
        question: The whole user's query.
        definendum: The term to be defined, as extracted from the user's query.
        legislation: The legislation to search in. Possible values: "EU", "IT", None
        date_filters: Date filters in the form of a tuple (from_date, to_date) to apply to the search. e.g. ("2021-01-01", "2021-12-31")
    """

    try:
        #await Task("Retrieve definition").start(data={"query": {'question': question, 'definendum': definendum, 'legislation': legislation, 'date_filters': date_filters}}, config=config)
        logger.info(f"Retrieving definition for query: 'question': {question}, 'definendum': {definendum}, 'legislation': {legislation}, 'date_filters': {date_filters}")


        # Retrieve definitions from the vector store
        #await Task("Query vector store").start(data={"input": definendum}, config=config)
        retrieved_definitions = query_vectorstore(definendum)
        logger.info(f"Retrieved definitions: {len(retrieved_definitions)}")
        #await Task("Query vector store").finish(result="success", data={"output": retrieved_definitions}, config=config)

        # Ask the llm to filter them
        #await Task("Semantic filtering").start(data={"input": retrieved_definitions}, config=config)
        relevant_definitions_ids = semantic_filtering(config, retrieved_definitions, question)
        logger.info(f"Relevant definitions IDs: {relevant_definitions_ids}")
        relevant_definitions = [definition for definition in retrieved_definitions if definition['metadata']['id'] in relevant_definitions_ids]
        logger.info(f"Relevant definitions: {len(relevant_definitions)}")
        #await Task("Semantic filtering").finish(result="success", data={"output": relevant_definitions}, config=config)

        # If a legislation filter is provided, filter the definitions
        if legislation:
            logger.info(f"Filtering by legislation: {legislation}")
            #await Task("Filter by legislation").start(data={"input": relevant_definitions}, config=config)
            relevant_definitions = filter_definitions_by_legislation(relevant_definitions, legislation)
            #await Task("Filter by legislation").finish(result="success", data={"output": relevant_definitions}, config=config)
            logger.info(f"Filtered definitions: {len(relevant_definitions)}")

        #await Task("Retrieving Definition Timeline...").start(data={"input": relevant_definitions}, config=config)
        for definition in relevant_definitions:
            definition['timeline'] = get_definition_timeline(definition)

        logger.info(f"Definition Timeline: {relevant_definitions}")
        
        # If date filters are provided, filter the definitions
        if date_filters:
            logger.info(f"Filtering by date: {date_filters}")
            #await Task("Filter by date").start(data={"input": relevant_definitions}, config=config)
            relevant_definitions = filter_documents_by_date(relevant_definitions, legislation)
            #await Task("Filter by date").finish(result="success", data={"output": relevant_definitions}, config=config)
            logger.info(f"Filtered definitions: {len(relevant_definitions)}")

        for definition in relevant_definitions:
            definition['eurovocs'] = get_definition_eurovocs(definition)

        if len(relevant_definitions) == 0:
                return Command(
        update={
            # update the state keys
            "user_info": user_info,
            # update the message history
            "messages": [ToolMessage("Successfully looked up user information", tool_call_id=tool_call_id)]
        }
    )
        
        for definition in relevant_definitions:
            del definition['metadata']['definendum_label']
            del definition['metadata']['frbr_expression']
            del definition['metadata']['frbr_work']
            del definition['definition_text']

        return {"relevant_definitions": relevant_definitions}    

    except Exception as e:

        return e