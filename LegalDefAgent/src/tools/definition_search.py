from typing import Annotated, Dict, Optional, Tuple
import logging
import datetime

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import ToolMessage, trim_messages, filter_messages
from langchain_core.tools.base import InjectedToolCallId
from langgraph.types import Command
from langgraph.prebuilt import InjectedState

from .. import utils
from ..existdb import existdb_handler
from ..vectorstore import retriever as retriever_
from ..settings import settings
from ..schema.task_data import Task
from ..schema.definition import RelevantDefinitionsIDList, AnswerDefinition, GeneratedDefinition
from ..llm import get_model
from ..utils import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Initialize retriever with constant
RETRIEVER = retriever_.setup_retriever(k=10)


def query_vectorstore(query: str) -> list[Dict]:
    """Query vector store for relevant definitions."""
    retrieved_definitions = RETRIEVER.invoke(query)
    return utils.docs_list_to_json_list(retrieved_definitions)


def get_relevant_definitions_id(config: RunnableConfig, retrieved_definitions: list[Dict], question: str) -> int:
    """Get IDs of relevant definitions based on the question."""
    model = get_model(config["configurable"].get(
        "model", settings.DEFAULT_MODEL))
    parser = JsonOutputParser(pydantic_object=RelevantDefinitionsIDList)

    prompt = PromptTemplate(
        template="""
        You are an AI legal expert tasked with assessing the relevance of multilingual legal definitions to a user's question. Your primary goal is to filter a list of provided definitions, finding the id of those that are relevant to the user's query.
        You will be provided with a dictionary containing definitions that were automatically retrieved. Your task is to check the relevance of each definition to the user's question and provide the id of the relevant definitions.
        If a definition contains keywords from the user's question or if its semantic meaning relates to the question's topic, even if in another language, it should be considered relevant.
        Format the relevant definitions following the formatting instructions provided.
        
        ### VERY IMPORTANT NOTES:
            - The retrieved definition dictionary can contain definitions in English and Italian. You must consider both languages when filtering the definitions.
            - Your final output must be valid, directly parsable JSON.
            - If no relevant definitions are found, output an empty array: "relevant_definitions": []

        Here are the formatting instructions: {format_instructions}

        Here are the retrieved definitions you need to analyze: \n{retrieved_definitions}\n

        Here is the question asked by the user: {question}

        Please proceed with your analysis and provide the filtered list of relevant definitions.
        """,
        input_variables=["retrieved_definitions", "question"],
        partial_variables={
            "format_instructions": parser.get_format_instructions()}
    )

    chain = prompt | model | parser
    response = chain.invoke(
        {"retrieved_definitions": retrieved_definitions, "question": question})
    return response['relevant_definitions']


def filter_definitions_by_legislation(definitions: list[Dict], legislation: str) -> list[Dict]:
    """Filter definitions by legislation type."""
    legislation_datasets = {
        'EU': ['EurLex'],
        'IT': ['Normattiva', 'PDL'],
    }
    return [
        definition for definition in definitions
        if definition['metadata']['dataset'] in legislation_datasets[legislation]
    ]


def get_definition_timeline(definition: Dict) -> Optional[list[Dict]]:
    """Get timeline for a definition."""
    if cons := existdb_handler.find_consolidated(definition['metadata']):
        earliest_entries = {}
        for item in cons:
            definition_text = item['definition'].strip()
            if definition_text not in earliest_entries or item['date'] < earliest_entries[definition_text]['date']:
                earliest_entries[definition_text] = item

        ordered_definitions = sorted(
            earliest_entries.values(), key=lambda x: x['date'])
        for entry in ordered_definitions:
            entry['date'] = entry['date'].strftime('%Y-%m-%d')
        return ordered_definitions

    if static := existdb_handler.extract_definition_from_exist(definition['metadata']):
        for entry in static:
            entry['date'] = entry['date'].strftime('%Y-%m-%d')
        return static

    return None


def filter_definitions_by_date(documents: list[Dict], date_filters: Tuple[str, str]) -> list[Dict]:
    """Filter legal definitions based on their timeline dates and given date filters."""
    parsed_filters = utils.parse_date_filters(date_filters)

    if isinstance(parsed_filters, datetime.date):
        target_date = parsed_filters
        return [
            doc for doc in documents
            if any(utils.parse_date(entry['date']) == target_date for entry in doc['timeline'])
        ]

    from_date, to_date = parsed_filters
    return [
        doc for doc in documents
        if any(
            from_date <= utils.parse_date(entry['date']) <= to_date
            for entry in doc['timeline']
        )
    ]


def generate_definition(config: RunnableConfig, question: str, definendum: str, example_definitions: list[str]) -> Dict:
    """Generate a definition for a given term."""
    model = get_model(config["configurable"].get(
        "model", settings.DEFAULT_MODEL))
    parser = JsonOutputParser(pydantic_object=GeneratedDefinition)

    prompt = PromptTemplate(
        template="""
        You are a legal expert specialized in drafting legal definitions. Your job is to draft a legal definition for a specific term.
        Provide a definition for the term "{definendum}" to answer the user's question provided below. 
        Your definition should be clear, concise, and accurate. Use the examples provided to guide you in creating a definition that is relevant to the user's query.
        Your generated definition has to follow the style, length and formatting of the definitions provided as examples.

        User Question: {question}

        Output Format Instructions: {format_instructions}

        Example definitions: 
        \n- {examples}\n

        ### IMPORTANT NOTES:
        """,

        input_variables=["question", "definendum", "examples"],
        partial_variables={
            "format_instructions": parser.get_format_instructions()}
    )

    chain = prompt | model | parser
    response = chain.invoke({
        "question": question,
        "definendum": definendum,
        "examples": '\n- '.join(set(example_definitions))
    })

    response['sources'] = example_definitions
    return response


def pick_definition(
    config: RunnableConfig,
    legislation: str,
    relevant_definitions: list[Dict],
    question: str,
    definendum: str
) -> Dict:
    """Pick most relevant definition from the list."""
    model = get_model(config["configurable"].get(
        "model", settings.DEFAULT_MODEL))
    parser = JsonOutputParser(pydantic_object=AnswerDefinition)
    prompt = PromptTemplate(
        template="""
        You are a legal expert specialized in legal definitions. Your job is to find the correct definition from a list of retrieved definitions.
        You will be provided with a bunch of legal definitions, along with associated metadata. 
        Your goal is to find a definition that answers the user's question.
        Select a definition that provides an explanation of the EXACT term the user is asking about. If none of the retrieved definitions are fit to answer the user's question, you should not pick any definition.
        
        Here are the formatting instructions: {format_instructions}

        Here is the user's question: {question}

        The term to be defined is: {definendum}

        Here are the retrieved definitions to choose from: \n{retrieved_definitions}\n

        ### IMPORTANT NOTES
        - Your final output must be valid, directly parsable JSON.
        - ONLY choose a definition that provides an explanation of the EXACT term the user is asking about. Do not choose a definition that explains a similar or related term.
        - If there are multiple definitions that could be relevant, choose the one from the EU legislation (EurLex).
        - ONLY use the information provided in the dictionary. Do not rely on or include any external knowledge in your response.
        """,
        input_variables=["question", "definendum", "retrieved_definitions"],
        partial_variables={
            "format_instructions": parser.get_format_instructions()}
    )

    chain = prompt | model | parser
    response = chain.invoke({
        "question": question,
        "definendum": definendum,
        "retrieved_definitions": utils.format_definitions_dict(relevant_definitions)
    })
    return response


async def definition_search(
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    config: RunnableConfig,
    question: str,
    definendum: str,
    legislation: Optional[str] = None,
    date_filters: Optional[str] = None,
) -> Dict:
    """
    Searches and retrieves the most similar definitions to the given query in a vector DB.

    Args:
        question: The whole user's query.
        definendum: The term to be defined, as extracted from the user's query.
        legislation: The legislation to search in. Possible values: "EU", "IT", None
        date_filters: Date filters in the form of a string "from_date - to_date" to apply to the search. e.g. "2021-01-01 - 2021-12-31"
    """
    logger.info(f"Searching for definitions: {definendum=}, {question=}, {legislation=}, {date_filters=}")
    state['messages'] = filter_messages(
        state['messages'], exclude_types="tool")

    # Start search process
    await Task("Retrieve definition").start(
        data={"query": {
            'question': question,
            'definendum': definendum,
            'legislation': legislation,
            'date_filters': date_filters
        }},
        config=config
    )

    # Query vector store
    await Task("Query vector store").start(data={"input": definendum}, config=config)
    relevant_definitions = query_vectorstore(question)
    logger.info(f"Retrieved {len(relevant_definitions)} definitions from vectorstore")
    await Task("Query vector store").finish(
        result="success",
        data={"retrieved_definitions": relevant_definitions},
        config=config
    )

    # Apply legislation filter if specified
    if legislation:
        logger.info(f"Filtering by legislation: {legislation}")
        await Task("Filter by legislation").start(data={"input": relevant_definitions}, config=config)
        relevant_definitions = filter_definitions_by_legislation(
            relevant_definitions, legislation)
        await Task("Filter by legislation").finish(
            result="success",
            data={"output": relevant_definitions},
            config=config
        )

    if not relevant_definitions:
        logger.info("No relevant definitions found. Generating definition...")
        examples = [definition.get('definition_text')
                    for definition in relevant_definitions]
        await Task("Retrieve definition").finish(result="success", data={"output": ""}, config=config)
        return generate_definition(config, question, definendum, examples)

    # Process timelines
    await Task("Retrieving Definition Timeline...").start(data={"input": relevant_definitions}, config=config)
    definitions_with_timeline = []

    for definition in relevant_definitions:
        if timeline := get_definition_timeline(definition):
            definition['timeline'] = timeline
            definitions_with_timeline.append(definition)

    await Task("Retrieving Definition Timeline...").finish(
        result="success",
        data={"output": definitions_with_timeline},
        config=config
    )

    # Apply date filters if specified
    if date_filters:
        date_filters = utils.parse_date_string(date_filters)
        logger.info(f"Filtering by date: {date_filters}")
        await Task("Filter by date").start(data={"input": definitions_with_timeline}, config=config)
        definitions_with_timeline = filter_definitions_by_date(
            definitions_with_timeline, date_filters)
        await Task("Filter by date").finish(
            result="success",
            data={"output": definitions_with_timeline},
            config=config
        )

    if not definitions_with_timeline:
        logger.info(
            "No definitions with timeline found. Generating definition...")
        examples = [definition.get('definition_text')
                    for definition in relevant_definitions]
        await Task("Retrieve definition").finish(result="success", data={"output": ""}, config=config)
        return generate_definition(config, question, definendum, examples)

    # Add keywords
    logger.info("Retrieving Eurovocs...")
    for definition in definitions_with_timeline:
        definition['keywords'] = existdb_handler.get_work_eurovocs(
            definition['metadata'])

    # Pick final definition
    picked_definition = pick_definition(
        config,
        legislation,
        definitions_with_timeline,
        question,
        definendum
    )
    logger.info(f"Picked definition: {picked_definition}")

    if picked_definition["chosen_definition_id"] is None:
        examples = [definition.get('definition_text')
                    for definition in definitions_with_timeline]
        return {"generated_definition": generate_definition(config, question, definendum, examples)}

    answer_def = [
        definition for definition in definitions_with_timeline
        if definition['metadata']['id'] == picked_definition["chosen_definition_id"]
    ]

    return {
        'retrieved_definition': utils.format_answer_definition(
            answer_def,
            picked_definition["timeline_id"]
        )
    }
