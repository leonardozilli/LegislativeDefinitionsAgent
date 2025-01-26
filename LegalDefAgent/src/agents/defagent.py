from typing import Literal

from langchain_core.runnables import RunnableConfig, RunnableLambda, RunnableSerializable
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage, trim_messages
from langchain_core.prompts import PromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.managed import RemainingSteps
from langgraph.prebuilt import ToolNode

from ..llm import get_model
from ..settings import settings
from ..utils import json_to_aimessage
from ..tools import definition_search


class AgentState(MessagesState, total=False):
    remaining_steps: RemainingSteps
    retrieved_definitions: str
    query: dict


def generate_definition(state: AgentState, config: RunnableConfig):
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

    prompt = PromptTemplate(
    template="""
        You are a legal expert specialized in legal definitions. Your job is to draft a legal definition for a specific term.
        Provide a definition for the term "{definendum}" to answer the user's question provided below. 
        Your generated definition has to follow the style, length and formatting of the definitions provided as examples.

        User Question: {question}

        Example definitions: 
        \n{examples}\n

        ### IMPORTANT NOTES:
        """,
        input_variables=["question", "definendum", "examples"],
    )

    chain = prompt | model
    response = chain.invoke({"question": state['query']['question'], "definendum": state['query']['definendum'], "examples": state['retrieved_definitions']})

    return {"messages": [json_to_aimessage(response)]}


def pick_definition(state: AgentState, config: RunnableConfig):
    """
    Pick a definition from the list of retrieved definitions.

    Args:
        state (AgentState): The current state
        config (RunnableConfig): The configuration of the agent

    Returns:
        dict: Updated state with the picked definition
    """
    model = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    prompt = PromptTemplate(
        template="""
        You are a legal expert specialized in legal definitions. Your job is to select the most relevant definition from a list of retrieved definitions.
        You will be provided with a set of legal definitions, along with associated metadata. Your goal is to choose the definition that best answers the user's question while considering the context provided by the legislation and keywords.
        
        Here is the user's question: {question}

        The term to be defined is: {definendum}

        Here are the retrieved definitions to choose from: {retrieved_definitions}

        To select the most relevant definition:

        1. Analyze the user's question to identify the key concept or term they are asking about.
        2. Review each legal definition and assess its relevance to the user's question.
        3. Examine the EuroVoc keywords associated with each definition. Definitions with keywords that align closely with the question's subject matter should be considered more relevant.
        4. If there are multiple definitions that could be relevant, choose the one from the EU legislation (EurLex).
        5. If there are multiple definitions, choose the one with the most recent date.
        6. Evaluate the specificity and comprehensiveness of each definition in relation to the user's question.

        Remember to consider all provided information carefully to ensure you select the most appropriate and relevant definition for the user's question.
        """,
        input_variables=["question", "definendum", "retrieved_definitions"],
    )

    chain = prompt | model
    response = chain.invoke({"question": state['query']['question'], "definendum": state['query']['definendum'], "retrieved_definitions": state['retrieved_definitions']})

    return {"messages": [response]}



instructions = f"""
    You are an AI legal assistant specializing in providing accurate definitions for legal terms and concepts. Your primary function is to answer user queries about legal definitions using only the information available through the tools at your disposal.
    The user will ask you questions about terms and concepts, and you will need to provide them with the most relevant definitions to their query.
    Your task is to answer the user's questions using ONLY the information provided by the tools at your disposal.

    To provide the most accurate and relevant definition, follow these steps:

    1. Analyze the user's query to extract key information:
        - Definendum: The exact term to be defined
        - Legislation filter: Specify "EU" for European Union, "IT" for Italy, or None if not specified
        - Date filters: Extract any date constraints in the format (from_date, to_date)

    2. Use the definition_search tool with the extracted information to retrieve relevant definitions.
        a. If the definition_search tool does not find any results, a generated definition will be provided instead.


    ### AVAILABLE TOOLS
    - **definition_search**
        Searches and retrieves the most similar definitions to the given query in a vector DB, filters them according to the filters provided and returns the most relevant ones.\n
        To use this tool, you need to extract the following information from the user's query:
        - **definendum**: This is the term to be defined. Examples:
            * "What is the definition of a contract?" -> "contract"
            * "What is a fishing net?" -> "fishing net"
            * "What's the definition of 'vessel'?" -> "vessel"
        - **legislation**: The legislation to restrict the search to. Possible values: "EU", "IT", None, where "EU" stands for European Union and "IT" stands for Italy. Examples:
            * "What is the definition of a contract in the EU?" -> "EU"
            * "What is the definition of a fishing net in Italy?" -> "IT"
            * "What is the definition of 'vessel'?" -> None
        - **date_filters**: A tuple of date filters to restrict the search to in the form of (from_date, to_date). Is None if there are no dates in the user's query. Examples:
            * "What was the definition of dog on the 8 of January 1999?" -> ("1999-01-08", "1999-01-08")
            * "What is the definition of a contract in 2015?" -> ("2015-01-01", "2015-12-31")
            * "What is the definition of a contract starting from 2015?" -> ("2015-01-01", None)
            * "What has been the definition of a contract up to 2015?" -> (None, "2015-12-31")
            * "What was the definition of 'bear' between 2010 and 2015?" -> ("2010-01-01", "2015-12-31")


    ### IMPORTANT NOTES
    Remember to use ONLY the information provided by the tools described. Do not rely on or include any external knowledge in your response.

    Now, please process the user's query.
    """


def wrap_model(model: BaseChatModel) -> RunnableSerializable[AgentState, AIMessage]:
    model = model.bind_tools(tools)
    preprocessor = RunnableLambda(
        lambda state: [SystemMessage(content=instructions)] + state["messages"],
        name="StateModifier",
    )
    return preprocessor | model


def acall_model(state: AgentState, config: RunnableConfig) -> AgentState:
    m = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    model_runnable = wrap_model(m)
    response = model_runnable.invoke(state, config)

    if state["remaining_steps"] < 2 and response.tool_calls:
        return {
            "messages": [
                AIMessage(
                    id=response.id,
                    content="Sorry, need more steps to process this request.",
                )
            ]
        }
    
    return {"messages": [response]}


def state_cleanup(state: AgentState) -> AgentState:
    state['messages'] = trim_messages(state['messages'], token_counter=len, strategy="last", max_tokens=1)
    state['messages'] = state['messages'][-1:]

    return {"messages": ""}


tools = [definition_search]

agent = StateGraph(AgentState)
agent.add_node("supervisor", acall_model)
agent.add_node("tools", ToolNode(tools))
agent.add_node("state_cleanup", state_cleanup)
agent.add_node("generate_definition", generate_definition)
agent.add_node("pick_definition", pick_definition)

# After "superviros", if there are tool calls, run "tools". Otherwise END.
def pending_tool_calls(state: AgentState) -> Literal["tools", "done"]:
    last_message = state["messages"][-1]
    if not isinstance(last_message, AIMessage):
        raise TypeError(f"Expected AIMessage, got {type(last_message)}")
    if last_message.tool_calls:
        return "tools"
    return "done"

def router(state: AgentState) -> str:
    retrieved_definitions = state['retrieved_definitions']

    if len(retrieved_definitions) > 0:
        return "pick_definition"
    else:
        return "generate_definition"


#agent.add_node("router", router)


agent.set_entry_point("supervisor")
agent.add_conditional_edges("supervisor", pending_tool_calls, {"tools": "tools", "done": END})
# Always run "supervisor" after "tools"
#agent.add_edge("tools", "supervisor")
#agent.add_conditional_edges("tools", router, {"pick_definition": "pick_definition", "generate_definition": "generate_definition"})
agent.add_edge("pick_definition", "state_cleanup")
agent.add_edge("generate_definition", "state_cleanup")
agent.add_edge("state_cleanup", END)
#agent.add_edge("generate_definition", END)


definitions_agent = agent.compile(checkpointer=MemorySaver())