from typing import Literal

from langchain_core.runnables import RunnableConfig, RunnableLambda, RunnableSerializable
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage, trim_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.managed import RemainingSteps
from langgraph.prebuilt import ToolNode

from ..llm import get_model
from ..settings import settings
from ..tools import definition_search


class AgentState(MessagesState, total=False):
    remaining_steps: RemainingSteps


instructions = f"""
    You are an AI legal assistant specializing in providing accurate definitions for legal terms and concepts. Your primary function is to answer user queries about legal definitions using only the information available through the tools at your disposal.
    The user will ask you questions about terms and concepts, and you will need to provide them with the most relevant definitions to their query.
    Your task is to answer the user's questions using ONLY the information provided by the tools at your disposal.

    1. Analyze the user's query to extract key information
    2. Use the definition_search tool with the extracted information to retrieve the most relevant definition or a generated definition if no results are found
    3. Provide the user with the text and metadata of the definitions the tool found in response to their query


    ### AVAILABLE TOOLS
    - **definition_search**
        Searches and retrieves the most similar definitions to the given query in a vector DB, filters them according to the filters provided and returns the most relevant one.\n
        If it does not find any results, it will return a generated definition instead.
        To use this tool, you need to extract the following information from the user's query:
        - **definendum**: This is the term to be defined. Examples:
            * "What is the definition of a contract?" -> "contract"
            * "What is a fishing net?" -> "fishing net"
            * "What's the definition of 'vessel'?" -> "vessel"
        - **legislation**: The legislation to restrict the search to. Possible values: None, "EU", "IT", where "EU" stands for European Union and "IT" stands for Italy. Examples:
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
    - Remember to use ONLY the information provided by the tools described. Do not rely on or include any external knowledge in your response.
    - If the definition_search tool does not return any results, you should provide a generated definition instead.
    - If the definitions retrieved by the tool are not in English, you should not translate them, but provide the original text as returned by the tool.
    - If the answer definition is generated, you should disclose this to the user in your response like: "I couldn't find a definition for [term], so here's a generated definition instead: \n\n > [definition in markdown quote]".
    - Instead, if the answer definitions are retrieved, present them by formatting the definition in markdown quote format (>) for better readability, as in the following example: "I found the following definition(s) for [term with eventual filters]: > [definition] \n\n This definition...[metadata]".
    """


def wrap_model(model: BaseChatModel) -> RunnableSerializable[AgentState, AIMessage]:
    model = model.bind_tools(tools)
    preprocessor = RunnableLambda(
        lambda state: [SystemMessage(
            content=instructions)] + state["messages"],
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
    state['messages'] = trim_messages(
        state['messages'], token_counter=len, strategy="last", max_tokens=1)
    state['messages'] = state['messages'][-1:]

    return {"messages": ""}


tools = [definition_search]

agent = StateGraph(AgentState)
agent.add_node("supervisor", acall_model)
agent.add_node("tools", ToolNode(tools))
agent.add_node("state_cleanup", state_cleanup)


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


agent.set_entry_point("supervisor")
agent.add_conditional_edges("supervisor", pending_tool_calls, {
                            "tools": "tools", "done": "state_cleanup"})
agent.add_edge("tools", "supervisor")
agent.add_edge("state_cleanup", END)


definitions_agent = agent.compile(checkpointer=MemorySaver())
