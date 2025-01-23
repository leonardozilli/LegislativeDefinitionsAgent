from typing import Literal

from langchain_core.runnables import RunnableConfig, RunnableLambda, RunnableSerializable
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.managed import RemainingSteps
from langgraph.prebuilt import ToolNode

from .llm import get_model
from .settings import settings
from .definition_search import definition_search


class AgentState(MessagesState, total=False):
    remaining_steps: RemainingSteps
#    relevant_definitions: list[str]
    #definendum: str
    #question: str


def answer(state: AgentState, config: RunnableConfig):
    """
    Generate a definition.
    Args:
        state (AgentState): The current state
    Returns:
        dict: Updated state with the generated definition
    """
    model = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    prompt = PromptTemplate(
        template="""
        You are a lawyer and your job is to provide legal definitions in response to user questions.

        User Question: {question}
        Retrieved_definitions: {retrieved_definitions}
        ### IMPORTANT NOTES
        - You can answer the user's questions ONLY using the information provided by the tools at your disposal. Do NOT use any pre-existing knowledge or external sources.
        - When answering the user, you should act as a professional legal assistant and provide the user with the details of each definition.\n In the list of retrieved definitions to use as the sole basis for your answer, you should illustrate the following details:
            * The definition itself
            * The source of the definition
            * The date of the definition
            * Eventual changes in the definition over time
        Use the following template to answer the user's questions, without adding any additional comment:
        I found the following definitions for "{definendum}":\n
            1. Definition: \n
                Source: \n
                Date: \n
            2. Definition: \n
        """,
        input_variables=["question", "definendum", "retrieved_definitions"]
    )

    definendum = state["definendum"]
    question = state["question"]
    retrieved_definitions = state["relevant_definitions"]
    chain = prompt | model
    response = chain.invoke({"question": question, "definendum": definendum, "retrieved_definitions": retrieved_definitions})

    return {"messages": [response]}


def generate_definition(state: AgentState, config: RunnableConfig, definendum: str, question: str):
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
        Provide a definition for the term "{definendum}" to answer the user's question provided below following the style and formatting of the definitions provided as examples.
        User Question: {question}
        Example definitions: 
        \n{examples}\n

        ### IMPORTANT NOTES:
        * You should answer with "I couldn't find a definition for "{definendum}", so here's a generated one:" followed by the generated definition.
        * If you use explicit text from the examples, you should provide the source of the definition used.
        """,
        input_variables=["question", "definendum", "examples"]
    )

    chain = prompt | model
    response = chain.invoke({"question": question, "definendum": definendum, "examples": state['relevant_definitions']})

    return {"messages": [response]}


tools = [definition_search]

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

    3. Analyze the retrieved definitions and present them in a clear, concise response to the user.


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

        The definition_search tool will return the most relevant definitions along with their metadata in a list with the following structure:

        [
            {{
                "definition": "The text of the definition",
                "source": "The source of the definition",
                "date": "The date of the definition"
            }},
            ...
        ]

        In case the tool does not find any results, a generated definition will be provided instead. 

        
    ### RESPONSE FORMAT
    Once you have the list of definitions, you should present them to the user
        1. Specify the search performed
        2. Consider any potential ambiguities or edge cases in the query.
        3. If the definition provided by the definition_search tool is generated, state that you couldn't find a definition and provide the generated one.
    
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


# Define the graph
agent = StateGraph(AgentState)
agent.add_node("supervisor", acall_model)
agent.add_node("tools", ToolNode(tools))
#agent.add_node("answer", answer)
#agent.add_node("generate_definition", generate_definition)

# After "superviros", if there are tool calls, run "tools". Otherwise END.
def pending_tool_calls(state: AgentState) -> Literal["tools", "done"]:
    last_message = state["messages"][-1]
    if not isinstance(last_message, AIMessage):
        raise TypeError(f"Expected AIMessage, got {type(last_message)}")
    if last_message.tool_calls:
        return "tools"
    return "done"


agent.set_entry_point("supervisor")
agent.add_conditional_edges("supervisor", pending_tool_calls, {"tools": "tools", "done": END})
# Always run "supervisor" after "tools"
agent.add_edge("tools", "supervisor")
#agent.add_edge("answer", END)
#agent.add_edge("generate_definition", END)


definitions_agent = agent.compile(checkpointer=MemorySaver())
