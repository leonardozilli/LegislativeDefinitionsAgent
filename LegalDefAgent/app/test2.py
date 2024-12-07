from datetime import datetime
from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda, RunnableSerializable
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, MessagesState, StateGraph

from typing import Annotated, Literal, Sequence, List, Any, Dict

from langchain_core.messages import BaseMessage, AIMessage, SystemMessage, HumanMessage
from langgraph.graph.message import add_messages
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.callbacks.base import BaseCallbackHandler
from langgraph.checkpoint.memory import MemorySaver

from LegalDefAgent.src import tools as modelTools
from LegalDefAgent.src import models
from LegalDefAgent.src import schema
from LegalDefAgent.src.agent import LegalDefAgent
from LegalDefAgent.src.retriever import vector_store


class AgentState(MessagesState, total=False):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    question: str
    #relevant_defs: str
    #definendum: str


model = models.groq
vectorstore = vector_store.setup_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 7})
retriever_tool = modelTools.create_vector_search_tool(retriever)
tools = [retriever_tool]


#change to tools? maybe routing to a common chat?
def LegalAgent(state) -> Literal["DefAgent",]:
    """
    Invokes the master agent that invokes the task-related agents
    Args:
        state (messages): The current state

    Returns:
        str: The name of the task-agent
    """
    
    prompt = PromptTemplate(
        template="""
            You are a legal drafting assistant.
            You have at your disposal a number of tools to aid a user in the legal drafting process.
            Given a user's question, your job is to find the best tool to use to answer the question.
            The tools at your disposal are:
            * "definition": a tool that retrieves the definition of a term.

            E.g.: 
            * "What is the definition of a contract?" -> "definition"
            * "What is a fishing net?" -> "definition"
            * "house" -> "definition"

            VERY IMPORTANT NOTES:
            * You should output only the single definendum, without any additional text.
            * If you don't find the right tool, simply answer with "I can't help you with that yet!"

            Here is the user question: {question}
           """,
        input_variables=["question"],
    )

    chain = prompt | model

    input = state["messages"][-1].content

    response = chain.invoke({"question": input})

    match response.content.lower():
        case "definition":
            return "DefAgent"
        case _:
            return END


def DefAgent(state):
    """
    Invokes the agent model to generate a response based on the current state. Given
    the question, it will decide to retrieve using the retriever tool, or simply end.

    Args:
        state (messages): The current state

    Returns:
        dict: The updated state with the agent response appended to messages
    """
    print("---DEFINITION AGENT---")

    # Prompt
    prompt = PromptTemplate(
        template="""
            You are a legal drafting assistant.
            Your job is to extract the definendum, that is the term that is to be defined, out of a sentence.
            E.g.: 
            * "What is the definition of a contract?" -> "contract"
            * "What is a fishing net?" -> "fishing net"
            * "What's the definition of "vessel"" -> "vessel"

            VERY IMPORTANT NOTES:
            * You should output only the single definendum, without any additional text.
            * If you don't find a definendum in the sentence, you should output: "None"

            Input sentence: {question}
           """,
        input_variables=["question"],
    )

    chain = prompt | model

    input = state["messages"][-1].content

    response = chain.invoke({"question": input})

    return {"messages": [response], "question": input, "definendum": response.content}


def retrieveAndFilterDefs(state) -> Literal["generate"]:
    """
    Filters the retrieved definitions based on their relevance to the question.

    Args:
        state (messages): The current state

    Returns:
        List[dict]: A list of dictionaries containing the relevant definitions and their metadata.
    """

    print("---FILTER DEFINITIONS---")

    # Set up a parser + inject instructions into the prompt template.
    parser = PydanticOutputParser(pydantic_object=schema.DefinitionsList)

    # Prompt
    prompt = PromptTemplate(
        template="""
        You are a legal expert assessing the relevance of legal definitions to a user question. \n 
        Your job is to filter the list of definitions provided to you keeping only the relevant ones. \n
        If the text of the definition contains keyword(s) or semantic meaning related to the user's question, keep it. Otherwise discard it.\n
        Output only the relevant definitions using the formatting instructions provided. \n

        VERY IMPORTANT NOTES:
        * You should output only the formatted relevant definitions, without any additional text.
        * If none of the definitions are relevant, you should an empty list.

        Here are the formatting instructions: {format_instructions} \n
        Here are the retrieved definitions, one for each line: \n {context} \n
        Here is the user's question: {question} \n

        """,
        input_variables=["context", "question"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    chain = prompt | model

    question = state["question"]
    retrieved_definitions = retriever.invoke(question) # QUERY or DEFINENDUM???

    response = chain.invoke({"question": question, "context": retrieved_definitions}, config={})

    return {"messages": [response], "relevant_defs": response.content}


def answer(state):
    """
    Answer user with a definition.

    Args:
        state (messages): The current state

    Returns:
        str: A legal definition
    """
    print("---ANSWER---")

    prompt = PromptTemplate(
        template="""
        You are a legal expert and your job is to generate legal definitions. \n
        Use the retrieved definitions as context to answer the user's question. \n
        If you don't know the answer, just say that you don't know. \n
        Keep the answer concise and straight to the point, giving only the definition.
 
        VERY IMPORTANT NOTES:
        * You should output only the string with the definition, without any additional text.

        User Question: {question} \n
        Context: {context} \n
        """,
        input_variables=["context", "question"]
    )

    chain = prompt | model

    response = chain.invoke({"context": state['relevant_defs'], "question": state['question']}, config={})

    return {"messages": [response]}


instructions = f"""
    You are a helpful legal assistant named ROB with the ability to find the right definition for a given term.

    NOTE: THE USER CAN'T SEE THE TOOL RESPONSE.

    A few things to remember:
    - Use the retrieve tool passing it the string of the term you want to find a definition of.
    - ALWAYS answer in French!!!
    """


# Define the graph

agent = StateGraph(AgentState)
#agent.add_node("model", acall_model)
#agent.add_node("model", acall_model)
#agent.add_node("LegalAgent", LegalAgent)
agent.add_node("DefAgent", DefAgent)
#agent.add_node("retrieve", ToolNode([retriever_tool]))  # retrieval
agent.add_node("retrieveAndFilterDefs", retrieveAndFilterDefs)  # retrieval
agent.add_node("answer", answer)  # Generating a response after we know the documents are relevant

#agent.set_entry_point("agent")
#agent.add_conditional_edges(
            #"agent",
            #tools_condition,
            #{
                #"tools": "retrieve",
                #END: END,
            #},
# )
# agent.add_edge("retrieve", "filter")
# agent.add_edge("filter", "answer")
agent.add_conditional_edges(START,
                            LegalAgent,
                            {
                                "DefAgent": "DefAgent",
                                END: END
                            })
agent.add_edge("DefAgent", 'retrieveAndFilterDefs')
# agent.add_edge("retrieve", END)
agent.add_edge("retrieveAndFilterDefs", "answer")
agent.add_edge("answer", END)



defagent_graph = agent.compile(checkpointer=MemorySaver())


from typing import Any, List, Union, Optional

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from pydantic import BaseModel, Field
from langserve import add_routes

import sys
sys.path.insert(1, '../LegalDefAgent')


app = FastAPI(
    title="LegalDefAgent",
    version="1.0",
    description=""
)

@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")

class QueryRequest(BaseModel):
    query: str

class Input(BaseModel):
    input: str

class Output(BaseModel):
    output: Any

def inp(question: str) -> dict:
    return {
            "question": question,
            "messages": [
                SystemMessage(content=''),
                HumanMessage(content=question)
            ]
        }

def out(state: dict) -> str:
    return state["answer"]['response']

class InputChat(BaseModel):
    """Input for the chat endpoint."""

    messages: List[Union[HumanMessage, AIMessage, SystemMessage]] = Field(
        ...,
        description="The chat messages representing the current conversation.",
    )

add_routes(
    app,
    RunnableLambda(inp) | defagent_graph | RunnableLambda(out),
    path='/defagent',
)

add_routes(
    app,
    #chain.with_types(input_type=InputChat),
    RunnableLambda(inp) | defagent_graph.with_types(input_type=dict,output_type=str),
    #enable_feedback_endpoint=True,
    #enable_public_trace_link_endpoint=True,
    playground_type="chat",
    path='/defagent_chat',
)



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000,)