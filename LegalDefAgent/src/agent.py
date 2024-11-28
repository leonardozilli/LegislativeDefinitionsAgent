# %pip install -qU langchain-community langgraph langgraph-checkpoint-sqlite langchain-openai langchain-groq langchain_mistralai

from typing import Annotated, Literal, Sequence, List, Any, Dict
from typing_extensions import TypedDict
from pprint import pprint

from langchain_core.messages import BaseMessage, AIMessage
from langgraph.graph.message import add_messages
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.callbacks.base import BaseCallbackHandler
from langgraph.checkpoint.memory import MemorySaver

import LegalDefAgent.src.retriever as retriever
import LegalDefAgent.src.tools as tools
import LegalDefAgent.src.schema as schema


class CustomHandler(BaseCallbackHandler):
    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> Any:
        formatted_prompts = "\n".join(prompts)
        print(f"-> PROMPT:\n{formatted_prompts}\n")


class AgentState(TypedDict):
    # The add_messages function defines how an update should be processed
    # Default is to replace. add_messages says "append"
    question: str
    messages: Annotated[Sequence[BaseMessage], add_messages]
    answer: AIMessage


class LegalDefAgent:
    def __init__(self, model):
        self.model = model
        self.vectorstore = retriever.setup_vectorstore()
        self.retriever_tool = tools.create_vector_search_tool(self.vectorstore)
        self.tools = [self.retriever_tool]
        self.model_with_tools = self.model.bind_tools(self.tools)
        self.workflow = self.setup_workflow()

    def agent(self, state):
        """
        Invokes the agent model to generate a response based on the current state. Given
        the question, it will decide to retrieve using the retriever tool, or simply end.

        Args:
            state (messages): The current state

        Returns:
            dict: The updated state with the agent response appended to messages
        """
        print("---QUERY AGENT---")
        question = state["question"]
        messages = state["messages"]
        response = self.model_with_tools.invoke(messages, config={"callbacks": [CustomHandler()]})
        # We return a list, because this will get added to the existing list
        return {"messages": [response], "question": question}
    

    def filter_definitions(self, state) -> Literal["generate"]:
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
            If the definition contains keyword(s) or semantic meaning related to the user's question, keep it.\n
            Output only the relevant definitions using the formatting instructions provided. \n
            Here are the formatting instructions: {format_instructions} \n
            Here are the retrieved definitions, one for each line: \n {context} \n
            Here is the user question: {question} \n
            """,
            input_variables=["context", "question"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)


        # Chain
        chain = prompt | self.model

        messages = state["messages"]
        last_message = messages[-1]

        question = state["question"]
        docs = last_message.content

        response = chain.invoke({"question": question, "context": docs}, config={"callbacks": [CustomHandler()]})

        return {"messages": [response]}



    def generate(self, state):
        """
        Generate definition

        Args:
            state (messages): The current state

        Returns:
            str: A legal definition
        """
        print("---GENERATE---")
        messages = state["messages"]
        question = state["question"]
        last_message = messages[-1]

        docs = last_message.content

        # Prompt
        prompt = PromptTemplate(
            template="""
            You are a legal assistant for the task of generating legal definitions. \n
            Use the following retrieved definitions as context to answer the question. \n
            If you don't know the answer, just say that you don't know. \n
            Keep the answer concise and straight to the point, giving only the definition.
            Question: {question} \n
            Context: {context} \n
            Answer:
            """,
            input_variables=["context", "question"]
        )

        # LLM
        llm = self.model

        # Post-processing
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        # Chain
        rag_chain = prompt | llm | StrOutputParser()

        # Run
        response = rag_chain.invoke({"context": docs, "question": question}, config={"callbacks": [CustomHandler()]})
        return {"messages": [response], "answer": response}
    

    def eurlex_agent(self, state):
        raise NotImplementedError

    def normattiva_agent(self, state):
        raise NotImplementedError

    def pdl_agent(self, state):
        raise NotImplementedError

    def setup_workflow(self):
        # Define a new graph
        workflow = StateGraph(AgentState)

        # Define the nodes we will cycle between
        workflow.add_node("agent", self.agent)  # agent
        workflow.add_node("retrieve", ToolNode([self.retriever_tool]))  # retrieval
        workflow.add_node("filter", self.filter_definitions)  # retrieval
        workflow.add_node("generate", self.generate)  # Generating a response after we know the documents are relevant
        #workflow.add_node("RefResolver", self.resolve_references)
        #workflow.add_node("EurLex agent", self.eurlex_agent)
        #workflow.add_node("Normattiva agent", self.normattiva_agent)
        #workflow.add_node("PDL agent", self.pdl_agent)


        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges(
            "agent",
            tools_condition,
            {
                "tools": "retrieve",
                END: END,
            },
        )

        workflow.add_edge("retrieve", "filter")
        #workflow.add_edge("filter", "RefResolver")
        workflow.add_edge("filter", "generate")
        #workflow.add_edge("generate", "EurLex agent")
        #workflow.add_edge("generate", "Normattiva agent")
        #workflow.add_edge("generate", "PDL agent")
        #workflow.add_edge("EurLex agent", END)
        #workflow.add_edge("Normattiva agent", END)
        #workflow.add_edge("PDL agent", END)
        workflow.add_edge("generate", END)

        #memory = MemorySaver()
        self.graph = workflow.compile()#checkpointer=memory)

        return workflow

    
    def invoke(self, question):
        inputs = {
            "question": question,
            "messages": [
                ("system", ''),
                ("user", question),
            ]
        }

        for output in self.graph.stream(inputs):
            for key, value in output.items():
                print(f"<- OUTPUT from node '{key}':\n")
                print(value["messages"][-1])#, indent=2, width=80, depth=None)
            print("\n---\n")