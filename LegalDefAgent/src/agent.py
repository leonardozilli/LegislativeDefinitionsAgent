# %pip install -qU langchain-community langgraph langgraph-checkpoint-sqlite langchain-openai langchain-groq langchain_mistralai

from typing import Annotated, Literal, Sequence, List, Any, Dict
from typing_extensions import TypedDict
from pprint import pprint

from langchain_core.messages import BaseMessage, AIMessage, SystemMessage, HumanMessage, AnyMessage
from langgraph.graph.message import add_messages
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.output_parsers import PydanticOutputParser, JsonOutputParser
from langchain_core.callbacks.base import BaseCallbackHandler
from langgraph.checkpoint.memory import MemorySaver

from LegalDefAgent.src import tools
from LegalDefAgent.src import schema
from LegalDefAgent.src import models
from LegalDefAgent.src import utils
from LegalDefAgent.src.vector_store import vector_store


class AgentState(TypedDict):
    # The add_messages function defines how an update should be processed
    # Default is to replace. add_messages says "append"
    input: str
    messages: Annotated[Sequence[BaseMessage], add_messages]
    question: str
    definendum: str
    relevant_defs: List[str]
    definiens: List[str]


class LegalDefAgent:
    def __init__(self, model, search_k=7):
        self.model = model
        self.vectorstore = vector_store.setup_vectorstore()
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": search_k})
        self.retriever_tool = tools.create_vector_search_tool(self.retriever)
        self.tools = [self.retriever_tool]
        self.model_with_tools = self.model.bind_tools(self.tools)
        self.workflow = self.setup_workflow()

    def LegalAgent(self, state) -> Literal["DefAgent",]:
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

        chain = prompt | self.model

        input = state["messages"][-1].content

        response = chain.invoke({"question": input})

        match response.content.lower():
            case "definition":
                return "DefAgent"
            case _:
                return END


    def DefAgent(self, state):
        """
        Invokes the agent model to generate a response based on the current state. Given
        the question, it will decide to retrieve using the retriever tool, or simply end.

        Args:
            state (messages): The current state

        Returns:
            dict: The updated state with the agent response appended to messages
        """

        print("---DEFINITION AGENT---")
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

        chain = prompt | self.model

        input = state["messages"][-1].content

        response = chain.invoke({"question": input})

        return {"messages": [response], "question": input, "definendum": response.content}

    
    def retrieveAndFilterDefs(self, state) -> Literal["generate"]:
        """
        Filters the retrieved definitions based on their relevance to the question.

        Args:
            state (messages): The current state

        Returns:
            List[dict]: A list of dictionaries containing the relevant definitions and their metadata.
        """

        print("---FILTER DEFINITIONS---")

        parser = JsonOutputParser(pydantic_object=schema.DefinitionsList)

        prompt = PromptTemplate(
            template="""
            You are a legal expert API assessing the relevance of legal definitions to a user question. \n 
            You can only answer with valid, directly parsable json. \n
            Your job is to filter the list of definitions provided to you keeping only the relevant ones. \n
            If the text of the definition contains keyword(s) or semantic meaning related to the user's question, keep it. Otherwise discard it.\n
            Output only the relevant definitions using the formatting instructions provided. \n

            VERY IMPORTANT NOTES:
            * You should output only the formatted relevant definitions, without any additional text.
            * If none of the definitions are relevant, you should an empty list.

            Here are the formatting instructions: {format_instructions} \n
            Here are the retrieved definitions: \n {context} \n
            Here is the user's question: {question} \n

            """,
            input_variables=["context", "question"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )

        chain = prompt | self.model | parser #something wrong here

        question = state["question"]
        retrieved_definitions = self.retriever.invoke(question) # QUERY or DEFINENDUM???

        response = chain.invoke({"question": question, "context": retrieved_definitions})

        return {"messages": [utils.convert_to_aimessage(response)], "relevant_defs": response}


    def answer(self, state):
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

        chain = prompt | self.model

        response = chain.invoke({"context": state['relevant_defs'], "question": state['question']}, config={})

        return {"messages": [response], "definiens": response.content}

    

    def eurlex_agent(self, state):
        raise NotImplementedError

    def normattiva_agent(self, state):
        raise NotImplementedError

    def pdl_agent(self, state):
        raise NotImplementedError

    def setup_workflow(self):
        workflow = StateGraph(AgentState)

        workflow.add_node("DefAgent", self.DefAgent)
        workflow.add_node("retrieveAndFilterDefs", self.retrieveAndFilterDefs)  # retrieval
        workflow.add_node("answer", self.answer)  # Generating a response after we know the documents are relevant

        workflow.add_conditional_edges(START,
                                    self.LegalAgent,
                                    {
                                        "DefAgent": "DefAgent",
                                        END: END
                                    })
        workflow.add_edge("DefAgent", 'retrieveAndFilterDefs')
        workflow.add_edge("retrieveAndFilterDefs", "answer")
        workflow.add_edge("answer", END)

        #memory = MemorySaver()
        self.graph = workflow.compile()#checkpointer=memory)

        return workflow

    
    def invoke(self, question):
        inputs = {
            "input": question,
            "messages": [
                SystemMessage(content=''),
                HumanMessage(content=question)
            ]
        }

        for output in self.graph.stream(inputs, stream_mode="values"):
            for key, value in output.items():
                print(f"<- OUTPUT from node '{key}':\n")
                print(value["messages"][-1].pretty_print())#, indent=2, width=80, depth=None)
            print("\n---\n")


defagent = LegalDefAgent(model=models._get_model('groq', streaming=False))
graph = defagent.graph
