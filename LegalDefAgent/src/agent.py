from typing import Annotated, Literal, Sequence, List, Dict, Any
from typing_extensions import TypedDict
from pprint import pprint

from langchain_core.messages import BaseMessage, AIMessage, SystemMessage, HumanMessage
from langgraph.graph.message import add_messages
from langchain_core.tools import tool, StructuredTool
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.callbacks.base import BaseCallbackHandler
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field
from langchain.agents.react.agent import create_react_agent
from langchain_core.callbacks import adispatch_custom_event

from LegalDefAgent.src import tools, schema, utils
from LegalDefAgent.src.retriever import vector_store

import uuid


class AgentState(TypedDict):
    input: str
    messages: Annotated[Sequence[BaseMessage], add_messages]
    question: str
    definendum: str
    retrieved_definitions: str
    relevant_defs: str
    def_missing: bool
    task: str


class LegalDefAgent:
    def __init__(self, model,milvusdb_uri=None, search_k=7):
        self.model = model
        self.vectorstore = vector_store.setup_vectorstore(milvusdb_uri=milvusdb_uri)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": search_k})
        #self.retriever_tool = tools.create_vector_search_tool(self.retriever)
        #self.tools = [self.retriever_tool]
        #self.model_with_tools = self.model.bind_tools(self.tools)
        #self.tasks_tools = [self.definition_agent]
        self.workflow = self.setup_workflow()

    def LegalAgent(self, state: AgentState) -> Literal["DefinitionsAgent",]:
        """
        Invokes the master agent that invokes the task-related agents.
        Args:
            state (AgentState): The current state
        Returns:
            str: The name of the task-agent
        """

        available_tasks = ["definition",]

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
        input_question = state["messages"][-1].content
        response = chain.invoke({"question": input_question})

        return response.content.lower() if response.content.lower() in available_tasks else END

    def task_manager(self, state: AgentState):
        """Analyze the user's query and determine the appropriate routing.

        This function uses a language model to classify the user's query and decide how to route it
        within the conversation flow.

        Args:
            state (AgentState): The current state of the agent, including conversation history.
            config (RunnableConfig): Configuration with the model used for query analysis.

        Returns:
            dict[str, Router]: A dictionary containing the 'task' key with the classification result (classification type and logic).
        """

        class task(BaseModel):
            """The task to perform."""

            task: str = Field(description="The task to perform. (e.g. 'definition')")


        prompt = PromptTemplate(
            template="""
                You are a legal drafting assistant.
                A user will come to you with some input. Your job is to classify what type of task the input belongs to.
                The types of tasks you should classify it as are:
                ## `definition`
                The task is to retrieve the definition of a term.
                Some examples (user_query -> what you should output): 
                * "What is the definition of a contract?" -> "definition"
                * "What is a fishing net?" -> "definition"
                * "house" -> "definition"

                VERY IMPORTANT NOTES:
                * You should output only the single definendum, without any additional text.
                * If you can't find a task suitable for the user input, you should inform the user that you cannot satisfy their query and instruct them that at the moment you can only deal with the tasks described previously.\n

                Here is the user question: {question}
            """,
            input_variables=["question"],
        )

        chain = prompt | self.model#.bind_tools(self.tasks_tools, parallel_tool_calls=False)

        input_question = state["messages"][-1].content
        response = chain.invoke({"question": input_question})

        return {"task": response.content.lower(), "question": input_question}


    def route_task(self, state: AgentState) -> str:
        """
        Routes the task to the appropriate agent.
        Args:
            state (AgentState): The current state
        Returns:
            str: The name of the task-agent
        """
        if state["task"] == "definition":
            return "definition"
        else:
            return "no_task"

    def extract_definendum(self, state: AgentState) -> Dict[str, Any]:
        """
        Extracts the definendum from the user's question.
        Args:
            state (AgentState): The current state
        Returns:
            dict: Updated state with the definendum
        """
        prompt = PromptTemplate(
            template="""
                You are a legal drafting assistant.
                Your job is to extract the definendum, that is the term that is to be defined, out of a sentence.
                E.g.: 
                * "What is the definition of a contract?" -> "contract"
                * "What is a fishing net?" -> "fishing net"
                * "What's the definition of 'vessel'" -> "vessel"
                VERY IMPORTANT NOTES:
                * You should output only the single definendum, without any additional text.
                * If you don't find a definendum in the sentence, you should output: "None"
                Input sentence: {question}
            """,
            input_variables=["question"],
        )

        chain = prompt | self.model
        input_question = state["question"]
        response = chain.invoke({"question": input_question})

        return {"messages": [response], "definendum": response.content}

    def query_vectorstore(self, state: AgentState) -> Dict[str, Any]:
        """
        Retrieves definitions from the vector store based on the definendum.
        Args:
            state (AgentState): The current state
        Returns:
            dict: Updated state with retrieved definitions
        """
        definendum = state["definendum"]
        retrieved_definitions = self.retriever.invoke(definendum)
        return {"retrieved_definitions": str(retrieved_definitions)}

    def filter_definitions(self, state: AgentState) -> Dict[str, Any]:
        """
        Filters the retrieved definitions based on their relevance to the question.
        Args:
            state (AgentState): The current state
        Returns:
            dict: Updated state with relevant definitions
        """
        parser = JsonOutputParser(pydantic_object=schema.DefinitionsList)
        prompt = PromptTemplate(
            template="""
            You are a legal expert API assessing the relevance of legal definitions to a user question.
            You can only answer with valid, directly parsable json.
            Your job is to filter the list of definitions provided to you keeping only the relevant ones.
            If the text of the definition contains keyword(s) or semantic meaning related to the user's question, keep it. Otherwise discard it.
            Output only the relevant definitions using the formatting instructions provided.
            VERY IMPORTANT NOTES:
            * You should output only the formatted relevant definitions, without any additional text.
            * If none of the definitions are relevant, you should output an empty list.
            Here are the formatting instructions: {format_instructions}
            Here are the retrieved definitions: {context}
            Here is the user's question: {question}
            """,
            input_variables=["context", "question"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )

        chain = prompt | self.model | parser
        question = state["question"]
        retrieved_definitions = state["retrieved_definitions"]
        response = chain.invoke({"question": question, "context": retrieved_definitions})

        return {"messages": [utils.json_to_aimessage(response)], "relevant_defs": response['relevant_definitions']}
    
    def empty_list(self, state: AgentState) -> bool:
        """
        Check if the list of definitions is empty.
        Args:
            state (AgentState): The current state
        Returns:
            bool: True if the list is empty, False otherwise
        """
        boo = len(list(state["relevant_defs"])) > 1

        return "more than one definition" if boo else "empty or single definition"

    def rank_definitions(self, state: AgentState) -> Dict[str, Any]:
        """
        Rank the definitions based on their relevance to the question.
        Args:
            state (AgentState): The current state
        Returns:
            dict: Updated state with the ranked definitions
        """
        return state

    def answer(self, state: AgentState) -> Dict[str, Any]:
        """
        Answer user with a definition.
        Args:
            state (AgentState): The current state
        Returns:
            dict: Updated state with the answer
        """
        prompt = PromptTemplate(
            template="""
            You are a legal expert and your job is to find the accurate legal definition for a term.
            Find in the retrieved definitions the one that answers the user's question.
            Keep the answer concise and straight to the point, giving only the definition.
            VERY IMPORTANT NOTES:
            * You should output only the string with the definition, without any additional text.
            * If you don't find a definition for the user's question, you should answer with: "I can't find a definition for that. Do you want me to generate one for you?"
            User Question: {question}
            Retrieved definitions: {context}
            """,
            input_variables=["context", "question"]
        )

        chain = prompt | self.model
        response = chain.invoke({"context": state['relevant_defs'], "question": state['question']})
        def_missing = response.content == "I can't find a definition for that. Do you want me to generate one for you?"

        return {"messages": [response], "def_missing": def_missing}

    def generate_definition(self, state: AgentState) -> Dict[str, Any]:
        """
        Generate a definition.
        Args:
            state (AgentState): The current state
        Returns:
            dict: Updated state with the generated definition
        """
        prompt = PromptTemplate(
            template="""
            You are a legal expert and your job is to generate legal definitions.
            Keep the answer concise and straight to the point, giving only the definition.
            User Question: {question}
            VERY IMPORTANT NOTES:
            * You should answer with "Here's the generated definition:" followed by the generated definition.
            """,
            input_variables=["question"]
        )

        chain = prompt | self.model
        response = chain.invoke({"question": state['question']})

        return {"messages": [response]}

    def evaluate_user_confirmation(self, state: AgentState) -> bool:
        """
        Check for user confirmation to generate a definition.
        Args:
            state (AgentState): The current state
        Returns:
            bool: True if user confirms, False otherwise
        """
        user_last_message = state["messages"][-1].content
        return user_last_message.lower() in ("y", "yes")
    
    def should_generate(self, state: AgentState) -> bool:
        """
        Check if the agent should generate a definition.
        Args:
            state (AgentState): The current state
        Returns:
            bool: True if the agent should generate a definition, False otherwise
        """
        return state["def_missing"]


    def perform_task(self, state: AgentState) -> bool:
        """
        Check if the agent should perform a task.
        Args:
            state (AgentState): The current state
        Returns:
            bool: True if the agent should perform a task, False otherwise
        """
        return state["task"]
    
    def _call_model(self, state: AgentState):
        messages = state["messages"]
        response = self.model.invoke(messages)
        return {"messages": [response]}
    
    def ask_to_generate(self, state: AgentState) -> Dict[str, Any]:
        """
        Ask the user if they want to generate a definition.
        Args:
            state (AgentState): The current state
        Returns:
            dict: Updated state with the question
        """
        print("---ask_to_generate---")
        return {"messages": [AIMessage(content="Do you want me to generate a definition for you?")]}

    def setup_workflow(self) -> StateGraph:
        """
        Setup the workflow for the agent.
        Returns:
            StateGraph: The compiled state graph
        """
        workflow = StateGraph(AgentState)

        # Nodes
        workflow.add_node("model", self._call_model)
        workflow.add_node("task_manager", self.task_manager)
        workflow.add_node("extract_definendum", self.extract_definendum)
        workflow.add_node("query_vectorstore", self.query_vectorstore)
        workflow.add_node("filter_definitions", self.filter_definitions)
        workflow.add_node("rank_definitions", self.rank_definitions)
        workflow.add_node("answer", self.answer)
        workflow.add_node("ask_to_generate", self.ask_to_generate)
        workflow.add_node("generate_definition", self.generate_definition)

        # Edges
        workflow.add_edge(START, "task_manager")
        workflow.add_conditional_edges("task_manager",
                                        self.route_task,
                                        {
                                            "definition": "extract_definendum",
                                            "no_task": "model"
                                        })
        workflow.add_edge("extract_definendum", 'query_vectorstore')
        workflow.add_edge("query_vectorstore", 'filter_definitions')
        workflow.add_conditional_edges("filter_definitions",
                                        self.empty_list,
                                        {
                                             "empty or single definition": "ask_to_generate",
                                             "more than one definition": "rank_definitions"
                                        })
        workflow.add_edge("rank_definitions", "answer")
        workflow.add_conditional_edges("ask_to_generate",
                                       self.evaluate_user_confirmation,
                                       {
                                           True: "generate_definition",
                                           False: END
                                       })
        workflow.add_edge("generate_definition", END)
        workflow.add_edge("answer", END)
        workflow.add_edge("model", END)

        memory = MemorySaver()
        self.graph_runnable = workflow.compile(checkpointer=memory, interrupt_after=["ask_to_generate"])

        return workflow

    def invoke(self, question: str=None, config=None) -> None:
        """
        Invoke the agent with a user's question.
        Args:
            question (str): The user's question
            config (optional): Additional configuration
        """
        if question:
            inputs = {
                "input": question,
                "messages": [
                    HumanMessage(content=question)
                ]
            }

        thread_id = config.get("configurable", None).get("thread_id")
        if thread_id is None:
            thread_id = uuid.uuid4()
        print(f"Thread ID: {thread_id}")

        config = {
            "configurable": {
                "thread_id": thread_id,
            }
        }

        _printed = set()
        while True:
            user = input("User (q/Q to quit): ")# if not question else question
            if user in {"q", "Q"}:
                print("\nClosing the conversation.")
                break
            output = None
            for output in self.graph_runnable.stream(
                {"messages": [HumanMessage(content=user)]}, config=config, stream_mode="values"
            ):
                utils._print_event(output, _printed)