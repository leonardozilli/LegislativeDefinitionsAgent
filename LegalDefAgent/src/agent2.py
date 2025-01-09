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


from . import llm, tools, utils
from .retriever import vector_store
from .settings import settings
from .schema.task_data import Task
from .schema.definition import DefinitionsList, Definition
from .schema.grader import DefinitionRelevance


class AgentState(TypedDict):
    input: str
    messages: Annotated[Sequence[BaseMessage], add_messages]
    question: str
    definendum: str
    retrieved_definitions: str
    relevant_defs: str
    answer_def: str
    task: str


class LegalDefAgent:
    def __init__(self, model, milvusdb_uri=None, search_k=7):
        self.model = model
        self.vectorstore = vector_store.setup_vectorstore(milvusdb_uri=milvusdb_uri)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": search_k})
        #self.retriever_tool = tools.create_vector_search_tool(self.retriever)
        #self.tools = [self.retriever_tool]
        #self.model_with_tools = self.model.bind_tools(self.tools)
        #self.tasks_tools = [self.definition_agent]
        #self.db_agents = [self.eurlex_agent, self.normattiva_agent, self.pdl_agent]
        #self.db_agents_tools = [tools.extract_definition_from_xmldb]
        self.workflow = self.setup_workflow()

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

        chain = prompt | self.model

        input_question = state["messages"][-1].content
        response = chain.invoke({"question": input_question})

        return {"task": response.content.lower(), "question": input_question}


    def route_task(self, state: AgentState) -> Literal["definition", "no_task"]:
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

    async def extract_definendum(self, state: AgentState, config: RunnableConfig) -> AgentState:
        """
        Extracts the definendum from the user's question.
        Args:
            state (AgentState): The current state
        Returns:
            dict: Updated state with the definendum
        """
        task = Task("Extract definendum")
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

        await task.start(data={"input": input_question}, config=config)
        await task.finish(result="success", data={"output": response.content}, config=config)
        return {"messages": [response], "definendum": response.content}

    async def query_vectorstore(self, state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
        """
        Retrieves definitions from the vector store based on the definendum.
        Args:
            state (AgentState): The current state
        Returns:
            dict: Updated state with retrieved definitions
        """
        definendum = state["definendum"]
        await Task("Query vector store").start(data={"input": state['definendum']}, config=config)
        retrieved_definitions = self.retriever.invoke(definendum)

        await Task("Query vector store").finish(result="success", data={"output": str(retrieved_definitions)}, config=config)
        return {"retrieved_definitions": str(retrieved_definitions)}

    async def filter_definitions(self, state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
        """
        Filters the retrieved definitions based on their relevance to the question.
        Args:
            state (AgentState): The current state
        Returns:
            dict: Updated state with relevant definitions
        """
        parser = JsonOutputParser(pydantic_object=DefinitionsList)
        prompt = PromptTemplate(
            template="""
            You are a legal expert assessing the relevance of legal definitions to a user question.
            Below you will find a list of definitions that were automatically retrieved.
            Your task is to filter the list of definitions provided to you keeping only the relevant ones.
            If the text of the definition contains keyword(s) or semantic meaning related to the user's question, keep it. Otherwise discard it.
            Output only the relevant definitions using the formatting instructions provided.
            VERY IMPORTANT NOTES:
            * You can only answer with valid, directly parsable json.
            * You should output only the formatted relevant definitions, without any additional text.
            * If you can't find any relevant definitions, you should output this: {{"relevant_definitions": []}}\n\n
            Here are the formatting instructions: {format_instructions}
            Here are the retrieved definitions: {context}
            Here is the term asked by the user: {question}
            """,
            input_variables=["context", "question"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )
        
        chain = prompt | self.model.with_structured_output(DefinitionsList)# | parser
        question = state["question"]
        definendum = state["definendum"]
        retrieved_definitions = state["retrieved_definitions"]
        await Task("Filter definitions").start(data={"input": retrieved_definitions}, config=config)
        response = chain.invoke({"context": retrieved_definitions, "question": definendum}) # definendum or question??

        await Task("Filter definitions").finish(result="success", data={"output": response}, config=config)
        return {"messages": [utils.json_to_aimessage(response)], "relevant_defs": response}
    
    def empty_list_check(self, state: AgentState) -> str:
        """
        Check if the list of definitions is empty.
        Args:
            state (AgentState): The current state
        Returns:
            bool: True if the list is empty, False otherwise
        """
        boo = len(list(state["relevant_defs"])) > 0

        return "definitions found" if boo else "no definitions"

    def answer_list_check(self, state: AgentState) -> str:
        """
        Check if the list of definitions is empty.
        Args:
            state (AgentState): The current state
        Returns:
            str: 
        """
        print("---answer_list_check---")
        match len(list(state["relevant_defs"])):
            case 0:
                print("no definitions")
                return "no definitions"
            case 1:
                print("one definition")
                return "one definition"
            case _:
                print("multiple definitions")
                return "multiple definitions"


    async def grade_definitions(self, state: AgentState, config: RunnableConfig) -> AgentState:
        """
        Determines whether the retrieved definitions answer the user's question.
        Args:
            state (AgentState): The current state
        Returns:
            dict: Updated state with the definitions
        """
        await Task("Grade definitions").start(config=config, data={"input": (state["relevant_defs"])})
        #parser = JsonOutputParser(pydantic_object=DefinitionRelevance)
        prompt = PromptTemplate(
            template="""
            You are a legal expert with the task of assessing whether a retrieved legal definition is the correct one answer to a user's question.
            Give a binary score 'yes' or 'no' score to indicate whether the definition is relevant to the question. \n
            Here is the user question: {question}
            Here is the retrieved definition: {definition}
            """,
            input_variables=["question", "definition"],
        )

        print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
        question = state["question"]
        structured_llm_grader = self.model.with_structured_output(DefinitionRelevance)
        grader_chain = prompt | structured_llm_grader

        # Score each doc
        answer_defs = []
        for doc in state["relevant_defs"]:
            score = grader_chain.invoke(
                {"question": question, "definition": d}
            )
            grade = score.binary_score
            if grade == "yes":
                print("---GRADE: DOCUMENT RELEVANT---")
                answer_defs.append(d)
            else:
                print("---GRADE: DOCUMENT NOT RELEVANT---")
                continue

        await Task("Grade definitions").finish(result="success", data={"output": (answer_defs)}, config=config)
        return {"relevant_defs": answer_defs}

    async def rank_definitions(self, state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
        """
        Picks the most relevant definition from a list of definitions.
        Args:
            state (AgentState): The current state
        Returns:
            dict: Updated state with the best definition
        """
        await Task("Pick definition").start(config=config, data={"input": (state["relevant_defs"])})
        parser = JsonOutputParser(pydantic_object=Definition)
        prompt = PromptTemplate(
            template="""
            You are a legal expert evaluating legal definitions.
            Your job is to pick the SINGLE most appropriate definition from the list provided to you in response to a user's query.
            You can only answer with valid, directly parsable json containe ONE definition.
            Here are the formatting instructions: {format_instructions}
            Here are the definitions to rank: {context}
            Here are some info to base your decision on:
                * EurLex is the European Union's legal database.
                * Normattiva is the Italian legal database.
                * PDL is the Italian Parliament's legal database.
                * If the user's query is in Italian or asks about the Italian legislation, Normattiva and PDL definitions should preferred.
                * If the user asks about the European legislation, EurLex definitions should preferred.
            """,
            input_variables=["context"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )

        chain = prompt | self.model# | parser
        response = chain.invoke({"context": state["relevant_defs"]})

        await Task("Pick definition").finish(result="success", data={"output": str(response)}, config=config)
        return {"messages": [utils.json_to_aimessage(response)], "answer_def": response}
 

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
            Use the retrieved definitions to style the generated definition.
            User Question: {question}
            Retrieved definition: {retrieved_definitions}
            VERY IMPORTANT NOTES:
            * You should answer with "I couldn't find a definition for "{definendum}", so here's a generated one:\n\n" followed by the generated definition.
            """,
            input_variables=["question", "definendum", "retrieved_definitions"]
        )

        definendum = state["definendum"]
        question = state["question"]
        chain = prompt | self.model
        response = chain.invoke({"question": question, "definendum": definendum, "retrieved_definitions": state["retrieved_definitions"]})

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

    async def eurlex_agent(self, state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
        """
        The interface agent for the Eurlex database.
        """
        await Task("Eurlex agent").start(data={"input": state["answer_def"]["metadata"]}, config=config)

        content = tools.extract_definition_from_xmldb(state["answer_def"]['metadata'])
        await Task("Eurlex agent").finish(result="success", data={"output": content}, config=config)
        return {"messages": [AIMessage(content)]}

    async def normattiva_agent(self, state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
        """
        The interface agent for the Normattiva database.
        """
        await Task("Normattiva agent").start(data={"input": state["answer_def"]["metadata"]}, config=config)
        content = tools.extract_definition_from_xmldb(state["answer_def"]['metadata'])

        await Task("Normattiva agent").finish(result="success", data={"output": content}, config=config)
        return {"messages": [AIMessage(content)]}

    async def pdl_agent(self, state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
        """
        The interface agent for the PDL database.
        """
        await Task("PDL agent").start(data={"input": state["answer_def"]["metadata"]}, config=config)
        content = tools.extract_definition_from_xmldb(state["answer_def"]['metadata'])

        await Task("PDL agent").finish(result="success", data={"output": content}, config=config)
        return {"messages": [AIMessage(content)]}

    def call_db_agents(self, state: AgentState) -> Literal["Eurlex", "Normattiva", "PDL"]:

        return state["relevant_defs"][0]['metadata']['dataset']

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
            You are a legal expert part of a legal workflow and your job is to provide the final answer to the user.
            You are provided with the content of a definition and some metadata.
            If the definition to the exact term that the user asked for is available, you should answer the definition and the metadata.
            Else, if the definition is not exactly what the user asked for, you should answer with "I couldn't find the exact definition, but here is a related one:" followed by the definition and the metadata.
            Lastly, if the definition is empty, you should ask the user if they want you to generate one.
            VERY IMPORTANT NOTES:
            * You should answer in the same language as the user's question (e.g. if the user asked in Italian, you should answer in Italian. If the definition is in English, you should explain that it has been translated).
            * You should provide the metadata to the user using natural language.
            User Question: {definendum}
            definition_metadata: {metadata}
            definition_content: {context}
            """,
            input_variables=["context", "definendum", "metadata"]
        )

        chain = prompt | self.model
        response = chain.invoke({"context": state['answer_def']['definition_text'], "metadata": state['answer_def']['metadata'], "definendum": state['definendum']})

        return {"messages": [response]}


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
        workflow.add_node("grade_definitions", self.grade_definitions)
        workflow.add_node("rank_definitions", self.rank_definitions)
        #workflow.add_node("eurlex_agent", self.eurlex_agent)
        #workflow.add_node("normattiva_agent", self.normattiva_agent)
        #workflow.add_node("pdl_agent", self.pdl_agent)
        workflow.add_node("answer", self.answer)
        #workflow.add_node("ask_to_generate", self.ask_to_generate)
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
        #workflow.add_conditional_edges("filter_definitions",
                                        #self.empty_list,
                                        #{
                                             #"empty or single definition": "answer",
                                             #"more than one definition": "rank_definitions"
                                        #})
        #workflow.add_edge("filter_definitions", "pick_definition")
        workflow.add_conditional_edges("filter_definitions",
                                       self.empty_list_check,
                                       {
                                           "definitions found": "grade_definitions",
                                           "no definitions": "generate_definition"
                                       })
        # workflow.add_edge("rank_definitions", "pick_definition")
        workflow.add_conditional_edges("grade_definitions",
                                       self.answer_list_check,
                                       {
                                           "no definitions": "generate_definition",
                                           "one definition": "answer",
                                           "multiple definitions": "rank_definitions"
                                       })
        #workflow.add_conditional_edges("grade_definitions",
                                       #self.call_db_agents,
                                        #{
                                             #"EurLex": "eurlex_agent",
                                             #"Normattiva": "normattiva_agent",
                                             #"PDL": "pdl_agent"
                                        #})
        #workflow.add_conditional_edges("ask_to_generate",
                                       #self.evaluate_user_confirmation,
                                       #{
                                           #True: "generate_definition",
                                           #False: END
                                       #})
        workflow.add_edge("rank_definitions", "answer")
        workflow.add_edge("generate_definition", END)
        #workflow.add_edge("eurlex_agent", "answer")
        #workflow.add_edge("normattiva_agent", "answer")
        #workflow.add_edge("pdl_agent", "answer")
        workflow.add_edge("answer", END)
        workflow.add_edge("model", END)

        memory = MemorySaver()
        self.graph_runnable = workflow.compile(checkpointer=memory)#, interrupt_after=["ask_to_generate"])

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


defagent = LegalDefAgent(model=llm.get_model(settings.DEFAULT_MODEL)).graph_runnable