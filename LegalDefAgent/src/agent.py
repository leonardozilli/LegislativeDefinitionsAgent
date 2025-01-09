from typing import Annotated, Literal, Sequence, Dict, Any
from typing_extensions import TypedDict
import uuid
from datetime import date

from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langgraph.graph.message import add_messages
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.callbacks import dispatch_custom_event
from langchain_core.runnables import RunnableConfig


from . import llm, tools, utils
from .retriever import vector_store, exist_db
from .settings import settings
from .schema.task_data import Task
from .schema.definition import DefinitionsList, Definition
from .schema.grader import DefinitionRelevance
from .schema.query import Query


class AgentState(TypedDict):
    input: str
    messages: Annotated[Sequence[BaseMessage], add_messages]
    question: str
    definendum: str
    retrieved_definitions: str
    relevant_defs: str
    answer_def: str
    task: str
    query_filters: Dict[str, Any]


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

    async def extract_query(self, state: AgentState, config: RunnableConfig) -> AgentState:
        """
        Extracts the definendum and date from the user's question.
        Args:
            state (AgentState): The current state
        Returns:
            dict: Updated state with the definendum and date
        """
        task = Task("Extract query")
        parser = JsonOutputParser(pydantic_object=Query)
        prompt = PromptTemplate(
            template="""
            Your tasks are to extract information about the definendum and date from the input sentence. 
            \n
            1. Extract the definendum, which is the term to be defined, from the sentence.
            Examples:
            * "What is the definition of a contract?" -> "contract"
            * "What is a fishing net?" -> "fishing net"
            * "What's the definition of 'vessel'?" -> "vessel"
            \n
            2. Extract possible dates from the sentence. The dates should fit into one of the following fields of the Pydantic model:
                - time_point: for specific dates (e.g., '2021-01-01')
                - from_date and to_date: for date ranges (e.g., 'from_date': '2010-01-01', 'to_date': '2015-12-31')
            Examples:
            * "What is the definition of a contract in 2015?" -> "time_point": "2015-01-01"
            * "What was the definition of 'bear' between 2010 and 2015?" -> "from_date": "2010-01-01", "to_date": "2015-12-31"
            * "What is a fishing net?" -> "time_point": None, "from_date": None, "to_date": None
            \n
            IMPORTANT NOTES:
            * If you don't find a definendum in the sentence, fill the definendum with: "None"
            * If you don't find a date or a date range in the sentence, fill their spots with: "None"
            * You should interpret single years as a date range. E.g., "2015" should be interpreted as from_date: "2015-01-01", to_date: "2015-12-31"
            * Follow the following formatting instructions: {format_instructions}
            \n
            Input sentence: {question}
            """,
            input_variables=["question"],
            ).partial(format_instructions=parser.get_format_instructions())

        chain = prompt | self.model | parser
        input_question = state["question"]
        response = chain.invoke({"question": input_question})

        await task.start(data={"input": input_question}, config=config)
        await task.finish(result="success", data={"output": response}, config=config)
        return {"messages": [utils.json_to_aimessage(response)], "definendum": response.get('definendum', None), "query_filters": {k:v for k,v in response.items() if k != 'definendum'} }

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

    async def grade_definitions(self, state: AgentState, config: RunnableConfig) -> AgentState:
        """
        Determines whether the retrieved definitions answer the user's question.
        Args:
            state (AgentState): The current state
        Returns:
            dict: Updated state with the definitions
        """
        await Task("Grade definitions").start(config=config, data={"input": (state["retrieved_definitions"])})

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

        date_filter = utils.parse_date_filters(state["query_filters"])

        # Score each doc
        relevant_defs = []
        for doc in state["relevant_defs"]:
            print(doc.page_content)

            if not date_filter:
                pass
            else:
                doc_date = exist_db.retrieve_doc_date(doc)
                if isinstance(date_filter, date):
                    if doc_date == date_filter:
                        pass
                    else:
                        continue
                elif isinstance(date_filter, tuple):
                    if date_filter[0] <= doc_date <= date_filter[1]:
                        pass
                    else:
                        continue



                structured_llm_grader = self.model.with_structured_output(DefinitionRelevance)
                grader_chain = prompt | structured_llm_grader

                score = grader_chain.invoke(
                    {"question": question, "definition": doc}
                )
                grade = score.binary_score
                if grade == "yes":
                    print("---GRADE: DOCUMENT RELEVANT---")
                    relevant_defs.append(doc)
                else:
                    print("---GRADE: DOCUMENT NOT RELEVANT---")
                    continue

        await Task("Grade definitions").finish(result="success", data={"output": (relevant_defs)}, config=config)
        return {"relevant_defs": relevant_defs}
    
    def empty_list(self, state: AgentState) -> bool:
        """
        Check if the list of definitions is empty.
        Args:
            state (AgentState): The current state
        Returns:
            bool: True if the list is empty, False otherwise
        """
        boo = len(list(state["relevant_defs"])) > 0

        return "definitions found" if boo else "no definitions"

    async def rank_definitions(self, state: AgentState, config: RunnableConfig) -> AgentState:
        """
        Rank the definitions based on their relevance to the question.
        Args:
            state (AgentState): The current state
        Returns:
            dict: Updated state with the ranked definitions
        """
        await Task("Rank definitions").start(config=config)
        parser = JsonOutputParser(pydantic_object=schema.DefinitionsList)
        prompt = PromptTemplate(
            template="""
            You are a legal expert ordering legal definitions based on their source.
            Your job is to order the list of definitions provided to you following the hierarchy provided.
            You can only answer with valid, directly parsable json.
            Here are the formatting instructions: {format_instructions}
            Here are the definitions to rank: {context}
            Here are the hierarchical schema instructions:
                * EurLex definitions should be ranked first.
                * If the user question is in Italian, Normattiva and PDL definitions should be ranked first
            """,
            input_variables=["context"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )

        chain = prompt | self.model | parser
        response = chain.invoke({"context": state["relevant_defs"]})

        await Task("Rank definitions").finish(result="success", data={"output": response['relevant_definitions']}, config=config)
        return {"messages": [utils.json_to_aimessage(response)], "relevant_defs": response['relevant_definitions']}
    
    async def pick_definition(self, state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
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

        chain = prompt | self.model | parser
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
            You are a lawyer and your job is to generate legal definitions.
            Keep the answer concise and straight to the point, giving only the definition.
            User Question: {question}
            VERY IMPORTANT NOTES:
            * You should answer with "I couldn't find a definition for "{definendum}", so here's a generated one:" followed by the generated definition.
            """,
            input_variables=["question", "definendum"]
        )

        definendum = state["definendum"]
        question = state["question"]
        chain = prompt | self.model
        response = chain.invoke({"question": question, "definendum": definendum})

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

        return state["answer_def"]['metadata']['dataset']

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
        response = chain.invoke({"context": state["messages"][-1].content, "metadata": state['answer_def']['metadata'], "definendum": state['definendum']})

        dispatch_custom_event("answering", {"foo": "bar"})
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
        workflow.add_node("extract_query", self.extract_query)
        workflow.add_node("query_vectorstore", self.query_vectorstore)
        workflow.add_node("filter_definitions", self.filter_definitions)
        workflow.add_node("pick_definition", self.pick_definition)
        workflow.add_node("eurlex_agent", self.eurlex_agent)
        workflow.add_node("normattiva_agent", self.normattiva_agent)
        workflow.add_node("pdl_agent", self.pdl_agent)
        workflow.add_node("answer", self.answer)
        #workflow.add_node("ask_to_generate", self.ask_to_generate)
        workflow.add_node("generate_definition", self.generate_definition)

        # Edges
        workflow.add_edge(START, "task_manager")
        workflow.add_conditional_edges("task_manager",
                                        self.route_task,
                                        {
                                            "definition": "extract_query",
                                            "no_task": "model"
                                        })
        workflow.add_edge("extract_query", 'query_vectorstore')
        workflow.add_edge("query_vectorstore", 'filter_definitions')
        #workflow.add_conditional_edges("filter_definitions",
                                        #self.empty_list,
                                        #{
                                             #"empty or single definition": "answer",
                                             #"more than one definition": "rank_definitions"
                                        #})
        #workflow.add_edge("filter_definitions", "pick_definition")
        workflow.add_conditional_edges("filter_definitions",
                                       self.empty_list,
                                       {
                                           "definitions found": "pick_definition",
                                           "no definitions": "generate_definition"
                                       }
                                       )
        #workflow.add_edge("rank_definitions", "pick_definition")
        workflow.add_conditional_edges("pick_definition",
                                        self.call_db_agents,
                                        {
                                             "EurLex": "eurlex_agent",
                                             "Normattiva": "normattiva_agent",
                                             "PDL": "pdl_agent"
                                        })
        #workflow.add_conditional_edges("ask_to_generate",
                                       #self.evaluate_user_confirmation,
                                       #{
                                           #True: "generate_definition",
                                           #False: END
                                       #})
        workflow.add_edge("generate_definition", END)
        workflow.add_edge("eurlex_agent", "answer")
        workflow.add_edge("normattiva_agent", "answer")
        workflow.add_edge("pdl_agent", "answer")
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