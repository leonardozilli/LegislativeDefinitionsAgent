import os
from dotenv import load_dotenv

from uuid import uuid4

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from typing import Callable, TypeVar, Any, Dict, Optional
import inspect

from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx
from streamlit.delta_generator import DeltaGenerator

from langchain_core.callbacks.base import BaseCallbackHandler
import streamlit as st


# Define a function to create a callback handler for Streamlit that updates the UI dynamically
def get_streamlit_cb(parent_container: DeltaGenerator) -> BaseCallbackHandler:
    """
    Creates a Streamlit callback handler that updates the provided Streamlit container with new tokens.
    Args:
        parent_container (DeltaGenerator): The Streamlit container where the text will be rendered.
    Returns:
        BaseCallbackHandler: An instance of a callback handler configured for Streamlit.
    """

    # Define a custom callback handler class for managing and displaying stream events in Streamlit
    class StreamHandler(BaseCallbackHandler):
        """
        Custom callback handler for Streamlit that updates a Streamlit container with new tokens.
        """

        def __init__(self, container: st.delta_generator.DeltaGenerator, initial_text: str = ""):
            """
            Initializes the StreamHandler with a Streamlit container and optional initial text.
            Args:
                container (st.delta_generator.DeltaGenerator): The Streamlit container where text will be rendered.
                initial_text (str): Optional initial text to start with in the container.
            """
            self.container = container  # The Streamlit container to update
            self.thoughts_placeholder = self.container.container()  # container to hold tool_call renders
            self.tool_output_placeholder = None # placeholder for the output of the tool call to be in the expander
            self.token_placeholder = self.container.empty()  # for token streaming
            self.text = initial_text  # The text content to display, starting with initial text

        def on_llm_new_token(self, token: str, **kwargs) -> None:
            """
            Callback method triggered when a new token is received (e.g., from a language model).
            Args:
                token (str): The new token received.
                **kwargs: Additional keyword arguments.
            """
            self.text += token  # Append the new token to the existing text
            self.token_placeholder.write(self.text)

        def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs: Any) -> None:
            """
            Run when the tool starts running.
            Args:
                serialized (Dict[str, Any]): The serialized tool.
                input_str (str): The input string.
                kwargs (Any): Additional keyword arguments.
            """
            with self.thoughts_placeholder:
                status_placeholder = st.empty()   # Placeholder to show the tool's status
                with status_placeholder.status("Calling Tool...", expanded=True) as s:
                    st.write("called ", serialized["name"])  # Show which tool is being called
                    st.write("tool description: ", serialized["description"])
                    st.write("tool input: ")
                    st.code(input_str)   # Display the input data sent to the tool
                    st.write("tool output: ")
                    # Placeholder for tool output that will be updated later below
                    self.tool_output_placeholder = st.empty()
                    s.update(label="Completed Calling Tool!", expanded=False)   # Update the status once done

        def on_tool_end(self, output: Any, **kwargs: Any) -> Any:
            """
            Run when the tool ends.
            Args:
                output (Any): The output from the tool.
                kwargs (Any): Additional keyword arguments.
            """
            # We assume that `on_tool_end` comes after `on_tool_start`, meaning output_placeholder exists
            if self.tool_output_placeholder:
                self.tool_output_placeholder.code(output.content)   # Display the tool's output

    # Define a type variable for generic type hinting in the decorator, to maintain
    # input function and wrapped function return type
    fn_return_type = TypeVar('fn_return_type')

    # Decorator function to add the Streamlit execution context to a function
    def add_streamlit_context(fn: Callable[..., fn_return_type]) -> Callable[..., fn_return_type]:
        """
        Decorator to ensure that the decorated function runs within the Streamlit execution context.
        Args:
            fn (Callable[..., fn_return_type]): The function to be decorated.
        Returns:
            Callable[..., fn_return_type]: The decorated function that includes the Streamlit context setup.
        """
        ctx = get_script_run_ctx()  # Retrieve the current Streamlit script execution context

        def wrapper(*args, **kwargs) -> fn_return_type:
            """
            Wrapper function that adds the Streamlit context and then calls the original function.
            Args:
                *args: Positional arguments to pass to the original function.
                **kwargs: Keyword arguments to pass to the original function.
            Returns:
                fn_return_type: The result from the original function.
            """
            add_script_run_ctx(ctx=ctx)  # Add the Streamlit context to the current execution
            return fn(*args, **kwargs)  # Call the original function with its arguments

        return wrapper

    # Create an instance of the custom StreamHandler with the provided Streamlit container
    st_cb = StreamHandler(parent_container)

    # Iterate over all methods of the StreamHandler instance
    for method_name, method_func in inspect.getmembers(st_cb, predicate=inspect.ismethod):
        if method_name.startswith('on_'):  # Identify callback methods
            setattr(st_cb, method_name, add_streamlit_context(method_func))  # Wrap and replace the method

    # Return the fully configured StreamHandler instance with the context-aware callback methods
    return st_cb

load_dotenv()


import sys
sys.path.insert(1, '../LegalDefAgent')
from src.agent import LegalDefAgent
import src.models as models

agent = LegalDefAgent(model=models.groq)

graph_runnable = agent.graph
def invoke_our_graph(st_messages, callables):
    if not isinstance(callables, list):
        raise TypeError("callables must be a list")
    return graph_runnable.invoke({"messages": st_messages}, config={"callbacks": callables, "thread_id": str(uuid4())})

st.title("StreamLit 🤝 LangGraph")
st.markdown("#### Chat Streaming and Tool Calling using Custom Callback")

# Initialize the expander state
if "expander_open" not in st.session_state:
    st.session_state.expander_open = True

# Check if the OpenAI API key is set
if not os.getenv('OPENAI_API_KEY'):
    # If not, display a sidebar input for the user to provide the API key
    st.sidebar.header("OPENAI_API_KEY Setup")
    api_key = st.sidebar.text_input(label="API Key", type="password", label_visibility="collapsed")
    os.environ["OPENAI_API_KEY"] = api_key
    # If no key is provided, show an info message and stop further execution and wait till key is entered
    if not api_key:
        st.info("Please enter your OPENAI_API_KEY in the sidebar.")
        st.stop()


# Capture user input from chat input
prompt = st.chat_input()

# Toggle expander state based on user input
if prompt is not None:
    st.session_state.expander_open = False  # Close the expander when the user starts typing

# st write magic
with st.expander(label="Simple Chat Streaming and Tool Calling Using Custom Callback Handler", expanded=st.session_state.expander_open):
    """
    In this example, we're going to be creating our own [`BaseCallbackHandler`](https://api.python.langchain.com/en/latest/callbacks/langchain_core.callbacks.base.BaseCallbackHandler.html) called StreamHandler
    to stream our [_LangGraph_](https://langchain-ai.github.io/langgraph/) invocations with `token streaming` or `tool calling` and leveraging callbacks in our
    graph's [`RunnableConfig`](https://api.python.langchain.com/en/latest/runnables/langchain_core.runnables.config.RunnableConfig.html).

    The BaseCallBackHandler is a [Mixin](https://www.wikiwand.com/en/articles/Mixin) overloader function which we will use
    to implement `on_llm_new_token`, a method that run on every new generation of a token from the ChatLLM model,
    `on_tool_start` a method that runs on every tool call invocation even multiple tool calls, `on_tool_end`
    a method that runs on the end of the tool call to get the final result.
    """

# Initialize chat messages in session state
if "messages" not in st.session_state:
    st.session_state["messages"] = [AIMessage(content="Ask a definition away!")]

# Loop through all messages in the session state and render them as a chat on every st.refresh mech
for msg in st.session_state.messages:
    # https://docs.streamlit.io/develop/api-reference/chat/st.chat_message
    # we store them as AIMessage and HumanMessage as its easier to send to LangGraph
    if isinstance(msg, AIMessage):
        st.chat_message("assistant").write(msg.content)
    elif isinstance(msg, HumanMessage):
        st.chat_message("Ask a definition away!").write(msg.content)

# Handle user input if provided
if prompt:
    st.session_state.messages.append(HumanMessage(content=prompt))
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        # create a new placeholder for streaming messages and other events, and give it context
        st_callback = get_streamlit_cb(st.container())
        response = invoke_our_graph(st.session_state.messages, [st_callback])
        st.session_state.messages.append(AIMessage(content=response["messages"][-1].content))   # Add that last message to the st_message_state