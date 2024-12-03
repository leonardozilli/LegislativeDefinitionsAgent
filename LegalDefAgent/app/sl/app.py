import os
from dotenv import load_dotenv

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
import asyncio

from astream_events_handler import invoke_our_graph   # Utility function to handle events from astream_events from graph

load_dotenv()

APP_TITLE = "Legal Definition Agent"
APP_ICON = "🧰"

st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    menu_items={},
)

models = {
    "llama-3.1-70b on Groq": 'groq',
    "OpenAI GPT-4o-mini (streaming)": 'gpt',
}

with st.sidebar:
    st.header(f"{APP_ICON} {APP_TITLE}")
    ""
    "Full toolkit for running an AI agent service built with LangGraph, FastAPI and Streamlit"
    with st.popover(":material/settings: Settings", use_container_width=True):
        m = st.radio("LLM to use", options=models.keys())
        model = models[m]
        use_streaming = st.toggle("Stream results", value=True)

    @st.dialog("Architecture")
    def architecture_dialog() -> None:
        st.image(
            "https://github.com/JoshuaC215/agent-service-toolkit/blob/main/media/agent_architecture.png?raw=true"
        )
        "[View full size on Github](https://github.com/JoshuaC215/agent-service-toolkit/blob/main/media/agent_architecture.png)"
        st.caption(
            "App hosted on [Streamlit Cloud](https://share.streamlit.io/) with FastAPI service running in [Azure](https://learn.microsoft.com/en-us/azure/app-service/)"
        )

    if st.button(":material/schema: Architecture", use_container_width=True):
        architecture_dialog()

    with st.popover(":material/policy: Privacy", use_container_width=True):
        st.write(
            "Prompts, responses and feedback in this app are anonymously recorded and saved to LangSmith for product evaluation and improvement purposes only."
        )

    st.markdown(
        f"Thread ID: **{3}**",
        help=f"Set URL query parameter ?thread_id={3} to continue this conversation",
    )

    "[View the source code](https://github.com/JoshuaC215/agent-service-toolkit)"
    st.caption(
        "Made with :material/favorite: by [Joshua](https://www.linkedin.com/in/joshua-k-carroll/) in Oakland"
    )

# Initialize the expander state
if "expander_open" not in st.session_state:
    st.session_state.expander_open = True


# Capture user input from chat input
prompt = st.chat_input()

# Toggle expander state based on user input
if prompt is not None:
    st.session_state.expander_open = False  # Close the expander when the user starts typing

# st write magic
with st.expander(label="Legal Definitions Agent", expanded=st.session_state.expander_open):
    """
    In this example, we're going to be creating our own events handler to stream our [_LangGraph_](https://langchain-ai.github.io/langgraph/)
    invocations with via [`astream_events (v2)`](https://langchain-ai.github.io/langgraph/how-tos/streaming-from-final-node/).
    This one is does not use any callbacks or external streamlit libraries and is asynchronous.
    we've implemented `on_llm_new_token`, a method that run on every new generation of a token from the ChatLLM model, and
    `on_tool_start` a method that runs on every tool call invocation even multiple tool calls, and `on_tool_end` giving final result of tool call.
    """

# Initialize chat messages in session state
if "messages" not in st.session_state:
    st.session_state["messages"] = [AIMessage(content="Hello! I'm the Legal Definitions Agent. What can I define for you?")]

# Loop through all messages in the session state and render them as a chat on every st.refresh mech
for msg in st.session_state.messages:
    # https://docs.streamlit.io/develop/api-reference/chat/st.chat_message
    # we store them as AIMessage and HumanMessage as its easier to send to LangGraph
    if isinstance(msg, AIMessage):
        st.chat_message("assistant").write(msg.content)
    elif isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)

# Handle user input if provided
if prompt:
    st.session_state.messages.append(HumanMessage(content=prompt))
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        # create a placeholder container for streaming and any other events to visually render here
        placeholder = st.container()
        response = asyncio.run(invoke_our_graph(st.session_state.messages, placeholder))
        st.session_state.messages.append(AIMessage(response))