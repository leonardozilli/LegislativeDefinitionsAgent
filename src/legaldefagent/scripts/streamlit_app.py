import asyncio
import logging
import urllib
from collections.abc import AsyncGenerator

import streamlit as st
from dotenv import load_dotenv
from pydantic import ValidationError
from streamlit.runtime.scriptrunner import get_script_run_ctx

from agent_service_toolkit.src.client.client import AgentClient
from agent_service_toolkit.src.schema.schema import ChatHistory, ChatMessage
from legaldefagent.schema.task_data import TaskData, TaskDataStatus
from legaldefagent.settings import settings

logger = logging.getLogger(__name__)

APP_TITLE = "Legal Definitions Agent"
APP_ICON = "🧰"

INTERMEDIATE_NODES = [
    "__start__",
    "eurlex_agent",
    "extract_query",
    "filter_definitions",
    "normattiva_agent",
    "pdl_agent",
    "pick_definition",
    "query_vectorstore",
    "task_manager",
]


async def main() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon=APP_ICON,
        menu_items={},
    )

    # Hide the streamlit upper-right chrome
    # st.html(
    # """
    # <style>
    # [data-testid="stStatusWidget"] {
    # visibility: hidden;
    # height: 0%;
    # position: fixed;
    # }
    # </style>
    # """,
    # )
    # if st.get_option("client.toolbarMode") != "minimal":
    # st.set_option("client.toolbarMode", "minimal")
    # await asyncio.sleep(0.1)
    # st.rerun()

    if "expander_open" not in st.session_state:
        st.session_state.expander_open = True

    st.header(f"{APP_TITLE}")

    with st.expander(label="Legal Definitions Agent", expanded=st.session_state.expander_open):
        """
        This chatbot can help you retrieve legal definitions from various sources.\n
        Ask for the definition of a legal term, and the chatbot will provide you with the most relevant definition, or generate a novel one.\n
        You can restrict the search to a specific source, such as the European Union's legal database (EUR-Lex), the Italian legal database (Normattiva), or the Italian Parliament's legislative proposals (PDLs), or specify a time period for the search by simply mentioning it in your query.
        """

    if "agent_client" not in st.session_state:
        load_dotenv()
        api_url = f"http://{settings.API_HOST}:{settings.API_PORT}"
        st.session_state.agent_client = AgentClient(base_url=api_url)
    agent_client: AgentClient = st.session_state.agent_client

    if "thread_id" not in st.session_state:
        thread_id = st.query_params.get("thread_id")
        if not thread_id:
            thread_id = get_script_run_ctx().session_id
            messages = []
        else:
            history: ChatHistory = agent_client.get_history(thread_id=thread_id)
            messages = history.messages
        st.session_state.messages = messages
        st.session_state.thread_id = thread_id

    # models = {v: k for k, v in _MODEL_TABLE.items()}

    # Config options
    with st.sidebar:
        st.header(f"{APP_TITLE}")
        ""
        "Legal Definitions Retrieval and Generation Agent"
        with st.popover(":material/settings: Settings", use_container_width=True):
            model_idx = agent_client.info.models.index(agent_client.info.default_model)
            # model = st.radio("LLM to use", options=agent_client.info.models, index=model_idx)
            model = st.selectbox("LLM to use", options=agent_client.info.models, index=model_idx)
            # agent_list = [a.key for a in agent_client.info.agents]
            # agent_idx = agent_list.index(agent_client.info.default_agent)
            # agent_client.agent = st.selectbox(
            # "Agent to use",
            # options=agent_list,
            # index=agent_idx,
            # )
            use_streaming = True  # st.toggle("Stream results", value=True)
            st.slider(
                "Number of definitions retrieved",
                min_value=1,
                max_value=30,
                value=10,
                step=1,
                label_visibility="visible",
            )

        @st.dialog("Agent Architecture")
        def architecture_dialog() -> None:
            st.image("docs/imgs/agent_architecture.png")

        if st.button(":material/schema: Architecture", use_container_width=True):
            architecture_dialog()

        @st.dialog("Retrieval pipeline")
        def pipeline_dialog() -> None:
            st.image("docs/imgs/definition_search_pipeline.png")

        if st.button(":material/schema: Retrieval Pipeline", use_container_width=True):
            pipeline_dialog()

        with st.popover(":material/policy: Privacy", use_container_width=True):
            st.write(
                "Prompts, responses and feedback in this app are anonymously recorded and saved to LangSmith for product evaluation and improvement purposes only."
            )

        @st.dialog("Share/resume chat")
        def share_chat_dialog() -> None:
            session = st.runtime.get_instance()._session_mgr.list_active_sessions()[0]
            st_base_url = urllib.parse.urlunparse(
                [
                    session.client.request.protocol,
                    session.client.request.host,
                    "",
                    "",
                    "",
                    "",
                ]
            )
            # if it's not localhost, switch to https by default
            if not st_base_url.startswith("https") and "localhost" not in st_base_url:
                st_base_url = st_base_url.replace("http", "https")
            chat_url = f"{st_base_url}?thread_id={st.session_state.thread_id}"
            st.markdown(f"**Chat URL:**\n```text\n{chat_url}\n```")
            st.info("Copy the above URL to share or revisit this chat")

        if st.button(":material/upload: Share/resume chat", use_container_width=True):
            share_chat_dialog()

        st.markdown(
            f"Thread ID: **{st.session_state.thread_id}**",
            help=f"Set URL query parameter ?thread_id={st.session_state.thread_id} to continue this conversation",
        )

        "[Source code](https://github.com/leonardozilli/LegalDefAgent)"
        st.caption("CIRSFID-Alma AI")

    # Draw existing messages
    messages: list[ChatMessage] = st.session_state.messages

    def close_expander():
        st.session_state.expander_open = False

    prompt = st.chat_input(on_submit=close_expander)

    # draw_messages() expects an async iterator over messages
    async def amessage_iter() -> AsyncGenerator[ChatMessage, None]:
        for m in messages:
            yield m

    await draw_messages(amessage_iter())

    # Generate new message if the user provided new input
    if user_input := prompt:
        messages.append(ChatMessage(type="human", content=user_input))
        st.chat_message("human").write(user_input)
        if use_streaming:
            stream = agent_client.astream(
                message=user_input,
                model=model,
                thread_id=st.session_state.thread_id,
            )
            await draw_messages(stream, is_new=True)
        else:
            response = await agent_client.ainvoke(
                message=user_input,
                model=model,
                thread_id=st.session_state.thread_id,
            )
            messages.append(response)
            st.chat_message("ai").write(response.content)
        st.rerun()  # Clear stale containers

    # If messages have been generated, show feedback widget
    if len(messages) > 0:
        with st.session_state.last_message:
            await handle_feedback()


async def draw_messages(
    messages_agen: AsyncGenerator[ChatMessage | str, None],
    is_new: bool = False,
) -> None:
    """
    Draws a set of chat messages - either replaying existing messages
    or streaming new ones.

    This function has additional logic to handle streaming tokens and tool calls.
    - Use a placeholder container to render streaming tokens as they arrive.
    - Use a status container to render tool calls. Track the tool inputs and outputs
      and update the status container accordingly.

    The function also needs to track the last message container in session state
    since later messages can draw to the same container. This is also used for
    drawing the feedback widget in the latest chat message.

    Args:
        messages_aiter: An async iterator over messages to draw.
        is_new: Whether the messages are new or not.
    """

    # Keep track of the last message container
    last_message_type = None
    st.session_state.last_message = None
    last_node_name = INTERMEDIATE_NODES[0]

    # Placeholder for intermediate streaming tokens
    streaming_content = ""
    streaming_placeholder = None

    # Iterate over the messages and draw them
    while msg := await anext(messages_agen, None):
        # if msg.custom_data["node_name"] in INTERMEDIATE_NODES:
        # pass
        # str message represents an intermediate token being streamed
        if isinstance(msg, str):
            # Skip streaming content for intermediate nodes
            if last_node_name in INTERMEDIATE_NODES:
                continue
            # If placeholder is empty, this is the first token of a new message
            # being streamed. We need to do setup.
            if not streaming_placeholder:
                if last_message_type != "ai":
                    last_message_type = "ai"
                    st.session_state.last_message = st.chat_message("ai")
                with st.session_state.last_message:
                    streaming_placeholder = st.empty()

            streaming_content += msg
            streaming_placeholder.write(streaming_content)
            continue
        if not isinstance(msg, ChatMessage):
            st.error(f"Unexpected message type: {type(msg)}")
            st.write(msg)
            st.stop()
        match msg.type:
            case "tool":
                if msg.status == "error":
                    st.error(f"Tool encountered an error:\n{msg.content}")
                    st.stop()
                continue
            # A message from the user, the easiest case
            case "human":
                last_message_type = "human"
                st.chat_message("human").write(msg.content)

            # A message from the agent is the most complex case, since we need to
            # handle streaming tokens and tool calls.
            case "ai":
                if msg.tool_calls:
                    continue

                # If we're rendering new messages, store the message in session state
                if is_new:
                    st.session_state.messages.append(msg)

                # If the last message type was not AI, create a new chat message
                if last_message_type != "ai":
                    last_message_type = "ai"
                    st.session_state.last_message = st.chat_message("ai")

                with st.session_state.last_message:
                    # If the message has content, write it out.
                    # Reset the streaming variables to prepare for the next message.
                    if msg.content:
                        if streaming_placeholder:
                            streaming_placeholder.write(msg.content)
                            streaming_content = ""
                            streaming_placeholder = None
                        else:
                            for chunk in msg.content.split("\n"):
                                st.write(chunk)

            case "custom":
                # CustomData example used by the bg-task-agent
                # See:
                # - src/agents/utils.py CustomData
                # - src/agents/bg_task_agent/task.py
                try:
                    task_data: TaskData = TaskData.model_validate(msg.custom_data)
                except ValidationError:
                    st.error("Unexpected CustomData message received from agent")
                    st.write(msg.custom_data)
                    st.stop()

                if is_new:
                    st.session_state.messages.append(msg)

                if last_message_type != "task":
                    last_message_type = "task"
                    st.session_state.last_message = st.chat_message(name="task", avatar=":material/manufacturing:")
                    with st.session_state.last_message:
                        status = TaskDataStatus()

                status.add_and_draw_task_data(task_data)

            # In case of an unexpected message type, log an error and stop
            case _:
                st.error(f"Unexpected ChatMessage type: {msg.type}")
                st.write(msg)
                st.stop()


async def handle_feedback() -> None:
    """Draws a feedback widget and records feedback from the user."""

    # Keep track of last feedback sent to avoid sending duplicates
    if "last_feedback" not in st.session_state:
        st.session_state.last_feedback = (None, None)

    latest_run_id = st.session_state.messages[-1].run_id
    feedback = st.feedback("stars", key=latest_run_id)

    # If the feedback value or run ID has changed, send a new feedback record
    if feedback is not None and (latest_run_id, feedback) != st.session_state.last_feedback:
        # Normalize the feedback value (an index) to a score between 0 and 1
        normalized_score = (feedback + 1) / 5.0

        agent_client: AgentClient = st.session_state.agent_client
        await agent_client.acreate_feedback(
            run_id=latest_run_id,
            key="human-feedback-stars",
            score=normalized_score,
            kwargs={"comment": "In-line human feedback"},
        )
        st.session_state.last_feedback = (latest_run_id, feedback)
        st.toast("Feedback recorded", icon=":material/reviews:")


if __name__ == "__main__":
    asyncio.run(main())
