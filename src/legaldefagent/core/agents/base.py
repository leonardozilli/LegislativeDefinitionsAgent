"""Base agent components shared between production and evaluation agents."""

import logging
from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage, trim_messages
from langchain_core.runnables import RunnableConfig, RunnableLambda, RunnableSerializable
from langgraph.graph import MessagesState
from langgraph.managed import RemainingSteps

from legaldefagent.core.providers import get_model
from legaldefagent.settings import settings


logger = logging.getLogger(__name__)


class AgentState(MessagesState, total=False):
    remaining_steps: RemainingSteps


class EvalAgentState(AgentState, total=False):
    response: dict


def wrap_model(model: BaseChatModel, tools: list, instructions: str) -> RunnableSerializable:
    model = model.bind_tools(tools)
    preprocessor = RunnableLambda(
        lambda state: [SystemMessage(content=instructions)] + state["messages"],
        name="StateModifier",
    )
    return preprocessor | model


def call_model(state: AgentState, config: RunnableConfig, tools: list, instructions: str) -> AgentState:
    try:
        m = get_model(config["configurable"].get("model", settings.default_model))
    except ValueError as e:
        raise ValueError(f"Could not configure model: {e}")
    model_runnable = wrap_model(m, tools, instructions)
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


def state_cleanup(state: AgentState) -> AgentState:
    state["messages"] = trim_messages(state["messages"], token_counter=len, strategy="last", max_tokens=1)
    state["messages"] = state["messages"][-1:]
    return {"messages": ""}


def pending_tool_calls(state: AgentState) -> Literal["tools", "done"]:
    """
    Check if there are pending tool calls in the last message.

    Args:
        state: Current agent state

    Returns:
        "tools" if there are pending tool calls, "done" otherwise
    """
    last_message = state["messages"][-1]
    if not isinstance(last_message, AIMessage):
        raise TypeError(f"Expected AIMessage, got {type(last_message)}")
    if last_message.tool_calls:
        return "tools"
    return "done"


def return_output(state: EvalAgentState) -> dict:
    """
    Extract the final output for evaluation.

    Args:
        state: Current evaluation agent state

    Returns:
        Dictionary with the response message
    """
    return {"response": state["messages"][-1]}
