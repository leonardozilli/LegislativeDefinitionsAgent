from dataclasses import dataclass

from langgraph.graph.state import CompiledStateGraph

from ..schema import AgentInfo

from LegalDefAgent.src.agents import definitions_agent, definitions_agent_eval


DEFAULT_AGENT = "LegalDefAgent"

<<<<<<< HEAD

=======
>>>>>>> origin/main
@dataclass
class Agent:
    description: str
    graph: CompiledStateGraph


agents: dict[str, Agent] = {
    "LegalDefAgent": Agent(
        description="A legal assistant for definition Retrieval and Generation", graph=definitions_agent
<<<<<<< HEAD
=======
    ),
    "LegalDefAgentEval": Agent(
        description="A legal assistant for definition Retrieval and Generation. Returns a json output for evaluation purposes.", graph=definitions_agent_eval
>>>>>>> origin/main
    )
}


def get_agent(agent_id: str) -> CompiledStateGraph:
    return agents[agent_id].graph


def get_all_agent_info() -> list[AgentInfo]:
    return [
        AgentInfo(key=agent_id, description=agent.description) for agent_id, agent in agents.items()
<<<<<<< HEAD
    ]
=======
    ]
>>>>>>> origin/main
