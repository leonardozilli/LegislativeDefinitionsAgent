from langgraph.graph.state import CompiledStateGraph

from LegalDefAgent.src.agent import defagent
from LegalDefAgent.src.agent2 import defagent as defagent2

DEFAULT_AGENT = "LegalDefAgent"

agents: dict[str, CompiledStateGraph] = {
    "LegalDefAgent": defagent,
    "LegalDefAgent2": defagent2
}
