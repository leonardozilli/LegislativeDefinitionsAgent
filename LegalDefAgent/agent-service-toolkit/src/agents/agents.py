from langgraph.graph.state import CompiledStateGraph

from LegalDefAgent.src.agent import LegalDefAgent
from LegalDefAgent.src.models import _get_model

DEFAULT_AGENT = "LegalDefAgent"

agents: dict[str, CompiledStateGraph] = {
    "LegalDefAgent": LegalDefAgent(model=_get_model('groq', streaming=True)).graph_runnable
}
