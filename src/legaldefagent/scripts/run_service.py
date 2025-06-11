import typer
import uvicorn

from legaldefagent.settings import settings


def run_service(
    host: str = typer.Option(None, help="Host to bind"),
    port: int = typer.Option(None, help="Port to bind"),
    reload: bool = typer.Option(False, help="Enable auto-reload"),
):
    uvicorn.run(
        "agent_service_toolkit.src.service.service:app",
        host=host or settings.API_HOST,
        port=port or settings.API_PORT,
        reload=reload,
    )
