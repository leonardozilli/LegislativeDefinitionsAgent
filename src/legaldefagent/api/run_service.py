import typer
import uvicorn

from legaldefagent.settings import settings


def run_service(
    host: str = typer.Option(None, help="Host to bind"),
    port: int = typer.Option(None, help="Port to bind"),
    reload: bool = typer.Option(False, help="Enable auto-reload"),
):
    uvicorn.run(
        "legaldefagent.api.service:app",
        host=host or settings.api.host,
        port=port or settings.api.port,
        reload=reload,
    )
