import uvicorn

from LegalDefAgent.src.settings import settings


if __name__ == "__main__":
    uvicorn.run("LegalDefAgent.agent_service_toolkit.src.service.service:app", host=settings.HOST, port=settings.PORT, reload=settings.is_dev())
