
from fastapi import FastAPI


from .v1.routes import default

def include_routers_v1(app: FastAPI):
    """
    Adds the backend Interface routes to the FastAPI. This is for Interface version 2.
    """
    app.include_router(default.router, prefix="/v1", tags=["chatbot"])
    
    @app.get("/health")
    async def health():
        return {"status": "ok"}

    return None
