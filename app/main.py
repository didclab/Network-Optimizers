from fastapi import FastAPI
from app.api.optimizer_routes import optimizer_api
from app.api.config_routes import config_router

app = FastAPI()

app.include_router(optimizer_api, tags=['Optimizers'], prefix="/api/optimizer")
app.include_router(config_router, tags=['Configurations'], prefix="/api/configs")


@app.get("/api/health")
def root():
    return {"message": "The API is live."}
