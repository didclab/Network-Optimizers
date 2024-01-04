from fastapi import FastAPI
from app.api.routers import optimizer_api

app = FastAPI()

app.include_router(optimizer_api)
