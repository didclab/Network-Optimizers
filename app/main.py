from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from app.api.optimizer_routes import optimizer_api
from app.api.config_routes import config_router
from app.api.transfer_job_routes import transfer_job_router
from app.api.metrics_routes import job_metrics_router

app = FastAPI()

templates = Jinja2Templates(directory="../job-metrics-visualization/build")
app.mount('/static', StaticFiles(directory="../job-metrics-visualization/build/static"), 'static')

app.include_router(optimizer_api, tags=['Optimizers'], prefix="/api/optimizer")
app.include_router(config_router, tags=['Configurations'], prefix="/api/configs")
app.include_router(transfer_job_router, tags=['Transfer Jobs'], prefix="/api/transfer_jobs")
app.include_router(job_metrics_router, tags=['Job Metrics'], prefix="/api/metrics")

@app.get("/api/health")
def root():
    return {"message": "The API is live."}

@app.get("/metrics/{rest_of_path:path}")
async def react_app(req: Request, rest_of_path: str):
    return templates.TemplateResponse('index.html', { 'request': req })