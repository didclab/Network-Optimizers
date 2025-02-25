from fastapi import APIRouter
from app.storage.StorageFactory import StorageFactory

job_metrics_router = APIRouter()
job_metrics_store = StorageFactory.get_metrics_storage()

@job_metrics_router.get("/{owner_id}")
def get_all_job_metrics(owner_id: str):
    job_metrics = {}
    job_uuids = job_metrics_store.list_job_metrics(owner_id)
    for job_uuid_path in job_uuids:
        job_uuid = job_uuid_path.stem
        metrics = job_metrics_store.load_job_metrics(owner_id, job_uuid)
        if metrics:
            job_metrics[job_uuid] = metrics.dict()
    return job_metrics

@job_metrics_router.get("/{owner_id}/{job_uuid}")
def get_job_metrics(owner_id: str, job_uuid: str):
    metrics = job_metrics_store.load_job_metrics(owner_id, job_uuid)
    if metrics:
        return metrics.dict()
    return {"message": "Job metrics not found"}