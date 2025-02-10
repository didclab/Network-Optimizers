from fastapi import APIRouter
from app.api.models import TransferJobRequest
from app.storage.TransferJobStore import TransferJobStore

transfer_job_router = APIRouter()
transfer_job_store = TransferJobStore()

@transfer_job_router.post("/")
def add_transfer_job(transfer_job: TransferJobRequest):
    transfer_job_store.save_transfer_job(transfer_job)
    return {"message": "Transfer job added successfully"}

@transfer_job_router.get("/")
def get_transfer_job(owner_id: str):
    return transfer_job_store.load_transfer_job(owner_id)

@transfer_job_router.get("/list")
def list_transfer_jobs():
    return transfer_job_store.list_transfer_jobs()

@transfer_job_router.delete("/")
def delete_transfer_job(owner_id: str):
    success = transfer_job_store.delete_transfer_job(owner_id)
    return {"message": "Transfer job deleted successfully" if success else "Transfer job not found"}