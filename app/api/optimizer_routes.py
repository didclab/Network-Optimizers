from fastapi import APIRouter, BackgroundTasks, UploadFile, File
from app.api.models import ModelType, TransferJobRequest, OptimizerFunctionType, EvaluateConfig, TuneConfig
from app.optimizers.RunnerFactory import RunnerFactory
from app.optimizers.EvaluateRunner import EvaluateRunner
from app.optimizers.TuneRunner import TuneRunner
from typing import Dict, Type, TypeVar, List
from app.storage.StorageFactory import StorageFactory

T = TypeVar('T', bound='BaseRunner')

optimizer_api = APIRouter()
RunnerMap: Dict[str, Type[T]] = {}

storage = StorageFactory.get_optimizer_storage()
config_store = StorageFactory.get_config_storage()
transfer_jobs_store = StorageFactory.get_transfer_job_storage()
job_metrics_store = StorageFactory.get_metrics_storage()


@optimizer_api.post("/optimize", status_code=200)
async def optimize_transfer(transfer_job_uuids: List[str], background_tasks: BackgroundTasks) -> None:
    for job_uuid in transfer_job_uuids:
        transferRequest = transfer_jobs_store.load_transfer_job(job_uuid)
        optimizerOptions = transferRequest.optimizerOptions
        config = config_store.get_config(model_type=optimizerOptions.modelType,
                                        config_name=optimizerOptions.config_name,
                                        owner_id=transferRequest.ownerId)

        if optimizerOptions.optimizerRequestType == OptimizerFunctionType.TRAIN:
            runner = RunnerFactory.create_runner(transfer_request=transferRequest, storage=storage, config=config)
            runner.load_model()
            background_tasks.add_task(runner.train())
        elif optimizerOptions.optimizerRequestType == OptimizerFunctionType.EVALUATE:
            eval_runner = EvaluateRunner(transfer_request=transferRequest, model_store=storage, metrics_store=job_metrics_store,
                                        config=EvaluateConfig(**config.dict()))
            eval_runner.load_model()
            background_tasks.add_task(eval_runner.evaluate())

        elif optimizerOptions.optimizerRequestType == OptimizerFunctionType.TUNE:
            tune_runner = TuneRunner(transfer_request=transferRequest, model_store=storage,
                                    tune_config=TuneConfig(**config.dict()))
            tune_runner.load_model()
            background_tasks.add_task(tune_runner.tune_transfer())


@optimizer_api.get("/download", status_code=200)
async def download_optimizer(ownerId: str, modelName: str, modelType: ModelType):
    return storage.download_optimizer(model_type=modelType, owner_id=ownerId, model_name=modelName)


@optimizer_api.post("/upload", status_code=200)
async def upload_optimizer(modelName: str, modelType: ModelType, ownerId: str, file: UploadFile = File(...)):
    return storage.upload_optimizer(model_type=modelType, model_name=modelName, owner_id=ownerId, file=file)


@optimizer_api.get("/list")
async def list_optimizers(ownerId: str, model_type: ModelType):
    return storage.list_optimizers(owner_id=ownerId, model_type=model_type)


@optimizer_api.delete("/rm")
async def rm_optimizer(ownerId: str, modelName: str, model_type: ModelType):
    return storage.delete_optimizer(owner_id=ownerId, model_name=modelName, model_type=model_type)
