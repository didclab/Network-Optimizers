from fastapi import APIRouter, BackgroundTasks, UploadFile, File
from app.api.models import CreateOptimizerRequest, TrainRequest, EvaluateRequest, ModelType
from app.optimizers.runner_factory import RunnerFactory
from app.optimizers.EvaluateRunner import EvaluateRunner
from typing import Dict, Type, TypeVar
from app.storage.StorageFactory import StorageFactory

T = TypeVar('T', bound='BaseRunner')

optimizer_api = APIRouter()
RunnerMap: Dict[str, Type[T]] = {}

storage = StorageFactory.get_optimizer_storage()


@optimizer_api.post("/train", status_code=200)
async def train_optimizer(trainRequest: TrainRequest, background_tasks: BackgroundTasks) -> None:
    runner = RunnerFactory.create_runner(train_request=trainRequest, storage=storage)
    runner.load_model()
    background_tasks.add_task(runner.train())


@optimizer_api.post("/evaluate", status_code=200)
async def evaluate_optimizer(evaluateRequest: EvaluateRequest, background_tasks: BackgroundTasks) -> None:
    evalRunner = EvaluateRunner(evaluate_request=evaluateRequest, model_persistence=storage)
    evalRunner.load_model()
    background_tasks.add_task(evalRunner.evaluate)


@optimizer_api.post("/tune")
async def tune_transfer(create_request: CreateOptimizerRequest):
    """
    Scheduler hits this to create the requested optimizer. Then the optimizer blocks till it sees the transfer has started with that jobUuid
    Once the optimizer see it, we optimize till the job ends. No need for env, this assumes the model is fully trained and simply loads an optimizer stored
    :return:
    """

    pass


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
