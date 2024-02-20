from fastapi import APIRouter
from app.api.models import GlobalConfig, TrainConfig, TuneConfig, EvaluateConfig, DDPGTrainingConfig, A2CTrainingConfig, \
    PPOTrainingConfig, ModelType
from typing import Union
from app.storage.StorageFactory import StorageFactory

config_router = APIRouter()
config_store = StorageFactory.get_config_storage()


@config_router.get("/")
def get_config(config_name: str, owner_id: str, model_type: ModelType):
    return config_store.get_config(config_name=config_name, owner_id=owner_id, model_type=model_type)


@config_router.post("/")
def create_config(config: Union[
    GlobalConfig, TrainConfig, TuneConfig, EvaluateConfig, DDPGTrainingConfig, A2CTrainingConfig, PPOTrainingConfig],
                  owner_id: str):
    return config_store.create_config(global_config=config, owner_id=owner_id)


@config_router.delete("/")
def delete_config(config_name: str, owner_id: str, model_type: ModelType):
    return config_store.delete_config(config_name=config_name, owner_id=owner_id, model_type=model_type)


@config_router.get("/list")
def list_config(model_type: ModelType, owner_id: str):
    return config_store.list_config(model_type=model_type, owner_id=owner_id)
