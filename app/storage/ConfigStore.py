import json
from abc import ABC, abstractmethod
import os

import boto3

from app.api.models import GlobalConfig, ModelType, SGDTrainingConfig, BOTrainingConfig, DDPGTrainingConfig, \
    PPOTrainingConfig


class ConfigStore(ABC):

    @abstractmethod
    def create_config(self, global_config: GlobalConfig, owner_id: str):
        pass

    @abstractmethod
    def delete_config(self, config_name: str, model_type: ModelType, owner_id: str):
        pass

    @abstractmethod
    def list_config(self, model_type: ModelType, owner_id: str):
        pass

    @abstractmethod
    def get_config(self, model_type: ModelType, owner_id: str, config_name: str):
        pass


class ConfigFileSystemStorage(ConfigStore):

    def create_config(self, global_config: GlobalConfig, owner_id: str):
        path = f"config/{owner_id}/{global_config.modelType}/"
        config_dict = global_config.dict()
        json.dumps(config_dict)
        os.makedirs(path, exist_ok=True)
        file_path = os.path.join(path, f"{global_config.configName}.json")
        with open(file_path, "w") as file:
            json.dump(config_dict, file)
        return file_path

    def delete_config(self, config_name: str, model_type: ModelType, owner_id: str):
        path = f"config/{owner_id}/{model_type}/{config_name}.json"
        if os.path.exists(path):
            os.remove(path)

    def list_config(self, model_type: ModelType, owner_id: str):
        path = f"config/{owner_id}/{model_type}/"
        if os.path.exists(path):
            files = os.listdir(path)
            file_names = [f for f in files if os.path.isfile(os.path.join(path, f))]
            return file_names
        else:
            return []

    def get_config(self, model_type: ModelType, owner_id: str, config_name: str):
        path = f"config/{owner_id}/{model_type}/{config_name}.json"
        if os.path.exists(path):
            with open(path, "r") as file:
                config_data = json.load(file)
                return ConfigFactory.create_config(model_type=model_type,
                                                   config_data=config_data)
        else:
            return None


class ConfigS3Storage(ConfigStore):

    def __init__(self):
        access_key = os.getenv('S3_ACCESS_KEY')
        secret_key = os.getenv('S3_SECRET_KEY')
        region = os.getenv("S3_REGION")
        self.s3 = boto3.resource('s3', aws_access_key_id=access_key, aws_secret_access_key=secret_key,
                                 region_name=region)
        self.bucket_name = os.getenv("S3_BUCKET_NAME")
        self.bucket = self.s3.Bucket(self.bucket_name)

    def create_config(self, global_config: GlobalConfig, owner_id: str):
        path = f"config/{owner_id}/{global_config.modelType}/{global_config.configName}.json"
        config_dict = global_config.dict()
        config_json = json.dumps(config_dict)
        self.s3.Bucket(self.bucket_name).put_object(Key=path, Body=config_json)

    def delete_config(self, config_name: str, model_type: ModelType, owner_id: str):
        path = f"config/{owner_id}/{model_type}/{config_name}.json"
        self.bucket.Object(path).delete()

    def list_config(self, model_type: ModelType, owner_id: str):
        path = f"config/{owner_id}/{model_type}/"
        bucket = self.s3.Bucket(self.bucket_name)
        file_keys = []
        for object_summary in bucket.objects.filter(Prefix=path):
            file_keys.append(object_summary.key)

        return file_keys

    def get_config(self, model_type: ModelType, owner_id: str, config_name: str):
        path = f"config/{owner_id}/{model_type}/{config_name}.json"
        obj = self.bucket.Object(path)
        response = obj.get()
        config_json = response['Body'].read().decode('utf-8')
        config_dict = json.loads(config_json)
        return ConfigFactory.create_config(model_type=config_dict['modelType'],
                                           config_data=config_dict)


class ConfigFactory:

    @staticmethod
    def create_config(model_type: ModelType, config_data: dict) -> GlobalConfig:
        if model_type == ModelType.ddpg:
            return DDPGTrainingConfig(**config_data)
        elif model_type == ModelType.ppo:
            return PPOTrainingConfig(**config_data)
        elif model_type == ModelType.bo:
            return BOTrainingConfig(**config_data)
        elif model_type == ModelType.sgd:
            return SGDTrainingConfig(**config_data)
        # Add more conditions for other configurations as needed
        else:
            raise ValueError("Invalid modelType or configType")
