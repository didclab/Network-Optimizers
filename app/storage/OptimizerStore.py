import os.path
import tempfile
from abc import ABC, abstractmethod

from app.api.models import GlobalConfig, ModelType
from stable_baselines3.common.base_class import BaseAlgorithm
from pathlib import Path
from fastapi import UploadFile, File
from fastapi.responses import FileResponse
import boto3


class OptimizerStore(ABC):

    @abstractmethod
    def save_model(self, owner_id: str, modelType: ModelType, modelName: str, base_algo: BaseAlgorithm):
        """
        Saves the model using the path format owner_id/modelType/modelName.ex
        :param owner_id:
        :param modelType:
        :param modelName:
        :param base_algo:
        :return:
        """
        pass

    @abstractmethod
    def load_model(self, owner_id: str, modelType: ModelType, modelName: str, ) -> Path:
        """
        Load the model from the following path format: owner_id/modelType/modelName
        """
        pass

    @abstractmethod
    def list_optimizers(self, model_type: ModelType, owner_id: str) -> []:
        """
        List all the optimizers in the following path format: owner_id/modelType
        :param model_type: If none then search through all model types if the paths exist.
        :param owner_id:
        :return:
        """
        pass

    @abstractmethod
    def delete_optimizer(self, model_type: ModelType, owner_id: str, model_name: str) -> bool:
        """
        Delete the model in the path format: owner_id/modelType/model_name
        :param model_type:
        :param owner_id:
        :param model_name:
        :return:
        """
        pass

    @abstractmethod
    def download_optimizer(self, model_type: ModelType, owner_id: str, model_name: str) -> FileResponse:
        """
        Download the specfied model to the user end. This allows users to get the model they have trained/evaluated/tuned on
        :param model_type:
        :param owner_id:
        :param model_name:
        :return:
        """
        pass

    @abstractmethod
    def upload_optimizer(self, model_type: ModelType, owner_id: str, model_name: str, file: UploadFile = File(...)) -> Path:
        """
        Currenly not sure how this will work but the idea is that users are able to upload their own optimizers(Pytorch I think so far) and use the specfied optimizer for transfers
        :param model_type:
        :param owner_id:
        :param model_name:
        :param file:
        :return:
        """
        pass


class FileSystemOptimizerStore(OptimizerStore):

    def save_model(self, owner_id: str, modelType: ModelType, modelName: str, base_algo: BaseAlgorithm):
        path = f"model/{owner_id}/{modelType}/{modelName}"
        base_algo.save(path=path)

    def load_model(self, owner_id: str, modelType: ModelType, modelName: str, ) -> Path:
        path = f"model/{owner_id}/{modelType}/{modelName}.zip"
        if os.path.exists(path):
            return Path(path)
        else:
            return Path('')

    def list_optimizers(self, model_type: ModelType, owner_id: str) -> []:
        path = f"model/{owner_id}/{model_type}/"
        if os.path.exists(path):
            files = os.listdir(path)
            file_names = [f for f in files if os.path.isfile(os.path.join(path, f))]
            return file_names
        else:
            return []

    def delete_optimizer(self, model_type: ModelType, owner_id: str, model_name: str) -> bool:
        path = f"model/{owner_id}/{model_type}/{model_name}.zip"
        if os.path.exists(path):
            os.remove(path)
            return True
        else:
            return False

    def download_optimizer(self, model_type: ModelType, owner_id: str, model_name: str, ) -> FileResponse:
        path = f"model/{owner_id}/{model_type}/{model_name}.zip"
        return FileResponse(path)

    def upload_optimizer(self, model_type: str, owner_id: str, model_name: str, file: UploadFile = File(...)) -> Path:
        path = f"model/{owner_id}/{model_type}/{model_name}.zip"
        file_path = Path(path)
        with open(file_path, "wb") as file_out:
            file_out.write(file.file.read())
        return file_path


class S3OptimizerStorage(OptimizerStore):

    def __init__(self):
        access_key = os.getenv('S3_ACCESS_KEY')
        secret_key = os.getenv('S3_SECRET_KEY')
        region = os.getenv("S3_REGION")
        self.s3 = boto3.resource('s3', aws_access_key_id=access_key, aws_secret_access_key=secret_key,
                                 region_name=region)
        self.bucket_name = os.getenv("S3_BUCKET_NAME")
        self.bucket = self.s3.Bucket(self.bucket_name)

    def save_model(self, owner_id: str, modelType: ModelType, modelName: str, base_algo: BaseAlgorithm):
        path = f"model/{owner_id}/{modelType}/{modelName}"
        base_algo.save(path=path)
        path += ".zip"
        with open(path, 'rb') as f:
            self.bucket.put_object(Key=path, Body=f)

    def load_model(self, owner_id: str, modelType: ModelType, modelName: str,) -> Path:
        key = f"model/{owner_id}/{modelType}/{modelName}.zip"
        file_path = f"models/{modelType}/{modelName}.zip"  # Specify download path
        self.s3.Bucket(self.bucket_name).download_file(key, file_path)
        return Path(file_path)

    def list_optimizers(self, model_type: ModelType, owner_id: str) -> []:
        path = f"model/{owner_id}/{model_type}/"
        optimizer_names = []
        for obj in self.bucket.objects.filter(Prefix=path):
            optimizer_names.append(obj.key.rsplit("/", 1)[-1])
        return optimizer_names

    def delete_optimizer(self, model_type: ModelType, owner_id: str, model_name: str) -> bool:
        key = f"model/{owner_id}/{model_type}/{model_name}.zip"
        try:
            self.bucket.Object(key).delete()
            return True
        except Exception as e:
            print(f"Failed to delete object: {e}")
            return False
        pass

    def download_optimizer(self, model_type: ModelType, owner_id: str, model_name: str) -> FileResponse:
        key = f"model/{owner_id}/{model_type}/{model_name}.zip"
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            file_path = tmp_file.name
            self.bucket.download_file(key, file_path)
            file_response = FileResponse(file_path, media_type="application/octet-stream", filename=model_name)
        return file_response

    def upload_optimizer(self, model_type: ModelType, owner_id: str, model_name: str, file: UploadFile = File(...)) -> Path:
        key = f"model/{owner_id}/{model_type}/{model_name}.zip"
        file_path = f"temp/{model_name}"  # Temporarily save the file locally
        with open(file_path, "wb") as f:
            f.write(file.file.read())

        with open(file_path, 'rb') as f:
            self.s3.put_object(Bucket=self.bucket_name, Key=key, Body=f)

        os.remove(file_path)  # Delete the temporary file
        return Path(key)
