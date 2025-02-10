from abc import ABC, abstractmethod
from typing import List
import os
import json
from pathlib import Path
import boto3
from app.api.models import TransferJobRequest
from app.storage.TransferJobStore import TransferJobStore

class TransferJobStore(ABC):
    @abstractmethod
    def save_transfer_job(self, transfer_job: TransferJobRequest):
        pass

    @abstractmethod
    def load_transfer_job(self, owner_id: str, job_uuid: str) -> TransferJobRequest:
        pass

    @abstractmethod
    def list_transfer_jobs(self, owner_id: str) -> List[Path]:
        pass

    @abstractmethod
    def delete_transfer_job(self, owner_id: str) -> bool:
        pass

class TransferJobFileSystemStorage(TransferJobStore):
    def save_transfer_job(self, transfer_job: TransferJobRequest):
        path = f"transfer_jobs/{transfer_job.ownerId}/"
        os.makedirs(path, exist_ok=True)
        file_path = os.path.join(path, f"{transfer_job.jobUuid}.json")
        with open(file_path, "w") as file:
            json.dump(transfer_job.dict(), file)

    def load_transfer_job(self, owner_id: str, job_uuid: str) -> TransferJobRequest:
        path = f"transfer_jobs/{owner_id}/{job_uuid}.json"
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as file:
                job_json = file.read()
                return TransferJobRequest(**json.loads(job_json))
        return None
    
    def list_transfer_jobs(self, owner_id: str) -> List[Path]:
        path = Path(f"transfer_jobs/{owner_id}/")
        if path.exists():
            return [path / f for f in path.iterdir() if f.is_file()]
        else:
            return []

    def delete_transfer_job(self, owner_id: str, job_uuid: str) -> bool:
        path = f"transfer_jobs/{owner_id}/{job_uuid}.json"
        if os.path.exists(path):
            os.remove(path)
            return True
        return False
    
class TransferJobS3Storage(TransferJobStore):
    def __init__(self):
        access_key = os.getenv('S3_ACCESS_KEY')
        secret_key = os.getenv('S3_SECRET_KEY')
        region = os.getenv("S3_REGION")
        self.s3 = boto3.resource('s3', aws_access_key_id=access_key, aws_secret_access_key=secret_key,
                                 region_name=region)
        self.bucket_name = os.getenv("S3_BUCKET_NAME")
        self.bucket = self.s3.Bucket(self.bucket_name)

    def save_transfer_job(self, transfer_job: TransferJobRequest):
        path = f"transfer_jobs/{transfer_job.ownerId}/{transfer_job.jobUuid}.json"
        job_json = json.dumps(transfer_job.dict())
        self.bucket.put_object(Key=path, Body=job_json)

    def load_transfer_job(self, owner_id: str, job_uuid: str) -> TransferJobRequest:
        path = f"transfer_jobs/{owner_id}/{job_uuid}.json"
        try:
            obj = self.bucket.Object(path)
            response = obj.get()
            job_json = response['Body'].read().decode('utf-8')
            return TransferJobRequest(**json.loads(job_json))
        except self.s3.meta.client.exceptions.NoSuchKey:
            return None
        except Exception as e:
            print(f"Failed to load object: {e}")
            return None

    def list_transfer_jobs(self, owner_id: str) -> List[Path]:
        path = f"transfer_jobs/{owner_id}/"
        job_paths = []
        for obj in self.bucket.objects.filter(Prefix=path):
            job_paths.append(Path(obj.key))
        return job_paths

    def delete_transfer_job(self, owner_id: str) -> bool:
        path = f"transfer_jobs/{owner_id}/"
        try:
            for obj in self.bucket.objects.filter(Prefix=path):
                obj.delete()
            return True
        except Exception as e:
            print(f"Failed to delete objects: {e}")
            return False