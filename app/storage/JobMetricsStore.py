from abc import ABC, abstractmethod
from typing import Optional, List
import os
import json
from pathlib import Path
from app.api.models import JobMetrics
import boto3

class JobMetricsStore(ABC):
    @abstractmethod
    def save_job_metrics(self, owner_id: str, job_uuid: str, metrics: JobMetrics):
        pass

    @abstractmethod
    def load_job_metrics(self, owner_id: str, job_uuid: str) -> Optional[JobMetrics]:
        pass

    @abstractmethod
    def list_job_metrics(self, owner_id: str) -> List[Path]:
        pass

class JobMetricsFileSystemStorage(JobMetricsStore):
    def save_job_metrics(self, owner_id: str, job_uuid: str, metrics: JobMetrics):
        path = f"job_metrics/{owner_id}/"
        os.makedirs(path, exist_ok=True)
        file_path = os.path.join(path, f"{job_uuid}.json")
        with open(file_path, "w") as file:
            json.dump(metrics.dict(), file)

    def load_job_metrics(self, owner_id: str, job_uuid: str) -> Optional[JobMetrics]:
        path = f"job_metrics/{owner_id}/{job_uuid}.json"
        if os.path.exists(path):
            with open(path, "r") as file:
                metrics_data = json.load(file)
                return JobMetrics(**metrics_data)
        return None

    def list_job_metrics(self, owner_id: str) -> List[Path]:
        path = Path(f"job_metrics/{owner_id}/")
        if path.exists():
            return [path / f for f in path.iterdir() if f.is_file()]
        else:
            return []
        
class JobMetricsS3Storage(JobMetricsStore):
    def __init__(self):
        access_key = os.getenv('S3_ACCESS_KEY')
        secret_key = os.getenv('S3_SECRET_KEY')
        region = os.getenv("S3_REGION")
        self.s3 = boto3.resource('s3', aws_access_key_id=access_key, aws_secret_access_key=secret_key,
                                 region_name=region)
        self.bucket_name = os.getenv("S3_BUCKET_NAME")
        self.bucket = self.s3.Bucket(self.bucket_name)

    def save_job_metrics(self, owner_id: str, job_uuid: str, metrics: JobMetrics):
        path = f"job_metrics/{owner_id}/{job_uuid}.json"
        metrics_json = json.dumps(metrics.dict())
        self.bucket.put_object(Key=path, Body=metrics_json)

    def load_job_metrics(self, owner_id: str, job_uuid: str) -> Optional[JobMetrics]:
        path = f"job_metrics/{owner_id}/{job_uuid}.json"
        try:
            obj = self.bucket.Object(path)
            response = obj.get()
            metrics_data = response['Body'].read().decode('utf-8')
            return JobMetrics(**json.loads(metrics_data))
        except self.s3.meta.client.exceptions.NoSuchKey:
            return None
        except Exception as e:
            print(f"Failed to load object: {e}")
            return None

    def list_job_metrics(self, owner_id: str) -> List[Path]:
        path = f"job_metrics/{owner_id}/"
        metrics_paths = []
        for obj in self.bucket.objects.filter(Prefix=path):
            metrics_paths.append(Path(obj.key))
        return metrics_paths