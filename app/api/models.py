import uuid

from pydantic import BaseModel


class CreateOptimizerRequest(BaseModel):
    nodeId: str
    maxConcurrency: int
    maxParallelism: int
    maxPipelining: int
    maxChunkSize: int
    optimizerType: str
    fileCount: int
    jobId: int
    dbType: str
    jobUuid: str
    userId: str


class DeleteOptimizerRequest(BaseModel):
    nodeId: str
