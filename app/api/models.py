from enum import Enum
from typing import List, Optional

from pydantic import BaseModel


class EntityInfo(BaseModel):
    id: str
    path: str
    size: int
    chunkSize: int


class FileSource(BaseModel):
    credId: str
    type: str
    fileSourcePath: str
    fileList: List[EntityInfo] = []


class FileDestination(BaseModel):
    credId: str
    type: str
    fileDestinationPath: str


class TransferOptions(BaseModel):
    compress: bool
    encrypt: bool
    optimizer: str
    overwrite: bool
    retry: int
    verify: str
    concurrencyThreadCount: int
    parallelThreadCount: int
    pipeSize: int
    maxConcurrency: Optional[int] = 32
    maxParallelism: Optional[int] = 32


class ModelType(str, Enum):
    ddpg = "DDPG"
    ppo = "PPO"
    bo = "BO"
    sgd = "SGD"


class OptimizerFunctionType(Enum):
    TRAIN = "TRAIN"
    EVALUATE = "EVALUATE"
    TUNE = "TUNE"


class OptimizerOptions(BaseModel):
    modelType: ModelType
    modelName: str
    optimizerRequestType: OptimizerFunctionType
    config_name: str


class TransferJobRequest(BaseModel):
    ownerId: str
    transferNodeName: str
    source: FileSource
    destination: FileDestination
    options: TransferOptions
    jobUuid: str
    optimizerOptions: OptimizerOptions


class ConfigType(str, Enum):
    train = "TRAIN"
    eval = "EVALUATE"
    tune = "TUNE"


class GlobalConfig(BaseModel):
    configName: str
    modelType: ModelType
    modelName: str
    episodeCount: int = 30
    reward_window: Optional[int] = 4
    query_time_window: Optional[str] = '-2m'
    configType: ConfigType


class TrainConfig(GlobalConfig):
    action_space_discrete: Optional[bool] = False
    obs_cols: Optional[list] = []
    render_type: Optional[str] = ""
    estimated_link_capacity: Optional[int] = 1000


class EvaluateConfig(GlobalConfig):
    obs_cols: Optional[list] = []
    render_type: Optional[str] = ""


class DDPGTrainingConfig(TrainConfig):
    bufferSize: int = 1000000
    learningStarts: int = 100
    batchSize: int = 256
    tau: float = 0.005
    gamma: float = 0.99
    trainFreq: int = 1
    gradientSteps: int = 1
    learningRate: float = .001
    net_arch: list = [400, 300]


class PPOTrainingConfig(TrainConfig):
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = .99
    gae_lambda: float = 0.95
    normalize_advantage: bool = True
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = None
    stats_window_size: int = 100
    learningRate: float = 0.0003


class BOTrainingConfig(TrainConfig):
    pass


class SGDTrainingConfig(TrainConfig):
    pass


class TuneConfig:
    pass