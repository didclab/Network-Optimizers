from app.optimizers.ddpg.ddpg_runner import DdpgRunner
from app.api.models import ModelType, TrainRequest
from app.api.models import DDPGTrainingConfig
from app.storage.OptimizerStore import OptimizerStore


class RunnerFactory:
    @staticmethod
    def create_runner(train_request: TrainRequest, storage: OptimizerStore):
        if ModelType.ddpg == train_request.config.modelType:
            file_request = train_request.fileTransferRequest
            config = train_request.config
            if isinstance(config, DDPGTrainingConfig):
                return DdpgRunner(transfer_request=file_request, ddpg_config=config, model_storage=storage)
        else:
            raise ValueError("Invalid Optimizer type")
