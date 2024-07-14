from app.optimizers.ddpg.ddpg_train_runner import DdpgTrainRunner
from app.api.models import ModelType, TransferJobRequest
from app.api.models import DDPGTrainingConfig, GlobalConfig
from app.storage.OptimizerStore import OptimizerStore


class RunnerFactory:
    @staticmethod
    def create_runner(transfer_request: TransferJobRequest, storage: OptimizerStore, config: GlobalConfig):
        if ModelType.ddpg == config.modelType:
            if isinstance(config, DDPGTrainingConfig):
                return DdpgTrainRunner(transfer_request=transfer_request, ddpg_config=config, model_storage=storage)
        else:
            raise ValueError("Invalid Optimizer type")
