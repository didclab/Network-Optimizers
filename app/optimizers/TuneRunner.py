from app.api.models import TransferJobRequest, TuneConfig
from app.optimizers.ModelFactory import ModelFactory
from app.storage.OptimizerStore import OptimizerStore
from app.environemnts.ods_real_transfer_env import InfluxEnv


class TuneRunner:

    def __init__(self, transfer_request: TransferJobRequest, model_store: OptimizerStore, tune_config: TuneConfig,
                 model_factory: ModelFactory):
        self.transfer_req = transfer_request
        self.model_store = model_store
        self.tune_config = tune_config
        self.model_factory = model_factory
        self.env = InfluxEnv(transfer_request=transfer_request, action_space_discrete=False,
                             render_type=None, reward_window=self.tune_config.reward_window,
                             query_time_window=self.tune_config.query_time_window)

    def tune_transfer(self):
        rewards = []
        actions = []
        pass

    def load_model(self):
        optimizer_options = self.transfer_req.optimizerOptions
        model_path = self.model_store.load_model(modelName=optimizer_options.modelName,
                                                 modelType=optimizer_options.modelType,
                                                 owner_id=self.transfer_req.ownerId)
        self.model = self.model_factory.load_model(model_type=optimizer_options.modelType, file_path=model_path)
