from app.api.models import TransferJobRequest
from app.environemnts.ods_real_transfer_env import InfluxEnv
from app.storage.OptimizerStore import OptimizerStore
from app.optimizers.ModelFactory import ModelFactory
from app.api.models import EvaluateConfig


class EvaluateRunner:
    def __init__(self, transfer_request: TransferJobRequest, model_store: OptimizerStore, config: EvaluateConfig):
        self.eval_config = config
        self.file_transfer_request = transfer_request
        self.model_storage = model_store
        self.env = InfluxEnv(transfer_request=self.file_transfer_request, action_space_discrete=False,
                             obs_cols=self.eval_config.obs_cols,
                             render_type=None, reward_window=self.eval_config.reward_window,
                             query_time_window=self.eval_config.query_time_window)
        self.model_path = self.model_storage.load_model(owner_id=self.file_transfer_request.ownerId,
                                                        modelType=self.eval_config.modelType,
                                                        modelName=self.eval_config.modelName)

    def evaluate(self):
        rewards = []
        actions = []
        for i in range(0, self.eval_config.episodeCount):
            obs = self.env.reset()
            action, _ = self.model.predict(observation=obs)
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            rewards.append(reward)
            actions.append(action)

        return actions, rewards

    def load_model(self):
        self.model = ModelFactory.load_model(model_type=self.eval_config.modelType, file_path=self.model_path)

    def close(self):
        self.env.close()
