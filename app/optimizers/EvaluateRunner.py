from app.api.models import EvaluateRequest
from app.environemnts.ods_real_transfer_env import InfluxEnv
from app.storage.OptimizerStore import OptimizerStore
from app.optimizers.ModelFactory import ModelFactory


class EvaluateRunner:
    def __init__(self, evaluate_request: EvaluateRequest, model_store: OptimizerStore):
        self.file_transfer_request = evaluate_request.fileTransferRequest
        self.eval_config = evaluate_request.config
        self.model_storage = model_store
        self.env = InfluxEnv(transfer_request=self.file_transfer_request, action_space_discrete=False,
                             obs_cols=self.eval_config.obs_cols,
                             render_type=None, reward_window=self.eval_config.reward_window,
                             query_time_window=self.eval_config.query_time_window)
        self.model_path = self.model_storage.load_model(owner_id=self.file_transfer_request.ownerId,
                                                        config=self.eval_config)
        self.model = ModelFactory.load_model(self.eval_config.modelType, self.model_path)

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
