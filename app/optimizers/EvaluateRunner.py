from app.api.models import TransferJobRequest
from app.environemnts.ods_real_transfer_env import InfluxEnv
from app.storage.OptimizerStore import OptimizerStore
from app.storage.JobMetricsStore import JobMetricsStore
from app.optimizers.ModelFactory import ModelFactory
from app.api.models import EvaluateConfig, JobMetrics
import torch


class EvaluateRunner:
    def __init__(self, transfer_request: TransferJobRequest, model_store: OptimizerStore, metrics_store: JobMetricsStore, config: EvaluateConfig):
        self.eval_config = config
        self.file_transfer_request = transfer_request
        self.model_storage = model_store
        self.metrics_store = metrics_store
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
        epoch_data = []
        for i in range(0, self.eval_config.episodeCount):
            obs = self.env.reset()
            action, _ = self.model.predict(observation=obs)
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            loss = 0.0
            if hasattr(self.model, 'policy') and hasattr(self.model.policy, 'compute_loss'):
                try:
                    if hasattr(self.model, 'replay_buffer') and self.model.replay_buffer.size() > 0:
                        batch = self.model.replay_buffer.sample(1)
                        loss_dict = self.model.policy.compute_loss(batch)
                        loss = loss_dict.loss.item() if hasattr(loss_dict, 'loss') else loss_dict.item()
                    else:
                        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                        action_tensor = torch.tensor(action).unsqueeze(0)
                        _, log_prob, _ = self.model.policy.evaluate_actions(obs_tensor, action_tensor)
                        loss = -log_prob.mean().item()
                except Exception as e:
                    print(f"Loss computation failed: {str(e)}")
            rewards.append(reward)
            actions.append(action)
            epoch_data.append({"reward": reward, "action": action, "loss": loss})
        
        metrics = JobMetrics(
            epoch_data=epoch_data,
            total_reward=sum(data["reward"] for data in epoch_data),
            action_count=len(epoch_data)
        )

        self.metrics_store.save_job_metrics(owner_id=self.file_transfer_request.ownerId,
                job_uuid=self.file_transfer_request.jobUuid,
                metrics=metrics)

        return actions, rewards

    def load_model(self):
        self.model = ModelFactory.load_model(model_type=self.eval_config.modelType, file_path=self.model_path)

    def close(self):
        self.env.close()
