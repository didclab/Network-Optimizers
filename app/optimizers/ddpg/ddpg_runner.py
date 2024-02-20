from typing import Optional

from app.optimizers.TrainRunner import TrainRunner
from app.api.models import RequestFromODS
from app.api.models import DDPGTrainingConfig
from app.environemnts.ods_real_transfer_env import InfluxEnv
from stable_baselines3.ddpg import DDPG
from stable_baselines3.common.noise import NormalActionNoise
from app.storage.OptimizerStore import OptimizerStore
import numpy as np


class DdpgRunner(TrainRunner):

    def __init__(self, transfer_request: RequestFromODS, ddpg_config: DDPGTrainingConfig,
                 model_storage: OptimizerStore):
        self.file_transfer = transfer_request
        self.ddpg_config = ddpg_config
        self.env = InfluxEnv(transfer_request=transfer_request, action_space_discrete=False,
                             obs_cols=self.ddpg_config.obs_cols,
                             render_type=None, reward_window=self.ddpg_config.reward_window,
                             query_time_window=self.ddpg_config.query_time_window)
        n_actions = self.env.action_space.shape[-1]
        self.action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
        self.model = DDPG('MlpPolicy', self.env, learning_rate=self.ddpg_config.learningRate,
                          buffer_size=self.ddpg_config.bufferSize, learning_starts=self.ddpg_config.learningStarts,
                          batch_size=self.ddpg_config.batchSize, tau=self.ddpg_config.tau, gamma=self.ddpg_config.gamma,
                          train_freq=self.ddpg_config.trainFreq, gradient_steps=self.ddpg_config.gradientSteps,
                          action_noise=self.action_noise, replay_buffer_class=None, replay_buffer_kwargs=None,
                          tensorboard_log=None,
                          policy_kwargs=None, verbose=1, seed=None, device='auto', _init_setup_model=True)
        self.episode_rewards = []
        self.model_save_path_local = ""
        self.model_storage = model_storage

    def train(self):
        self.model.learn(total_timesteps=self.ddpg_config.episodeCount, log_interval=1)
        self.save_model()

    def save_model(self, path: Optional[str] = None):
        return self.model_storage.save_model(owner_id=self.file_transfer.ownerId, config=self.ddpg_config,
                                             base_algo=self.model)

    def load_model(self, path: Optional[str] = None):
        model_path = self.model_storage.load_model(owner_id=self.file_transfer.ownerId, config=self.ddpg_config)
        self.model = DDPG.load(path=model_path, env=self.env)