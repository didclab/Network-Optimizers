import logging
import os.path

import torch

from app.api.models import CreateOptimizerRequest
from app.environemnts import ods_real_transfer_env
from app.optimizers.BaseRunner import BaseRunner
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.logger import configure
import numpy as np
from typing import Optional

class DdpgRunner(BaseRunner):
    def __init__(self, create_req: CreateOptimizerRequest):
        self.env = ods_real_transfer_env.InfluxEnv(create_req=create_req, reward_window=1)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        policy_kwargs = dict(net_arch=[400, 300])
        replay_buffer_kwargs = {"handle_timeout_termination": False}
        self.action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(self.env.action_space.shape),
                                                         sigma=0.1 * np.ones(self.env.action_space.shape))
        self.agent = DDPG("MlpPolicy", self.env, policy_kwargs=policy_kwargs, verbose=1, device=self.device,
                          replay_buffer_kwargs=replay_buffer_kwargs, action_noise=self.action_noise)
        self.new_logger = configure("tmp/sb3_ddpg_log/", ["stdout", "csv"])
        self.agent.set_logger(self.new_logger)
        self.episodes = 1  # Adjust as needed
        self.batch_size = 64
        self.create_req = create_req
        self.output_csv_path = 'ddpg_output_output.csv'
        self.save_model_path = 'ddpg_sb3_model'
        self.gradient_steps = 5
        self.job_ended = False

    def train(self):
        self.agent.learn(total_timesteps=2, log_interval=10)
        self.save_model()
        done = False
        rewards = []
        actions = []
        for i in range(0,30):
            obs = self.env.reset()
            while done == False:
                action, _ = self.agent.predict(obs)
                obs, reward, done, info = self.env.step(action)
                actions.append(action)
                rewards.append(reward)
                if done:
                    obs = self.env.reset()

        # for episode in range(0, self.episodes, self.batch_size):
        #     if episode == 0:
        #         obs = self.env.reset()
        #     else:
        #         obs = self.env.reset(launch_job=True)
        #     terminated = False
        #     while not terminated:
        #         action, _ = self.agent.predict(obs, deterministic=False)
        #         next_obs, reward, terminated, options = self.env.step(action)
        #         if self.job_ended: terminated = True
        #         logging.info(
        #             f"Train action: {action} has shape {action.shape} Terminated: {terminated}, Reward: {reward}, NextObs: {next_obs}")
        #         self.agent.replay_buffer.add(obs=obs, action=action, reward=reward, next_obs=next_obs, done=terminated,
        #                                      infos=[{}])
        #         obs = next_obs
        #
        #     self.agent.train(batch_size=self.batch_size, gradient_steps=self.gradient_steps)
        # self.save_model(self.save_model_path)

    def evaluate(self, data):
        pass

    def save_model(self, path: Optional[str] = None):
        if path is None:
            self.agent.save(self.save_model_path)
        else:
            self.agent.save(path)

    def load_model(self, path: Optional[str] = None):
        model_path = path
        if model_path is None:
            model_path = self.save_model_path

        if os.path.exists(model_path):
            self.agent = self.agent.load(model_path, env=self.env, device=self.device)

    def save_episode(self, episode_count=0):
        pass

    def warm_buffer(self):
        logging.info("Starting to warm the buffer")
        logging.info(f"Create request used: {self.create_req}")
        # Query historical data for warming the buffer
        df = self.env.influx_client.query_job_data(bucket_name=self.create_req.userId,
                                                   transfer_node_name=self.create_req.nodeId, time_window='-1d')
        past_row = df.iloc[0]
        past_obs = df[self.env.data_columns].iloc[0]

        for i in range(1, df.shape[0] - 1):  # Start loop from index 1 to avoid indexing errors
            current_row = df.iloc[i]
            next_obs = df[self.env.data_columns].iloc[i]

            action = current_row[['parallelism', 'concurrency']].values
            logging.info(f"Action Warm Buffer: {action} has shape {action.shape}")
            # Check if the action has changed
            reward = current_row['read_throughput']
            # Check if 'jobId' has changed for termination
            terminated = not past_row['jobId'] == (current_row['jobId'])
            self.agent.replay_buffer.add(obs=past_obs.to_numpy(dtype=np.float32), action=action, reward=reward, next_obs=next_obs.to_numpy(dtype=np.float32),
                                         done=terminated, infos={})

            past_row = current_row
            past_obs = next_obs

        logging.info(f"Warmed the buffer with {self.agent.replay_buffer.size()}")

    def set_job_ended(self):
        self.job_ended = True