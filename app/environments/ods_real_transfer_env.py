import json

import gymnasium as gym
import numpy as np
from gymnasium import spaces
import logging
from app.api.models import TransferJobRequest
from app.db.influx_db import InfluxDb
import app.db.db_helper as oh
import time
import subprocess

gym.logger.set_level(gym.logger.INFO)


class InfluxEnv(gym.Env):

    def __init__(self, transfer_request: TransferJobRequest, action_space_discrete=False, obs_cols=[],
                 render_type=None, reward_window=4, reward_func=None, query_time_window='-2m'):
        super(InfluxEnv, self).__init__()
        self.transfer_request = transfer_request
        self.reward_window = reward_window
        self.influx_bucket_name = transfer_request.transferNodeName.split("-")[0]
        self.influx_client = InfluxDb()
        if len(obs_cols) > 0:
            self.data_columns = obs_cols
        else:
            self.data_columns = ['active_core_count', 'allocatedMemory', 'chunkSize', 'concurrency',
                                 'destination_latency', 'destination_rtt', 'jobSize', 'parallelism',
                                 'pipelining', 'read_throughput', 'source_latency', 'source_rtt', 'write_throughput']

        self.observation_space = spaces.Box(low=1, high=np.inf, shape=(len(self.data_columns),), dtype=np.float32)
        if action_space_discrete:
            self.action_space = spaces.Discrete(3)
        else:
            self.action_space = spaces.Box(low=1, high=self.transfer_request.options.maxConcurrency, shape=(2,))
            logging.info(f"Action space Shape {self.action_space.shape}")
        self.past_actions = []
        self.past_rewards = []
        self.render_type = render_type
        self.reward_function = reward_func or self.default_reward_avg
        self.influx_time_window = query_time_window

    def default_reward_avg(self, df):
        return df['read_throughput'].tail(n=self.reward_window).mean()

    def step(self, action):
        """

        :param action:
        :return: next_obs, reward, terminated, truncated, info
        """
        next_cc = action[0]
        next_p = action[1]
        send_next_action = True
        truncated = False
        if (0 < next_cc <= self.transfer_request.options.maxConcurrency) and (
                0 < next_p <= self.transfer_request.options.maxParallelism):
            send_next_action = False
        if len(self.past_actions) > 1:
            past_action = self.past_actions[-1]
            if past_action[0] == next_cc and past_action[1] == next_p:
                send_next_action = False

        if send_next_action:
            oh.send_application_params_tuple(
                transfer_node_name=self.transfer_request.transferNodeName,
                cc=next_cc, p=next_p, pp=1, chunkSize=0)
            logging.info(f'Sent next action: cc:{next_cc}, p:{next_p} to Node:{self.transfer_request.transferNodeName}')

        while True:
            logging.info(f'Blocking till action {action} takes effect')
            df = self.influx_client.query_space(job_uuid=self.transfer_request.jobUuid,
                                                time_window=self.influx_time_window,
                                                bucket_name=self.transfer_request.ownerId,
                                                transfer_node_name=self.transfer_request.transferNodeName)
            if set(self.data_columns).issubset(df.columns):
                last_n_row = df.tail(n=self.reward_window)
                last_row = df.iloc[-1]
                if not last_row['isRunning']:
                    terminated = True
                else:
                    terminated = False
                logging.info(f"Terminated: {terminated}")
                obs_cc = last_n_row['concurrency'].iloc[-1]
                obs_p = last_n_row['parallelism'].iloc[-1]
                logging.info(f'CC P tuple waiting for: {next_cc} {next_p}')
                logging.info(f'CC P tuple got: {obs_cc} {obs_p}')

                if terminated:
                    self.past_actions.append((next_cc, next_p))
                    reward = self.reward_function(df)
                    self.past_rewards.append(reward)
                    return last_row[self.data_columns].to_numpy(dtype=np.float32), reward, terminated, truncated, {}

                if (last_n_row['concurrency'] == next_cc).all() and (last_n_row['parallelism'] == next_p).all():
                    self.past_actions.append((next_cc, next_p))
                    reward = self.reward_function(df)
                    logging.info(f"Agent reward for CC:{next_cc} P:{next_p} gave avg read_throughput:{reward}")
                    self.past_rewards.append(reward)
                    return last_row[self.data_columns].to_numpy(dtype=np.float32), reward, terminated, truncated, {}
            time.sleep(2)

    def reset(self, seed=None, **kwargs):
        self.past_rewards.clear()
        self.past_actions.clear()
        oh.submit_transfer_request(self.transfer_request)
        time.sleep(20)
        return self.influx_client.query_job_data(bucket_name=self.transfer_request.ownerId,
                                                 transfer_node_name=self.transfer_request.transferNodeName,
                                                 time_window='-30s')[self.data_columns].iloc[-1].to_numpy(
            dtype=np.float32)

    def render(self):
        pass

    def close(self):
        self.influx_client.close_client()
        pass
