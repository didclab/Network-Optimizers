import json

import gymnasium as gym
import numpy as np
from gymnasium import spaces
import logging
from app.api.models import CreateOptimizerRequest
from app.db.influx_db import InfluxDb
import app.db.db_helper as oh
import time
import subprocess
gym.logger.set_level(gym.logger.INFO)


class InfluxEnv(gym.Env):

    def __init__(self, create_req: CreateOptimizerRequest, action_space_discrete=False, obs_cols=[], reward_type=None,
                 render_type=None, reward_window=4, reward_func=None, query_time_window='-2m', job_config_path="'../../job_config/concurrency_dataset_transfer.json'"):
        super(InfluxEnv, self).__init__()
        self.replay_buffer = None
        self.create_req = create_req
        self.reward_window = reward_window
        self.influx_bucket_name = create_req.nodeId.split("-")[0]
        self.influx_client = InfluxDb()
        if len(obs_cols) > 0:
            self.data_columns = obs_cols
        else:
            self.data_columns = ['active_core_count', 'allocatedMemory', 'chunkSize', 'concurrency',
                                 'destination_latency', 'destination_rtt', 'jobSize', 'parallelism',
                                 'pipelining', 'read_throughput', 'source_latency', 'source_rtt', 'write_throughput']

        self.reward_type = reward_type
        self.observation_space = spaces.Box(low=1, high=np.inf, shape=(len(self.data_columns),), dtype=np.float32)
        if action_space_discrete:
            self.action_space = spaces.Discrete(3)
        else:
            self.action_space = spaces.Box(low=1, high=self.create_req.maxConcurrency, shape=(2,))
            logging.info(f"Action space Shape {self.action_space.shape}")
        self.past_actions = []
        self.past_rewards = []
        self.render_type = render_type
        self.reward_function = reward_func or self.default_reward_avg
        self.influx_time_window = query_time_window
        self.job_config_path = job_config_path

    def default_reward_avg(self, df):
        return df['read_throughput'].tail(n=self.reward_window).mean()

    def step(self, action):
        next_cc = action[0]
        next_p = action[1]
        # Execute action agent is taking
        send_next_action = True

        if (0 < next_cc <= self.create_req.maxConcurrency) and (0 < next_p <= self.create_req.maxParallelism):
            send_next_action = False
        if len(self.past_actions) > 1:
            past_action = self.past_actions[-1]
            if past_action[0] == next_cc and past_action[1] == next_p:
                send_next_action = False

        if send_next_action:
            oh.send_application_params_tuple(
                        transfer_node_name=self.create_req.nodeId,
                        cc=next_cc, p=next_p, pp=1, chunkSize=0)
            logging.info(f'Sent next action: cc:{next_cc}, p:{next_p} to Node:{self.create_req.nodeId}')

        while True:
            logging.info(f'Blocking till action {action} takes effect')
            df = self.influx_client.query_space(job_uuid=self.create_req.jobUuid, time_window=self.influx_time_window,
                                                bucket_name=self.create_req.userId,
                                                transfer_node_name=self.create_req.nodeId)
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
                    return last_row[self.data_columns].to_numpy(dtype=np.float32), reward, terminated, {}

                if (last_n_row['concurrency'] == next_cc).all() and (last_n_row['parallelism'] == next_p).all():
                    self.past_actions.append((next_cc, next_p))
                    reward = self.reward_function(df)
                    logging.info(f"Agent reward for CC:{next_cc} P:{next_p} gave avg read_throughput:{reward}")
                    self.past_rewards.append(reward)
                    return last_row[self.data_columns].to_numpy(dtype=np.float32), reward, terminated, {}
            time.sleep(2)

    def reset(self, seed=None, **kwargs):
        with open(self.job_config_path, 'r') as file:
            transfer_request = json.load(file)
            req = oh.submit_transfer_request(transfer_request)
            logging.info(req.json())
        time.sleep(10)

        self.past_rewards.clear()
        self.past_actions.clear()
        return (self.influx_client.query_space(job_uuid=self.create_req.jobUuid, time_window='-30s',
                                               bucket_name=self.create_req.userId,
                                               transfer_node_name=self.create_req.nodeId)[self.data_columns].iloc[-1]
                .to_numpy(dtype=np.float32))

    def render(self):
        pass

    def close(self):
        self.influx_client.close_client()
        pass
