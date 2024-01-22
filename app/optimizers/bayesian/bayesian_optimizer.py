import csv
import json
import os
import time
import logging
logging.basicConfig(level=logging.INFO)

from skopt import gp_minimize, dump, load
from skopt.plots import plot_convergence
from skopt.space import Integer
from app.db.influx_db import InfluxDb
from app.api.models import CreateOptimizerRequest
import app.db.db_helper as oh
from app.optimizers.bayesian.skopt_callbacks import JobStopper
import pandas as pd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


class BayesianOpt:
    def __init__(self, time_window="-2m"):
        self.job_id = None
        self.create_req = None
        self.params = None
        self.bayes_model = None
        self.influx_client = InfluxDb()
        self.time_window = time_window
        self.past_rewards = []
        self.data_cols = ['active_core_count', 'allocatedMemory', 'chunkSize', 'concurrency',
                          'destination_latency', 'destination_rtt', 'jobSize', 'parallelism',
                          'pipelining', 'read_throughput', 'source_latency', 'source_rtt', 'write_throughput']
        self.terminated = False
        self.dump_path = 'models/bayesian.pkl'
        self.past_actions = []
        self.json_file = "actions_throughput.json"
        self.ema_alpha = 0.2

    def adjust_to_create_request(self, create_req: CreateOptimizerRequest):
        self.params = [Integer(1, int(create_req.maxConcurrency), name='concurrency'),
                       Integer(1, int(create_req.maxParallelism), name='parallelism')]
        self.create_req = create_req
        self.job_id = create_req.jobId

    def object_func(self, params):
        next_cc = params[0]
        next_p = params[1]
        # Apply bayesian params
        if (1 < next_cc < self.create_req.maxConcurrency) and (1 < next_p < self.create_req.maxParallelism):
            oh.send_application_params_tuple(
                transfer_node_name=self.create_req.nodeId,
                cc=next_cc, p=next_p, pp=1, chunkSize=0)
            logging.info(f'Sent next action: cc:{next_cc}, p:{next_p}')

        re_push_params = 0
        while True:
            logging.info(f'Blocking till action {params}')
            print("Blocking till action: ", params)
            df = self.influx_client.query_space(job_uuid=self.create_req.jobUuid, time_window="-30s",
                                                bucket_name=self.create_req.userId,
                                                transfer_node_name=self.create_req.nodeId)
            # if self.create_req.dbType == "hsql":
            #     terminated, _ = oh.query_if_job_done_direct(self.create_req.jobId)
            # else:
            #     terminated, _ = oh.query_if_job_done(self.create_req.jobId)

            if set(self.data_cols).issubset(df.columns):
                last_n_row = df.tail(n=4)
                last_row = df.tail(n=1)
                if not last_row['isRunning'].iloc[-1]:
                    terminated = True
                else:
                    terminated = False
                obs_cc = last_n_row['concurrency'].iloc[-1]
                obs_p = last_n_row['parallelism'].iloc[-1]
                logging.info(f'Concurrency Value waiting for: {next_cc} got {obs_cc}')
                logging.info(f'Parallelism Value waiting for: {next_p} got {obs_p}')

                ema_throughput = self.ema_for_last_n(last_n_row, n=4)
                if ema_throughput is None:
                    ema_throughput = last_n_row['read_throughput'].iloc[-1]

                if terminated:
                    self.past_actions.append((int(next_cc), int(next_p)))
                    self.past_rewards.append(ema_throughput)
                    return -abs(ema_throughput)
                if (last_n_row['concurrency'] == next_cc).all() and (last_n_row['parallelism'] == next_p).all():
                    logging.info(f'App Tuple cc:{next_cc} p:{next_p}')
                    logging.info(f'Read Throughput Reward: {ema_throughput}')
                    self.past_actions.append((next_cc, next_p))
                    self.past_rewards.append(ema_throughput)
                    return -abs(ema_throughput)
                else:
                    logging.info('Sleeping for 2 seconds for the next df')
                    re_push_params += 1
                    if re_push_params >= 5:
                        oh.send_application_params_tuple(
                            transfer_node_name=self.create_req.nodeId,
                            cc=next_cc, p=next_p, pp=1, chunkSize=0)
                        re_push_params = 0
                    time.sleep(10)

    def run_bayesian(self, episodes=10):
        job_stopper = JobStopper(create_req=self.create_req, influx_client=self.influx_client)
        self.bayes_model = gp_minimize(self.object_func, self.params, callback=[job_stopper])
        self.checkpoint()
        self.graph_model()

    def graph_model(self):
        # combined_list = [{'jobUuid': self.create_req.jobUuid, 'jobId': self.create_req.jobId,
        #                   'actions_rewards': {'concurrency': self.convert_to_python_int(concurrency),
        #                                       'parallelism': self.convert_to_python_int(parallelism),
        #                                       'throughput': self.convert_to_python_int(rewards)}}
        #                  for (concurrency, parallelism), rewards in zip(self.past_actions, self.past_rewards)]
        #
        # with open(self.json_file,'a+', newline='') as file:
        #     json.dump(combined_list, file, indent=2)
        combined_list = []
        for(concurrency, parallelism), rewards in zip(self.past_actions, self.past_rewards):
            entry = {
                'jobUuid': self.create_req.jobUuid,
                'jobId': self.create_req.jobId,
                'actions': [{'concurrency': self.convert_to_python_int(concurrency),
                             'parallelism': self.convert_to_python_int(parallelism)}],
                'throughput': [self.convert_to_python_int(rewards)]
            }
            combined_list.append(entry)
        with open(self.json_file, 'a+', newline='') as file:
            json.dump(combined_list, file)

        plot_convergence(self.bayes_model)
        os.makedirs('graphs/', exist_ok=True)
        plt.savefig('graphs/transfer_test_plot.png')
        plt.close()
        logging.info(f'Optimizer took actions: {self.past_actions}')
        logging.info(f'Rewards Corresponding to actions: {self.past_rewards}')
        print("Optimization result: {}".format(self.bayes_model))
        labels = [f"({concurrency}, {parallelism})" for concurrency, parallelism in self.past_actions]
        plt.plot(labels, self.past_rewards, marker='o', linestyle='-', color='b', label='Throughput')
        plt.xlabel('Concurrency and Parallelism')
        plt.ylabel('Throughput')
        plt.title('Concurrency and Parallelism vs Throughput')
        # Rotating x-axis labels for better visibility
        plt.xticks(rotation=45, ha='right')
        plt.legend()
        plt.savefig('graphs/bo_actions_cc_p_throughput.png')
        plt.close()
        self.past_actions.clear()
        self.past_rewards.clear()

    def delete_optimizer(self):
        self.terminated = True
        plot_convergence(self.bayes_model)
        plt.savefig('convergence_plot.png')

    def close(self):
        self.influx_client.close_client()

    def checkpoint(self):
        print(self.bayes_model)
        dump(self.bayes_model, self.dump_path, store_objective=False)

    def load(self):
        self.bayes_model = load(self.dump_path)
        logging.info('Bayes Model loaded')

    def convert_to_python_int(self, value):
        if pd.api.types.is_integer_dtype(value):
            return int(value)
        return value

    def ema_for_last_n(self, values, n):
        if len(values) < n:
            return None  # Not enough values to calculate EMA

        ema = values[0]
        for i in range(1, len(values)):
            ema = self.ema_alpha * values[i] + (1 - self.ema_alpha) * ema
        return ema
