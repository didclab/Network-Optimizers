import csv
import json
import os
import time

from skopt import gp_minimize, dump, load
from skopt.plots import plot_convergence
from skopt.space import Integer
from app.db.influx_db import InfluxDb
from app.api.models import CreateOptimizerRequest
import app.db.db_helper as oh
from app.optimizers.bayesian.skopt_callbacks import JobStopper

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

        re_push_params = 0
        while True:
            print("Blocking till action: ", params)
            df = self.influx_client.query_space(job_uuid=self.create_req.jobUuid, time_window="-30s",
                                                bucket_name=self.create_req.userId,
                                                transfer_node_name=self.create_req.nodeId)
            if self.create_req.dbType == "hsql":
                terminated, _ = oh.query_if_job_done_direct(self.create_req.jobId)
            else:
                terminated, _ = oh.query_if_job_done(self.create_req.jobId)

            if set(self.data_cols).issubset(df.columns):
                last_n_row = df.tail(n=4)
                print("Concurrency Value waiting for: " + str(next_cc) + " got: " + str(
                    last_n_row['concurrency'].iloc[-1]))
                print("Parallelism Value waiting for: " + str(next_p) + " got: " + str(
                    last_n_row['parallelism'].iloc[-1]))

                throughput = last_n_row['read_throughput'].iloc[-1]
                if terminated:
                    self.past_actions.append((next_cc, next_p))
                    self.past_rewards.append(throughput)
                    return -abs(throughput)
                if (last_n_row['concurrency'] == next_cc).all() and (last_n_row['parallelism'] == next_p).all():
                    print(last_n_row[['concurrency', 'parallelism']])
                    print("Read throughput reward: " + str(throughput))
                    self.past_actions.append((next_cc, next_p))
                    self.past_rewards.append(throughput)
                    return -abs(throughput)
                else:
                    print("Sleeping for 2 seconds for the next df")
                    re_push_params += 1
                    if re_push_params >= 5:
                        oh.send_application_params_tuple(
                            transfer_node_name=self.create_req.nodeId,
                            cc=next_cc, p=next_p, pp=1, chunkSize=0)
                        re_push_params = 0
                    time.sleep(10)

    def run_bayesian(self, episodes=10):
        job_stopper = JobStopper(jobId=self.create_req.jobId, dbType=self.create_req.dbType)
        self.bayes_model = gp_minimize(self.object_func, self.params, callback=[job_stopper])
        self.checkpoint()
        self.graph_model()

    def graph_model(self):
        combined_list = [{'jobUuid': self.create_req.jobUuid, 'jobId': self.create_req.jobId,
                          'actions_rewards': {'concurrency': concurrency, 'parallelism': parallelism, 'throughput': rewards}}
                         for (concurrency, parallelism), rewards in zip(self.past_actions, self.past_rewards)]

        with open(self.json_file,'w+', newline='') as file:
            json.dump(combined_list, file, indent=2)

        plot_convergence(self.bayes_model)
        os.makedirs('graphs/', exist_ok=True)
        plt.savefig('graphs/transfer_test_plot.png')
        plt.close()
        print("Optimizer actions taken: ", self.past_actions)
        print("Rewards corresponding to actions: ", self.past_rewards)
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
        print(self.bayes_model)
        print("Bayes Model has loaded")
