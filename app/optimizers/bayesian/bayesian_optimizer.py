import time
import matplotlib.pyplot as plt

from skopt import gp_minimize
from skopt.plots import plot_convergence
from skopt.space import Integer
from skopt.callbacks import CheckpointSaver
from app.db.influx_db import InfluxDb
from app.api.models import CreateOptimizerRequest
import app.db.db_helper as oh
from app.optimizers.bayesian.skopt_callbacks import JobStopper


class BayesianOpt:
    def __init__(self, create_req: CreateOptimizerRequest, time_window="-2m"):
        self.influx_client = InfluxDb(time_window=time_window)
        self.params = [Integer(1, int(create_req.maxConcurrency), name='concurrency'),
                       Integer(1, int(create_req.maxParallelism), name='parallelism')]
        self.create_req = create_req
        self.time_window = time_window
        self.rewards = []
        self.data_cols = ['active_core_count', 'allocatedMemory',
                          'dropin', 'dropout', 'packet_loss_rate', 'chunkSize', 'concurrency',
                          'destination_latency', 'destination_rtt', 'jobSize', 'parallelism',
                          'pipelining', 'read_throughput', 'source_latency', 'source_rtt', 'write_throughput']
        self.job_id = create_req.jobId
        self.past_rewards = []
        self.terminated = False

    # def create_optimizer(self, create_req: CreateOptimizerRequest):

    def object_func(self, params):
        next_cc = params[0]
        next_p = params[1]

        # Apply bayesian params
        if (1 < next_cc < self.create_req.maxConcurrency) and (1 < next_p < self.create_req.maxParallelism):
            oh.send_application_params_tuple(
                transfer_node_name=self.create_req.nodeId,
                cc=next_cc, p=next_p, pp=1, chunkSize=0)

        fail_count = 0
        while True:
            print("Blocking till action: ", params)
            df = self.influx_client.query_space(job_uuid=self.create_req.jobUuid, time_window="-30s",
                                                bucket_name=self.create_req.userId,
                                                transfer_node_name=self.create_req.nodeId)

            if set(self.data_cols).issubset(df.columns):
                last_n_row = df.tail(n=1)
                print("Concurrency Value waiting for: " + str(next_cc) + " got: " + str(
                    last_n_row['concurrency'].iloc[-1]))
                print("Parallelism Value waiting for: " + str(next_p) + " got: " + str(
                    last_n_row['parallelism'].iloc[-1]))

                if all(last_n_row['concurrency'] == next_cc) and all(last_n_row['parallelism'] == next_p):
                    throughput = last_n_row['read_throughput'].iloc[-1]
                    print("Read throughput reward: " + str(throughput))
                    return -abs(throughput)
                else:
                    print("Sleeping for 2 seconds for the next df")
                    time.sleep(2)

    def run_bayesian(self, episodes=10):

        # episodes are entire transfer jobs
        # print("Starting to optimize BO")
        # for i in range(0, episodes):
        #     if self.terminated:
        #         return
        checkpoint_callback = CheckpointSaver("./bayesian_run")
        job_stopper = JobStopper(jobId=self.create_req.jobId, dbType=self.create_req.dbType)
        self.bayes_model = gp_minimize(self.object_func, self.params, callback=[job_stopper])
        # plot_convergence(self.bayes_model)
        # plt.savefig('transfer_test_plot.png')

        # print("Optimization result: {}".format(result))
        # self.bayes = result
        # print("InfluxEnv: relaunching job: ", first_meta_data['jobParameters'])
        # oh.submit_transfer_request(first_meta_data, optimizer="DDPG")
        # self.terminated = False
        # time.sleep(30)
        # self.delete_optimizer()

    def delete_optimizer(self):
        self.terminated = True
        plot_convergence(self.bayes_model)
        plt.savefig('convergence_plot.png')

    def close(self):
        self.influx.close_client()
