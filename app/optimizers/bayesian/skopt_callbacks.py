from skopt.callbacks import EarlyStopper
from app.db.influx_db import InfluxDb
from app.api.models import CreateOptimizerRequest


class JobStopper(EarlyStopper):
    def __init__(self, create_req: CreateOptimizerRequest, influx_client: InfluxDb):
        super().__init__()
        self.influx_client = influx_client
        self.create_req = create_req

    def _criterion(self, result):
        df = self.influx_client.query_space(job_uuid=self.create_req.jobUuid, time_window="-30s",
                                            bucket_name=self.create_req.userId,
                                            transfer_node_name=self.create_req.nodeId)
        last_row = df.tail(n=1)
        if not last_row['isRunning'] and last_row['jobUuid'] == self.create_req.jobUuid:
            terminated = True
        else:
            terminated = False
        print("Job Stopper callback has value: " + str(terminated))
        return terminated
