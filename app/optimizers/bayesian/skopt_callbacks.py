from skopt.callbacks import EarlyStopper
import app.db.db_helper as oh


class JobStopper(EarlyStopper):
    def __init__(self, jobId, dbType):
        super().__init__()
        self.jobId = jobId
        self.dbType = dbType

    def _criterion(self, result):
        terminated = False
        if self.dbType == "hsql":
            terminated, _ = oh.query_if_job_done_direct(self.jobId)
        else:
            terminated, _ = oh.query_if_job_done(self.jobId)
        print("Job Stopper callback has value: " +str(terminated))
        return terminated