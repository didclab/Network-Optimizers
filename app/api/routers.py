import os.path

from fastapi import APIRouter, BackgroundTasks
from app.api.models import CreateOptimizerRequest, DeleteOptimizerRequest
from app.optimizers.bayesian.bayesian_optimizer import BayesianOpt
from app.optimizers.BaseRunner import BaseRunner
from app.environemnts.ods_real_transfer_env import InfluxEnv
from stable_baselines3.common import env_checker
from app.optimizers.ddpg.sb3_ddpg_runner import DdpgRunner
from typing import Dict, Type, TypeVar
import logging

T = TypeVar('T', bound='BaseRunner')

optimizer_api = APIRouter()
RunnerMap: Dict[str, Type[T]] = {}
thread_map = {}
bayes_opt = BayesianOpt()
if os.path.exists(bayes_opt.dump_path):
    bayes_opt.load()


@optimizer_api.post("/optimizer/create", status_code=201)
async def create_optimizer(create_request: CreateOptimizerRequest, background_tasks: BackgroundTasks) -> None:
    print("Creating optimizer: " + create_request.__str__())
    if create_request.optimizerType == "BO":
        bayes_opt.adjust_to_create_request(create_request)
        print(create_request)
        background_tasks.add_task(bayes_opt.run_bayesian)

    if create_request.optimizerType == "DDPG":
        runner = DdpgRunner(create_req=create_request)
        RunnerMap[create_request.nodeId] = runner
        runner.load_model()
        runner.warm_buffer()
        background_tasks.add_task(runner.train)


@optimizer_api.post("/optimizer/delete", status_code=201)
async def delete_optimizer(delete_request: DeleteOptimizerRequest) -> None:
    runner = RunnerMap.get(delete_request.nodeId)
    runner.set_job_ended()
    # runner.save_model()
    # time.sleep(5)
    # bayes_opt.graph_model()

@optimizer_api.post("/optimizer/train", status_code=200)
async def train_optimizer():
    pass

@optimizer_api.get("/env/baselines/compatible", status_code=201)
async def check_env():
    req = CreateOptimizerRequest(nodeId="jgoldverg@gmail.com-mac", maxConcurrency=32, maxParallelism=32,
                                 maxPipelining=100, maxChunkSize=100000,
                                 optimizerType="DDPG", fileCount=22, jobId=0, dbType="hsql",
                                 jobUuid="014d84ae-f4fb-4658-a617-824f28930f1a", userId="jgoldverg@gmail.com")

    env = InfluxEnv(create_req=req)
    logging.info("Created the ENV")
    env_checker.check_env(env)
