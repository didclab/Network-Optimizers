import time

from fastapi import APIRouter, BackgroundTasks, Depends
from app.api.models import CreateOptimizerRequest, DeleteOptimizerRequest
from app.optimizers.bayesian.bayesian_optimizer import BayesianOpt
import threading

optimizer_api = APIRouter()
map_opt = {}
thread_map = {}
bayes_opt = BayesianOpt()
bayes_opt.load()

@optimizer_api.post("/optimizer/create", status_code=201)
async def create_optimizer(create_request: CreateOptimizerRequest, background_tasks: BackgroundTasks) -> None:
    print("Creating optimizer: " + create_request.__str__())
    if create_request.optimizerType == "BO":
        bayes_opt.adjust_to_create_request(create_request)
        print(create_request)
        map_opt[create_request.nodeId] = bayes_opt
        background_tasks.add_task(bayes_opt.run_bayesian)


@optimizer_api.post("/optimizer/delete", status_code=201)
async def delete_optimizer(delete_request: DeleteOptimizerRequest) -> None:
    time.sleep(5)
    bayes_opt.graph_model()
