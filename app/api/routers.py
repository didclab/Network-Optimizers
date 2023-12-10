from fastapi import APIRouter
from app.api.models import CreateOptimizerRequest, DeleteOptimizerRequest
from app.optimizers.bayesian.bayesian_optimizer import BayesianOpt
import threading

optimizer_api = APIRouter()
map_opt = {}
thread_map = {}


@optimizer_api.post("/optimizer/create", status_code=201)
async def create_optimizer(create_request: CreateOptimizerRequest) -> None:
    print("Creating optimizer: " + create_request.__str__())
    if create_request.optimizerType == "BO":
        print(create_request)
        bayes_opt = BayesianOpt(create_req=create_request)
        map_opt[create_request.nodeId] = bayes_opt
        t = threading.Thread(target=bayes_opt.run_bayesian)
        # thread_map[create_request.nodeId] = t
        t.start()
        # bayes_opt.run_bayesian()


@optimizer_api.post("/optimizer/delete", status_code=201)
async def delete_optimizer(delete_request: DeleteOptimizerRequest) -> None:
    pass