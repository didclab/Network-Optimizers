from stable_baselines3 import DDPG, PPO, A2C
from stable_baselines3.common.base_class import BaseAlgorithm
from pathlib import Path


class ModelFactory:

    @staticmethod
    def load_model(model_type: str, file_path: Path) -> BaseAlgorithm:
        if model_type == "DDPG":
            return DDPG.load(file_path)
        elif model_type == "PPO":
            return PPO.load(file_path)
        elif model_type == "A2C":
            return A2C.load(file_path)

