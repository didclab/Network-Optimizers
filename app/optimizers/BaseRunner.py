from typing import Optional

class BaseRunner:
    def train(self):
        pass

    def evaluate(self, data):
        pass

    def save_model(self, path: Optional[str] = None):
        pass

    def load_model(self, path: Optional[str] = None):
        pass

    def set_job_ended(self):
        pass
