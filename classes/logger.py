import neptune
from stable_baselines3.common.logger import Logger, KVWriter
from stable_baselines3.common.callbacks import BaseCallback


class NeptuneLogger(Logger):
    def __init__(self, api_token, project_qualified_name, experiment_name=None, params=None, tags=None, upload_source_files=None):
        self.api_token = api_token
        self.project_qualified_name = project_qualified_name
        self.experiment_name = experiment_name
        self.params = params if params else {}
        self.tags = tags if tags else []
        self.upload_source_files = upload_source_files if upload_source_files else []
        
        self.run = neptune.init_run(api_token=self.api_token, project=self.project_qualified_name, name=self.experiment_name)

        super().__init__(folder="", output_formats=[])

    def writekvs(self, kvs):
        for key, value in kvs.items():
            self.run[key].append(value)
        super().writekvs(kvs)

    def dumpkvs(self, kvs):
        super().dumpkvs(kvs)

    def logkv(self, key, val):
        self.run[key].append(val)
        super().logkv(key, val)

    def logkv_mean(self, key, val):
        self.run[key].append(val)
        super().logkv_mean(key, val)

    def record(self, key, value, exclude=None):
        """Log a value of a scalar variable (and possibly record it to the terminal and/or do other logging)."""
        if exclude is not None and key in exclude:
            return
        self.run[key].append(value)
        super().record(key, value)

    def close(self):
        self.run.stop()
        super().close()


class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=1):
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        available_capital = self.locals['env'].get_attr("available_capital")[0]
        available_capital = available_capital if not isinstance(available_capital, list) else available_capital[0]
        last_action = self.locals['env'].get_attr("last_action")[0]
        last_action = last_action if not isinstance(last_action, list) else last_action[0]
        position = self.locals['env'].get_attr("position")[0]
        position = position if not isinstance(position, list) else position[0]
        total_asset = self.locals['env'].get_attr("total_asset")[0]
        total_asset = total_asset if not isinstance(total_asset, list) else total_asset[0]

        self.logger.record("training/available_capital", available_capital)
        self.logger.record("training/last_action", last_action)
        self.logger.record("training/position", position)
        self.logger.record("training/total_asset", total_asset)

        return True
