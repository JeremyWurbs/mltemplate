"""Core Registry module."""
import json
from typing import Dict, List, Optional

import mlflow
from mlflow import MlflowClient

from mltemplate import MltemplateBase
from mltemplate.utils import ifnone


class Registry(MltemplateBase):
    """Registry class. Provides unified access to the MLFlow tracking server.

    The Registry class provides a unified interface for accessing the MLFlow tracking server. It is used by the
    frontend to fetch information about models and experiments in the registry.

    Args:
        tracking_server_uri: The URI of the MLFlow tracking server. If not given, defaults to the URI specified in the
            config file.

    Example::

        from mltemplate import Registry

        registry = Registry()
        for model in registry.models:
            print(model)
    """

    def __init__(self, tracking_server_uri: Optional[str] = None):
        super().__init__()
        tracking_server_uri = ifnone(tracking_server_uri, default=self.config["DIR_PATHS"]["MLFLOW"])
        mlflow.set_tracking_uri(tracking_server_uri)
        self.client = MlflowClient(tracking_uri=tracking_server_uri)

        self.models = None
        self.experiments = None
        self.refresh()

    def refresh(self):
        """Refreshes the information about all models and experiments in the registry."""
        self.models = self._fetch_models_info()
        self.experiments = self._fetch_experiments_info()

    def _fetch_models_info(self) -> Dict[str, Dict]:
        """Helper method to fetch the information for all models in the registry."""
        models = {}
        for registered_model in self.client.search_registered_models():
            for version in self.client.search_model_versions(f"name='{registered_model.name}'"):
                run = self.client.get_run(run_id=version.run_id)
                params = json.loads(run.data.params["model"].replace("'", '"'))
                models[version.run_id] = {
                    "name": registered_model.name,
                    "version": str(version.version),
                    "dataset": run.data.params["dataset_name"],
                    "status": version.status,
                    "train_acc": run.data.metrics.get("train_acc_epoch", 0.0),
                    "val_acc": run.data.metrics.get("val_acc_epoch", 0.0),
                    "test_acc": run.data.metrics.get("test_acc_epoch", 0.0),
                    "params": {key: params[key] for key in params.keys() - {"_target_", "name"}},
                    "experiment_id": run.info.experiment_id,
                    "run_id": version.run_id,
                }
        return models

    def _fetch_experiments_info(self) -> Dict[str, Dict]:
        """Helper method to fetch the names of all experiments in the registry."""
        experiments = {}
        for experiment in self.client.search_experiments():
            if experiment.name != "Default":
                experiments[experiment.experiment_id] = {"name": experiment.name}
        return experiments

    def model_name(self, run_id: str) -> str:
        return self.models[run_id]["name"]

    def model_versions(self, model_name: str) -> List[str]:
        return [model["version"] for model in self.models.values() if model["name"] == model_name]

    def model_name_and_version(self, run_id: str) -> str:
        model = self.models.get(run_id)
        if model is None:
            raise ValueError(f"No model found with run_id: {run_id}")
        return f'{model["name"]}/{model["version"]}'

    @property
    def experiment_names(self) -> List[str]:
        """Returns the names of all experiments in the registry."""
        return [experiment["name"] for experiment in self.experiments.values()]

    @property
    def experiment_ids(self) -> List[str]:
        """Returns the ids of all experiments in the registry."""
        return list(self.experiments.keys())

    def experiment_id(self, experiment_name: str) -> str:
        """Returns the id of the specified experiment."""
        return self.client.get_experiment_by_name(experiment_name).experiment_id

    def best_model_for_experiment(self, experiment_id: str) -> Optional[Dict]:
        """Returns the best model for the specified experiment or None, if one cannot be found."""
        self.refresh()
        if len(self.models) == 0 or experiment_id not in self.experiment_ids:
            return None
        run_id = max(
            self.models,
            key=(
                lambda key: self.models[key]["test_acc"] if self.models[key]["experiment_id"] == experiment_id else -1.0
            ),
        )
        return self.models.get(run_id)

    def best_model_for_experiment_name(self, experiment_name: str) -> Dict:
        """Returns the best model for the specified experiment."""
        experiment_id = self.experiment_id(experiment_name)
        return self.best_model_for_experiment(experiment_id)

    def run_id_from_request_id(self, request_id: str) -> Optional[str]:
        """Returns the run_id associated with the specified request_id or None, if one is not found or not finished."""
        for run in self.client.search_runs(self.experiment_ids):
            if run.data.tags.get("request_id") == request_id:
                if run.info.status == "FINISHED":
                    return run.info.run_id
                else:
                    return None
        return None
