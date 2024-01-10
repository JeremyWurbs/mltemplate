"""Mltemplate Gateway Server"""
from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional

import mlflow
import numpy as np
from fastapi import FastAPI, HTTPException
from pytorch_lightning import LightningDataModule

from mltemplate import MltemplateBase, Registry
from mltemplate.backend.gateway.types import (
    BestModelForExperimentInput,
    ChatInput,
    ClassifyIDInput,
    ClassifyImageInput,
    ListRunsInput,
    LoadModelInput,
    TrainInput,
)
from mltemplate.backend.gateway.connection_client import ConnectionClient as GatewayConnection
from mltemplate.backend.training import TrainingServer
from mltemplate.data import MNIST
from mltemplate.types import Message
from mltemplate.utils import ascii_to_pil, default_logger, pil_to_ascii, pil_to_ndarray, tensor_to_pil


class GatewayServer(MltemplateBase):
    """Mltemplate Gateway Server

    The Mltemplate Gateway Server provides a unified interface for communicating with the Mltemplate backend. It is used
    by the frontend to fetch information about models and experiments in the registry, and to start training runs. The
    gateway server is currently also responsible for loading and serving models, although further development would
    likely move this functionality to a separate server better suited for parallelizing resource-heavy tasks.

    Args:
        tracking_server_uri: The URI of the MLFlow tracking server. If not given, defaults to the URI specified in the
            config file.

    code::

        $ python -m gunicorn -w 1 -b localhost:8081 -k uvicorn.workers.UvicornWorker \
            mltemplate.backend.gateway.server:app

    """

    loaded_models: Dict[str, mlflow.pyfunc.PyFuncModel] = {}  # model_name_and_version: PyFuncModel
    default_model: Optional[str] = None

    def __init__(self, tracking_server_uri: Optional[str] = None):
        super().__init__()
        self.registry = Registry(tracking_server_uri=tracking_server_uri)
        self.training_server = TrainingServer.connection(self.config["HOSTS"]["TRAINING_SERVER"])

        self.commands: List[str] = [
            "commands",
            "models",
            "load-model",
            "classify-by-id",
            "list-experiments",
            "list-runs",
            "list-models",
            "best-model-for-experiment",
        ]
        if "XXXXXXXXXX" not in self.config["API_KEYS"]["OPENAI"]:
            self.commands.append("chat")

        # TODO: Make Registry save and load datasets dynamically for us, instead of hardcoding them here
        self.loaded_datasets: Dict[str, LightningDataModule] = {"MNIST": MNIST()}
        self.default_dataset: Optional[str] = "MNIST"

    def _retrieve_model(self, model_name: Optional[str] = None):
        model = None
        if model_name is not None:
            model = self.loaded_models.get(model_name)
        if model is None:
            model = self.loaded_models.get(self.default_model)
        if model is None:
            raise ValueError("No model loaded or given.")
        return model

    def _retrieve_dataset(self, dataset_name: Optional[str] = None):
        dataset = None
        if dataset_name is not None:
            dataset = self.loaded_datasets.get(dataset_name)
        if dataset is None:
            dataset = self.loaded_datasets.get(self.default_dataset)
        if dataset is None:
            raise ValueError("No dataset loaded or given.")
        return dataset

    def app(self):
        app_ = FastAPI()

        @app_.post("/commands")
        def list_commands():
            return {"commands": self.commands}

        @app_.post("/chat")
        def chat(payload: ChatInput):
            # TODO
            self.logger.debug(f"Server received request for chat with payload: {payload}")
            response = Message(sender="mltemplate", text="Sorry, I don't know how to chat yet.")
            self.logger.debug(f"Server returning response for chat: {response}")
            return {"sender": response.sender, "text": response.text}

        @app_.post("/models")
        def models():
            self.registry.refresh()
            return list(self.registry.models.values())

        @app_.post("/fetch-experiments")
        def fetch_experiments():
            self.logger.debug("Received fetch_experiments request.")
            experiments = self.registry.mlflow.fetch_experiments()
            self.logger.debug(f"Returning list_experiments request with data: {experiments}.")
            return experiments

        @app_.post("/fetch-runs")
        def fetch_runs(payload: ListRunsInput):
            self.logger.debug(f"Received fetch_runs request with payload: {payload}.")
            runs = self.registry.mlflow.fetch_runs(payload.experiment_name)
            self.logger.debug(f"Returning list_runs request with data: {runs}.")
            return runs

        @app_.post("/fetch-models")
        def fetch_models():
            self.logger.debug("Received fetch_models request.")
            models = self.registry.mlflow.fetch_models()
            self.logger.debug(f"Returning list_models request with data: {models}.")
            return models

        @app_.post("/best-model-for-experiment")
        def best_model_for_experiment(payload: BestModelForExperimentInput):
            self.logger.debug(f"Received best_model_for_experiment request with payload: {payload}.")
            model = self.registry.best_model_for_experiment_name(payload.experiment_name)
            self.logger.debug(f"Returning best_model_for_experiment request with data: {model}.")
            return model

        @app_.post("/load-model")
        def load_model(payload: LoadModelInput):
            self.logger.debug(f"Received load_model request with payload: {payload}.")
            if (payload.model is None or payload.version is None) and payload.run_id is None:
                err_message = "Must specify either (1) model and version or (2) run_id."
                self.logger.error(err_message)
                raise HTTPException(status_code=400, detail=err_message)

            if payload.run_id is not None:
                try:
                    model_name_and_version = self.registry.model_name_and_version(payload.run_id)
                except ValueError as e:  # If the model is not found in the registry
                    self.logger.error(e)
                    raise HTTPException(status_code=400, detail=str(e)) from e
            else:
                model_name_and_version = f"{payload.model}/{payload.version}"

            model = mlflow.pyfunc.load_model(f"models:/{model_name_and_version}")
            self.loaded_models[model_name_and_version] = model
            self.default_model = model_name_and_version
            self.logger.debug("Returning load_model request.")
            return True

        @app_.post("/classify-id")
        def classify_id(payload: ClassifyIDInput):
            self.logger.debug(f"Received classify_by_id request with payload: {payload}.")
            model = self._retrieve_model(payload.model)
            dataset = self._retrieve_dataset(payload.dataset)
            image, label = dataset.sample(stage=payload.stage, idx=payload.idx)

            logits = model.predict(image.numpy())
            prediction = logits.argmax()

            response = {
                "label": label,
                "prediction": int(prediction),
                "logits": logits.tolist(),
            }
            self.logger.debug(f"Returning classify_by_id request with data: {response}.")
            response["image"] = pil_to_ascii(tensor_to_pil(image))
            return response

        @app_.post("/classify-image")
        def classify_image(payload: ClassifyImageInput):
            self.logger.debug(f"Received classify_image request with payload {payload}.")
            model = self._retrieve_model(payload.model)

            # Convert image to ndarray with an added batch dimension
            image = ascii_to_pil(payload.image)  # (C, H, W)
            if image.mode in ["L", "LA"]:
                image_format = "L"
            else:
                image_format = "RGB"
            image = pil_to_ndarray(image, image_format=image_format).astype(np.float32)
            image = np.expand_dims(image, axis=0)  # (B, C, H, W)

            logits = model.predict(image)
            prediction = logits.argmax()

            response = {"prediction": int(prediction), "logits": logits.tolist()}
            self.logger.debug(f"Returning classify_image request with data: {response}.")
            return response

        @app_.post("/train")
        def train(payload: TrainInput):
            self.logger.debug(f"Received train request with payload {payload}.")
            response = self.training_server.start_training_run(
                request_id=payload.request_id,
                command_line_arguments=payload.command_line_arguments,
            )
            self.logger.debug(f"Returning train request with data: {response}.")
            return response

        @app_.post("/training-complete")
        def training_complete(payload: TrainInput):
            self.logger.debug(f"Received training_complete request with payload {payload}.")
            self.registry.refresh()
            self.logger.debug("Returning training_complete request.")
            return True

        return app_

    @classmethod
    def connection(cls, host):
        return GatewayConnection(host)


def app():
    server = GatewayServer()
    server.logger = default_logger(
        name=server.name,
        stream_level=logging.INFO,
        file_level=logging.DEBUG,
        file_name=os.path.join(server.config["DIR_PATHS"]["LOGS"], "gateway_server_logs.txt"),
    )
    server.logger.info(f"Starting Gateway Server {id(server)}.")
    return server.app()
