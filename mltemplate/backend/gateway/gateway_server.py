"""Mltemplate Gateway Server"""
from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional

import mlflow
import numpy as np
from fastapi import FastAPI, HTTPException
from pytorch_lightning import LightningDataModule

from mltemplate import MltemplateBase
from mltemplate.backend.gateway.connection_client import ConnectionClient as GatewayConnection
from mltemplate.backend.gateway.types import (
    BestModelForExperimentInput,
    ChatInput,
    ClassifyIDInput,
    ClassifyImageInput,
    DebugInput,
    LoadModelInput,
    TrainInput,
)
from mltemplate.backend.training import TrainingServer
from mltemplate.data import MNIST
from mltemplate.modules import GPT, Registry
from mltemplate.types import Message
from mltemplate.utils import ascii_to_pil, default_logger, ifnone, pil_to_ascii, pil_to_ndarray, tensor_to_pil


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

        self.commands: List[str] = ["commands", "models", "load-model", "classify-by-id"]
        self.gpt = None
        if "XXXXXXXXXX" not in self.config["API_KEYS"]["OPENAI"]:
            self.gpt = GPT()
            self.commands.append("chat")
            self.commands.append("debug")

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
            self.logger.debug(f"Server received request for chat with payload: {payload}")
            if self.gpt is not None:
                response = self.gpt(payload.text)
            else:
                response = Message(sender="mltemplate", text="Sorry, I don't know how to chat yet.")
            self.logger.debug(f"Server returning response for chat: {response}")
            return {"sender": response.sender, "text": response.text}

        @app_.post("/models")
        def models():
            self.logger.debug("Received models request.")
            self.registry.refresh()
            model_list = list(self.registry.models.values())
            self.logger.debug(f"Returning models request with data: {model_list}.")
            return model_list

        @app_.post("/experiments")
        def experiments():
            self.logger.debug("Received experiments request.")
            self.registry.refresh()
            experiment_list = self.registry.experiment_names
            self.logger.debug(f"Returning experiments request with data: {experiment_list}.")
            return experiment_list

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
                except ValueError as err:  # If the model is not found in the registry
                    self.logger.error(err)
                    raise HTTPException(status_code=400, detail=str(err)) from err
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

        @app_.post("/debug")
        def debug(payload: DebugInput):
            self.logger.debug(f"Received debug request with payload {payload}.")

            discord_logfile = os.path.join(self.config["DIR_PATHS"]["LOGS"], "discord_logs.txt")
            gateway_server_logfile = os.path.join(self.config["DIR_PATHS"]["LOGS"], "gateway_server_logs.txt")
            training_server_logfile = os.path.join(self.config["DIR_PATHS"]["LOGS"], "training_server_logs.txt")
            training_logfile = os.path.join(self.config["DIR_PATHS"]["LOGS"], "train_logs.txt")

            log_files = []
            if os.path.exists(discord_logfile):
                log_files.append(discord_logfile)
            if os.path.exists(gateway_server_logfile):
                log_files.append(gateway_server_logfile)
            if os.path.exists(training_server_logfile):
                log_files.append(training_server_logfile)
            if os.path.exists(training_logfile):
                log_files.append(training_logfile)

            instructions = (
                "You are a professional python developer. You're job is to help debug the application servers for the "
                f"mltemplate project. You will talk with the mltemplate application developers and answer their "
                f"questions. You have access to the following log files: {log_files}.\n\n"
                f"Use the provided log files to provide useful debugging information to developer queries."
            )
            try:
                with GPT(filenames=log_files, instructions=instructions) as gpt:
                    text = ifnone(payload.text, default="Please help me debug the most recent command I ran.")
                    response = gpt(text)
                self.logger.debug(f"Returning debug request with response:\n{response}")
                return response
            except Exception as err:
                self.logger.error(err)
                raise HTTPException(status_code=400, detail=str(err)) from err

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
