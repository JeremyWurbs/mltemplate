"""Mltemplate Deployment Server"""
from __future__ import annotations

import logging
import os
from typing import Dict, Optional

import mlflow
import numpy as np
from fastapi import FastAPI, HTTPException
from pytorch_lightning import LightningDataModule

from mltemplate import MltemplateBase
from mltemplate.backend.deployment.connection_client import ConnectionClient as DeploymentConnection
from mltemplate.backend.deployment.types import ClassifyIDInput, ClassifyImageInput, LoadModelInput
from mltemplate.data import MNIST
from mltemplate.modules import Registry
from mltemplate.utils import ascii_to_pil, default_logger, pil_to_ascii, pil_to_ndarray, tensor_to_pil


class DeploymentServer(MltemplateBase):
    """Mltemplate Deployment Server

    The Mltemplate Deployment Server provides a unified interface for deploying models. It is used by the gateway server
    to deploy models.

    Args:
        tracking_server_uri: The URI of the MLFlow tracking server. If not given, defaults to the URI specified in the
            config file.

    code::

        $ python -m gunicorn -w 1 -b localhost:8083 -k uvicorn.workers.UvicornWorker \
            mltemplate.backend.deployment.deployment_server:app

    """

    loaded_models: Dict[str, mlflow.pyfunc.PyFuncModel] = {}  # model_name_and_version: PyFuncModel
    default_model: Optional[str] = None

    def __init__(self, tracking_server_uri: Optional[str] = None):
        super().__init__()
        self.registry = Registry(tracking_server_uri=tracking_server_uri)

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
            print(response)
            print(f'type: {type(response["image"])}')
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

        return app_

    @classmethod
    def connection(cls, host):
        return DeploymentConnection(host)


def app():
    server = DeploymentServer()
    server.logger = default_logger(
        name=server.name,
        stream_level=logging.INFO,
        file_level=logging.DEBUG,
        file_name=os.path.join(server.config["DIR_PATHS"]["LOGS"], "deployment_server_logs.txt"),
    )
    server.logger.info(f"Starting Deployment Server {id(server)}.")
    return server.app()
