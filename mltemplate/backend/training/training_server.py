"""Mltemplate Training Server"""
from __future__ import annotations

import logging
import os
import shlex
import subprocess

from fastapi import BackgroundTasks, FastAPI

from mltemplate import Config, MltemplateBase
from mltemplate.backend.training.connection_client import ConnectionClient as TrainingConnection
from mltemplate.backend.training.types import TrainingRunInput
from mltemplate.utils import default_logger


class TrainingServer(MltemplateBase):
    """Mltemplate Training Server

    The Mltemplate Training Server provides a unified interface for starting training runs. It is used by the gateway
    server to start training runs on the training server.

    code::

        $ python -m gunicorn -w 1 -b localhost:8081 -k uvicorn.workers.UvicornWorker \
            mltemplate.backend.training.training_server:app

    """

    logger = default_logger(
        name="mltemplate.backend.training.training_server.TrainingServer",
        stream_level=logging.INFO,
        file_level=logging.DEBUG,
        file_name=os.path.join(Config()["DIR_PATHS"]["LOGS"], "training_server_logs.txt"),
        file_mode="a",
    )

    def __init__(self):
        super().__init__()

    @staticmethod
    def train_background_task(payload: TrainingRunInput):
        TrainingServer.logger.debug(
            f"Server processing train_background_task (id: {payload.request_id}) with payload: {payload}"
        )
        command_line_arguments = payload.command_line_arguments + f' request_id="{payload.request_id}"'
        arguments = ["run", "train"]
        arguments += shlex.split(command_line_arguments)
        if "--multirun" in arguments:  # Make sure multiruns is the last argument, if given
            arguments.remove("--multirun")
            arguments.append("--multirun")
        TrainingServer.logger.debug(f"train_background_task (id: {payload.request_id}) arguments: {arguments}")
        try:
            _ = subprocess.run(["rye", *arguments], check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            TrainingServer.logger.error(
                f"Server failed processing train_background_task (id: {payload.request_id}) with error: " f"{e}"
            )
            raise e
        TrainingServer.logger.debug(f"Server finished processing train_background_task (id: {payload.request_id}).")

    def app(self):
        app_ = FastAPI()

        @app_.post("/start_training_run")
        def start_training_run(payload: TrainingRunInput, background_tasks: BackgroundTasks):
            self.logger.debug(f"Server received request for start_training_run with payload: {payload}")
            background_tasks.add_task(self.train_background_task, payload)
            response = "Server received request for start_training_run"
            self.logger.debug(f"Server returning response for start_training_run: {response}")
            return response

        return app_

    @classmethod
    def connection(cls, host):
        return TrainingConnection(host)


def app():
    server = TrainingServer()
    server.logger = default_logger(
        name=server.name,
        stream_level=logging.INFO,
        file_level=logging.DEBUG,
        file_name=os.path.join(server.config["DIR_PATHS"]["LOGS"], "training_server_logs.txt"),
    )
    server.logger.info(f"Starting Training Server {id(server)}.")
    return server.app()
