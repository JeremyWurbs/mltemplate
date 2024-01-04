"""Client-side helper class for communicating with the Mltemplate gateway server."""
import json
from typing import List, Optional

import requests
from fastapi import HTTPException
from PIL.Image import Image

from mltemplate.types import Message
from mltemplate.utils import ascii_to_pil, pil_to_ascii


class ConnectionClient:
    """Client-side helper class for communicating with the Mltemplate gateway server."""

    def __init__(self, host: str = "http://localhost:8080/"):
        self.host = host

    def commands(self) -> List[str]:
        response = requests.request("POST", self.host + "commands", timeout=60)
        if response.status_code != 200:
            raise HTTPException(response.status_code, response.content)
        return json.loads(response.content)["commands"]

    def chat(self, text: str) -> Message:
        response = requests.request("POST", self.host + "chat", json={"text": text}, timeout=60)
        if response.status_code != 200:
            raise HTTPException(response.status_code, response.content)
        data = json.loads(response.content)
        return Message(sender=data["sender"], text=data["text"])

    def models(self) -> List[str]:
        response = requests.request("POST", self.host + "models", timeout=60)
        if response.status_code != 200:
            raise HTTPException(response.status_code, response.content)
        return json.loads(response.content)

    def list_experiments(self):
        response = requests.request("POST", self.host + "fetch-experiments", timeout=60)
        if response.status_code != 200:
            raise HTTPException(response.status_code, response.content)
        return json.loads(response.content)

    def list_runs(self, experiment: Optional[str] = None):
        response = requests.request(
            "POST",
            self.host + "fetch-runs",
            json={"experiment": experiment},
            timeout=60,
        )
        if response.status_code != 200:
            raise HTTPException(response.status_code, response.content)
        return json.loads(response.content)

    def list_models(self):
        response = requests.request("POST", self.host + "fetch-models", timeout=60)
        if response.status_code != 200:
            raise HTTPException(response.status_code, response.content)
        return json.loads(response.content)

    def best_model_for_experiment(self, experiment_name: str):
        response = requests.request(
            "POST",
            self.host + "best-model-for-experiment",
            json={"experiment_name": experiment_name},
            timeout=60,
        )
        if response.status_code != 200:
            raise HTTPException(response.status_code, response.content)
        return json.loads(response.content)

    def load_model(
        self,
        model: Optional[str] = None,
        version: Optional[str] = None,
        run_id: Optional[str] = None,
    ):
        response = requests.request(
            "POST",
            self.host + "load-model",
            json={"model": model, "version": version, "run_id": run_id},
            timeout=60,
        )
        if response.status_code != 200:
            raise HTTPException(response.status_code, response.content)
        return json.loads(response.content)

    def classify_id(
        self,
        dataset: str = "MNIST",
        stage: str = "test",
        idx: int = 0,
        model: Optional[str] = None,
    ):
        response = requests.request(
            "POST",
            self.host + "classify-id",
            json={"dataset": dataset, "stage": stage, "idx": idx, "model": model},
            timeout=60,
        )
        print(response.content)
        response = json.loads(response.content)
        return {
            "image": ascii_to_pil(response["image"]),
            "label": response["label"],
            "prediction": response["prediction"],
            "logits": response["logits"],
        }

    def classify_image(self, image: Image, model: Optional[str] = None):
        response = requests.request(
            "POST",
            self.host + "classify-image",
            json={"image": pil_to_ascii(image), "model": model},
            timeout=60,
        )
        if response.status_code != 200:
            raise HTTPException(response.status_code, response.content)
        return json.loads(response.content)

    def train(
        self,
        request_id=str,
        command_line_arguments: str = "--config-name train.yaml model=mlp dataset=mnist",
    ):
        response = requests.request(
            "POST",
            self.host + "train",
            json={
                "request_id": request_id,
                "command_line_arguments": command_line_arguments,
            },
            timeout=24 * 60 * 60,
        )
        if response.status_code != 200:
            raise HTTPException(response.status_code, response.content)
        return json.loads(response.content)
