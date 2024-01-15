"""Client-side helper class for communicating with the Mltemplate gateway server."""
import json
from typing import Optional

import requests
from fastapi import HTTPException
from PIL.Image import Image

from mltemplate.utils import ascii_to_pil, pil_to_ascii


class ConnectionClient:
    """Client-side helper class for communicating with the Mltemplate deployment server."""

    def __init__(self, host: str = "http://localhost:8080/"):
        self.host = host

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
