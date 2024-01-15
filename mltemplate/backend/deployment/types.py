"""Pydantic models for the deployment server API."""
from typing import Optional

from pydantic import BaseModel


class ClassifyIDInput(BaseModel):
    dataset: str = "MNIST"
    stage: str = "test"
    idx: int = 0
    model: Optional[str] = None


class ClassifyImageInput(BaseModel):
    image: str
    model: Optional[str] = None


class LoadModelInput(BaseModel):
    model: Optional[str] = None
    version: Optional[str] = None
    run_id: Optional[str] = None


class TrainInput(BaseModel):
    request_id: str
    command_line_arguments: str = "--config-name train.yaml model=mlp dataset=mnist"
