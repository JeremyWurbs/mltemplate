"""Pydanitc models for training server API."""
from typing import Optional

from pydantic import BaseModel


class TrainingRunInput(BaseModel):
    command_line_arguments: str = "--config-name train.yaml model=mlp dataset=mnist"
    request_id: Optional[str] = None
