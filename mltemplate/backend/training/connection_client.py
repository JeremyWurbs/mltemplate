"""Client-side helper class for communicating with the Mltemplate gateway server."""
import json

import requests
from fastapi import HTTPException


class ConnectionClient:
    """Client-side helper class for communicating with the Mltemplate training server."""

    def __init__(self, host: str = "http://localhost:8081/"):
        self.host = host

    def start_training_run(
        self, request_id: str, command_line_arguments: str = "--config-name train.yaml model=mlp dataset=mnist"
    ):
        response = requests.request(
            "POST",
            self.host + "start_training_run",
            json={"request_id": request_id, "command_line_arguments": command_line_arguments},
            timeout=24 * 24 * 60,
        )
        if response.status_code != 200:
            raise HTTPException(response.status_code, response.content)
        return json.loads(response.content)
