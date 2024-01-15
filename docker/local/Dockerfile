# First build a base image, which should only need be changed if the project dependencies are update.

# If building for CPU, use the base Ubuntu image that matches your host OS
#FROM ubuntu:22.04 as base

# Else, if building for GPU, use the Nvidia/CUDA image
FROM nvidia/cuda:mltemplate as base

ARG DEBIAN_FRONTEND=noninteractive
ARG USER=docker
ENV USER=${USER}

# Set up non-root user
ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/home/${USER}" \
    --shell "/bin/bash" \
    --uid "${UID}" \
    ${USER}
WORKDIR /home/${USER}/dev

# Update and install dependencies
RUN apt-get -y update && apt-get install -y \
  apt-utils \
  build-essential \
  python3-pip \
  python3-opencv \
  curl

# Set Rye environment variables
ENV RYE_HOME="/home/${USER}/.rye"
ENV PATH="$RYE_HOME/shims:$PATH"

# Install Rye
RUN curl -sSf https://rye-up.com/get | RYE_INSTALL_OPTION="--yes" bash

# Copy the project files
COPY pyproject.toml .
COPY README.md .

# Install dependencies
RUN rye fetch 3.10  # torch currently does not support Python 3.11
RUN rye sync --no-dev


FROM base as release

# Install Mltemplate
COPY mltemplate ./mltemplate
RUN rye build
RUN rye install . --python 3.10

USER ${USER}

FROM release as dev

RUN rye sync  # includes the dev dependencies (e.g. pytest)


FROM dev as test

CMD PYTHONPATH=/home/${USER}/dev rye run test


FROM release as backend

CMD rye run mlflow server --backend-store-uri ${HOME}/mltemplate/mlflow --port 8080 & \
    PYTHONPATH=${HOME}/dev python3 -m gunicorn -w 4 -b localhost:8082 -k uvicorn.workers.UvicornWorker "mltemplate.backend.training.training_server:app()" & \
    PYTHONPATH=${HOME}/dev python3 -m gunicorn -w 1 -b localhost:8083 -k unicorn.workers.UvicornWorker "mltemplate.backend.deployment.deployment_server:app()" & \
    python3 -m gunicorn -w 1 -b localhost:8081 -k uvicorn.workers.UvicornWorker "mltemplate.backend.gateway.gateway_server:app()" & \
    PYTHONPATH=${HOME}/dev rye run discord_client
