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
RUN rye run create_project_directories

USER ${USER}
