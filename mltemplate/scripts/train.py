"""Main mltemplate training script. Saves models to the model registry when finished."""
import logging
import os
import time

import hydra
import mlflow
import torch
from hydra.utils import HydraConfig
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from mltemplate import Config as PackageConfig
from mltemplate.utils import LightningModel, default_logger, experiment_id, ifnone

torch.set_float32_matmul_precision("medium")


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(config: DictConfig):
    """Train a model and register it to the model registry. Returns the run_id of the training run."""

    request_id = ifnone(config.get("request_id"), default="no-request-id")
    logger = default_logger(
        name="train.py",
        stream_level=logging.INFO,
        file_level=logging.DEBUG,
        file_name=os.path.join(PackageConfig()["DIR_PATHS"]["LOGS"], "train_logs.txt"),
        file_mode="a",
    )
    logger.debug(f"Starting training run (request_id: {request_id}, model: {config.model}, dataset: {config.dataset}).")

    try:
        mlflow.set_tracking_uri(PackageConfig()["DIR_PATHS"]["MLFLOW"])
        mlflow.autolog()
        run_name = ifnone(config.mlflow.run_name, default=f'{config.mlflow.user}-{time.strftime("%Y%m%d-%H%M%S")}')
        if "multi" in config["paths"]["output_dir"]:
            config_dir = os.path.join(config["paths"]["output_dir"], HydraConfig.get().output_subdir)
        else:
            config_dir = os.path.join(HydraConfig.get().run["dir"], HydraConfig.get().output_subdir)
        with mlflow.start_run(experiment_id=experiment_id(config.dataset.name), run_name=run_name):
            logger.debug(
                f"Starting training run (request_id: {request_id}, run_id: {mlflow.last_active_run().info.run_id})."
            )

            # Log config params
            mlflow.set_experiment(config.dataset.name)
            mlflow.log_param("dataset_name", config.dataset.name)  # Explicitly log the dataset being trained on
            mlflow.log_params(OmegaConf.to_container(config, resolve=True))  # Make config param searchable in MLFlow UI
            mlflow.log_artifacts(config_dir, artifact_path="config")  # Save the actual config (yaml) files as artifacts
            mlflow.set_experiment_tag("_dataset_", config.dataset._target_)  # pylint: disable=W0212

            # If the caller wishes to track this run through multiple levels of abstraction (e.g. the discord client
            # passing through an end user request), they can pass a request_id to track this run through the registry
            # and it will be saved as a request_id tag.
            if config.get("request_id") is not None:
                mlflow.set_tag("request_id", request_id)

            # Train and test model
            dm = hydra.utils.instantiate(config.dataset, **config.dataset)
            model = hydra.utils.instantiate(config.model, **config.model)
            model = LightningModel(model)

            tb_logger = TensorBoardLogger(
                save_dir=PackageConfig()["DIR_PATHS"]["TENSORBOARD"],
                name=f"{os.path.join('tensorboard', f'{config.model.name}-{config.model}')}",
            )
            checkpoint_callback = ModelCheckpoint(dirpath=tb_logger.experiment.log_dir)
            trainer = Trainer(logger=tb_logger, callbacks=[checkpoint_callback], **config.trainer)
            trainer.fit(model, dm)
            trainer.test(model, dm)

            # Register model to the model registry
            result = mlflow.pytorch.log_model(
                pytorch_model=model,
                artifact_path="model",
                input_example=dm.sample()[0].numpy(),  # MLFlow requires numpy arrays as input
                registered_model_name=config.model.name,
            )

    except Exception as err:
        logger.error(f"An exception occurred during training: {err}")
        raise err

    logger.debug(
        f"Training run for (request_id: {request_id}, run_id: {mlflow.last_active_run().info.run_id}) has "
        "finished. The resulting model has been added to the model registry."
    )
    return result.run_id


def entry():
    """Required entry point to enable hydra to work as a console_script."""
    main()  # pylint: disable=E1120


if __name__ == "__main__":
    main()  # pylint: disable=E1120
