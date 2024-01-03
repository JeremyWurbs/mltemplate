"""Utility methods relating to mlflow."""
import mlflow


def experiment_id(name: str) -> str:
    """Get the experiment ID for the given name.

    If an experiment of the given name does not exist, it will be created.

    Args:
        name: Name of the experiment.

    Returns:
        Experiment ID.
    """
    experiment = mlflow.get_experiment_by_name(name)
    if experiment is None:
        return mlflow.create_experiment(name)
    else:
        return experiment.experiment_id
