"""Unit test methods for the mltemplate.core.registry.Registry class."""
import pytest

from mltemplate import Registry


def test_registry():
    """Tests the Registry class."""
    registry = Registry()

    if len(registry.mlflow.fetch_runs()) == 0:
        pytest.skip("No runs found in MLflow.")
        # TODO: Add a training run to the test suite.

    run_id = registry.mlflow.fetch_runs()[0]["run_id"]

    model_name = registry.model_name(run_id)
    assert isinstance(model_name, str)

    model_versions = registry.model_versions(model_name)
    assert isinstance(model_versions, list)

    model_name_and_version = registry.model_name_and_version(run_id)
    assert isinstance(model_name_and_version, str)

    experiment_names = registry.experiment_names
    assert isinstance(experiment_names, list)

    experiment_ids = registry.experiment_ids
    assert isinstance(experiment_ids, list)

    runs = registry.mlflow.fetch_runs()
    assert isinstance(runs, list)

    runs = registry.mlflow.fetch_runs(experiments="MNIST")
    assert isinstance(runs, list)

    runs = registry.mlflow.fetch_runs(experiments=["MNIST"])
    assert isinstance(runs, list)

    with pytest.raises(ValueError):
        registry.model_name_and_version(run_id="")
