"""Unit test methods for the mltemplate.core.registry.Registry class."""
import pytest

from mltemplate.modules import Registry


def test_registry():
    """Tests the Registry class."""
    registry = Registry()

    if len(registry.models) == 0:
        pytest.skip("No models found in the model registry.")
        # TODO: Add a training run to the test suite.

    run_id = list(registry.models.keys())[0]

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

    with pytest.raises(ValueError):
        registry.model_name_and_version(run_id="")
