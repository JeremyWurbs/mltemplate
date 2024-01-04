"""Utility methods relating to dynamically generating objects."""
import importlib


def dynamic_instantiation(module_name: str, class_name: str) -> object:
    """Dynamically instantiates a class from a module."""
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    instance = class_()
    return instance


def instantiate_target(target: str):
    """Instantiates a target object from a string.

    The target string should be in the same format as expected from Hydra targets. I.e. 'module_name.class_name'.

    Args:
        target: A string representing a target object.

    Example::

        from mltemplate.utils import instantiate_target

        target = 'mltemplate.data.mnist.MNISTDataModule'
        mnist = instantiate_target(target)

        print(type(mnist))  # <class 'mltemplate.data.mnist.MNISTDataModule'>
    """
    module_name, class_name = target.rsplit(".", 1)
    return dynamic_instantiation(module_name, class_name)
