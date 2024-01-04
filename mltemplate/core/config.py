"""Mltemplate Config class."""
import json
import os
from configparser import ConfigParser, ExtendedInterpolation
from typing import Any, Dict


class Config:
    """Mltemplate Config.

    To use the config parser, fill out the *config.ini* file. It includes paths to local directories as well as API keys
    necessary to use some services, such as OpenAI (ChatGPT) or the Discord Client.

    Some notes to help you:
        1. API keys are required by most services. Refer to the services documentation for more information.
        2. You may use tildes (~) to denote the user home directory in the config file.
        3. The config file may refer to other parts of itself using ${}. Refer to the config file itself for examples.
        4. Most demos / scripts that ask for a resource path can be left blank if the config has the associated info.

    Args:
        config_path: The complete path to the .ini file. If not provided it will be looked for in the same directory as
            this config.py file.

    Example::

        import json
        from mltemplate import Config

        config = Config()
        print(config['MLTEMPLATE']['ROOT_DIR'])  # '/home/user/mltemplate'

        config = Config().as_dict()  # May use Config.__str__() or Config.as_dict() for serializable operations
        print(json.dumps(config, indent=4))

    """

    def __init__(self, config_path: str = None):
        self.config = ConfigParser(interpolation=ExtendedInterpolation())
        self.config.optionxform = str  # Sets ConfigParser to maintain case sensitivity

        if config_path is None:
            search_dir = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
            config_path = os.path.join(search_dir, "config.ini")

        if not os.path.exists(config_path):
            raise FileNotFoundError(f'No such file "{config_path}".')

        self.config.read(config_path)

        for section in self.config.sections():
            for k, v in self.config.items(section):
                if v[0] == "~":
                    self.config[section][k] = v.replace("~", os.path.expanduser("~"))

    def __getitem__(self, item: str) -> Any:
        return self.config[item]

    def __str__(self) -> str:
        return str({section: dict(self.config[section]) for section in self.config.sections()}).replace("'", '"')

    def as_dict(self) -> Dict:
        return json.loads(str(self))

    def pretty_print(self) -> str:
        return json.dumps(self.as_dict(), indent=4)
