from importlib import import_module
from typing import Dict, List

import yaml


class ConfigManager:
    def __init__(self, config: Dict):
        self.config = config

    def init_object(self, name: str, *args, **kwargs) -> object:
        # Root module
        root = "src"

        object_path = self.config[name]
        if object_path is None:
            return None

        module_name, object_name = object_path.rsplit(".", 1)

        module = import_module(f"{root}.{module_name}")

        object_args = self.config[f"{name}_args"] or {}
        kwargs = {**kwargs, **object_args}

        return getattr(module, object_name)(*args, **kwargs)

    def init_objects(self, name: str, *args, **kwargs) -> List[object]:
        # Root module
        root = "src"

        objects = []

        object_paths = self.config[name]
        n_objects = len(object_paths)
        object_args = self.config[f"{name}_args"] or [{}] * n_objects

        # Repeat single args across objects
        args = [arg if isinstance(arg, list) else [arg] * n_objects for arg in args]
        # print(args)
        args = list(zip(*args))
        # FIXME Figure out something for kwargs

        for object_path, object_arg, arg in zip(object_paths, object_args, args):
            module_name, object_name = object_path.rsplit(".", 1)
            module = import_module(f"{root}.{module_name}")

            objects.append(getattr(module, object_name)(*arg, **object_arg))

        return objects


def load_yaml(path):
    with open(path, "r") as file:
        try:
            yaml_file = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    return yaml_file
