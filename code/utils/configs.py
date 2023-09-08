import yaml
import os
from code.utils.paths import resolve_root_path

def load_data_config(path_config):
    with open(os.path.join(os.environ["ROOT_DIRECTORY"], path_config), 'r') as stream:
        try:
            config = yaml.safe_load(stream)
            config["path_data"] = resolve_root_path(config["path_data"])
            return config
        except yaml.YAMLError as exc:
            print(exc)

def load_config(path_config) : 
    with open(os.path.join(os.environ["ROOT_DIRECTORY"], path_config), 'r') as stream : 
        try : 
            config = yaml.safe_load(stream)
            return config
        except yaml.YAMLError as exc : 
            print(exc)