import os
from dotenv import load_dotenv

def resolve_root_path(path) :
    if "$ROOT_DIRECTORY" in path :
        load_dotenv()
        path = path.replace("$ROOT_DIRECTORY", os.environ["ROOT_DIRECTORY"])

    return path