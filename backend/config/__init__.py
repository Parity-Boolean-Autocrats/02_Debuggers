import os
from dotenv import load_dotenv

BASE_URL = "/api/v1"


def get_config(config_key):
    load_dotenv()
    try:
        flask_env = os.environ["FLASK_ENV"]
    except KeyError:
        return "ERROR"
    file_names = [flask_env]
    mod = __import__("config", fromlist=file_names)
    for file in file_names:
        config_class = getattr(mod, file)
        config_value = getattr(config_class, config_key, None)
        if config_value:
            return config_value
