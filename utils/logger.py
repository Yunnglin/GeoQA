import json
import logging.config
import os
import warnings


def setup_logging(default_path="./logger_config.json", default_level=logging.INFO, env_key="LOG_CFG"):
    warnings.filterwarnings("ignore")
    path = default_path
    value = os.getenv(env_key, None)
    if value:
        path = value
    if os.path.exists(path):
        with open(path, "r") as f:
            config = json.load(f)
            # TODO : can't configure handler 'info_file_handler'
            logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)
