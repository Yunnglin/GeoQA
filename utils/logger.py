import json
import logging.config
import os
import warnings
import yaml


def setup_logging(default_path="./logger_config.yaml", default_level=logging.INFO, env_key="LOG_CFG"):
    warnings.filterwarnings("ignore")
    path = default_path
    value = os.getenv(env_key, None)
    if value:
        path = value
    if os.path.exists(path):
        with open(path, "r") as f:
            # 根据后缀来判断加载的配置
            config = json.load(f) if path.endswith("json") else yaml.load(stream=f, Loader=yaml.FullLoader)
            logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)
