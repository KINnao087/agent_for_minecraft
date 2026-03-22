import json
from pathlib import Path

from log import get_logger


# 加载并校验项目配置文件。
def load_config(config_path=None):
    logger = get_logger()
    if config_path is None:
        config_path = Path(__file__).with_name("config.json")
    else:
        config_path = Path(config_path)
        if not config_path.is_absolute():
            config_path = Path.cwd() / config_path

    try:
        with open(config_path, "r", encoding="utf-8") as file:
            content = file.read().strip()
            if not content:
                raise ValueError(f"config file is empty: {config_path}")
            config = json.loads(content)
            if not isinstance(config, dict):
                raise TypeError(f"config file must be a JSON object: {config_path}")
    except FileNotFoundError:
        logger.error("config file not found: {}", config_path)
        raise
    except json.JSONDecodeError as exc:
        logger.error("failed to parse config json: {} ({})", config_path, str(exc))
        raise

    required_keys = ("model", "base_system", "tool_defs")
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise KeyError(f"config file missing required keys: {', '.join(missing_keys)}")

    if not isinstance(config["model"], str) or not config["model"].strip():
        raise TypeError("config field 'model' must be a non-empty string")
    if not isinstance(config["base_system"], str) or not config["base_system"].strip():
        raise TypeError("config field 'base_system' must be a non-empty string")
    if not isinstance(config["tool_defs"], list):
        raise TypeError("config field 'tool_defs' must be a list")
    if "provider" in config and (not isinstance(config["provider"], str) or not config["provider"].strip()):
        raise TypeError("config field 'provider' must be a non-empty string")
    if "api_key" in config and config["api_key"] is not None and not isinstance(config["api_key"], str):
        raise TypeError("config field 'api_key' must be a string or null")
    if "deepseek_api" in config and config["deepseek_api"] is not None and not isinstance(config["deepseek_api"], dict):
        raise TypeError("config field 'deepseek_api' must be an object or null")
    if "deepseek_web" in config and config["deepseek_web"] is not None and not isinstance(config["deepseek_web"], dict):
        raise TypeError("config field 'deepseek_web' must be an object or null")

    return config
