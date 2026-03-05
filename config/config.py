import json
from pathlib import Path

from log import get_logger


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
                raise ValueError(f"配置文件为空: {config_path}")
            config = json.loads(content)
            if not isinstance(config, dict):
                raise TypeError(f"配置文件必须是 JSON object: {config_path}")
    except FileNotFoundError:
        logger.error("未找到配置文件: {}", config_path)
        raise
    except json.JSONDecodeError as exc:
        logger.error("配置文件 JSON 解析失败: {} ({})", config_path, str(exc))
        raise

    required_keys = ("model", "base_system", "tool_defs")
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise KeyError(f"配置文件缺少必要字段: {', '.join(missing_keys)}")

    if not isinstance(config["model"], str) or not config["model"].strip():
        raise TypeError("配置项 model 必须是非空字符串")
    if not isinstance(config["base_system"], str) or not config["base_system"].strip():
        raise TypeError("配置项 base_system 必须是非空字符串")
    if not isinstance(config["tool_defs"], list):
        raise TypeError("配置项 tool_defs 必须是数组")

    return config
