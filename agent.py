import os
import platform
import time

from openai import OpenAI

from config.config import load_config
from core.agent_loop import run_main_loop
from core.message_store import append_message, calc_total_tokens
from core.message_store import trim_messages as _trim_messages
from core.token_counter import count_message_tokens as _count_message_tokens
from core.token_counter import estimate_tokens as _estimate_tokens
from log import get_logger

CONFIG = load_config()
_CLIENT = None

MODEL = CONFIG["model"]
BASE_SYSTEM = CONFIG["base_system"]
KEEP_LAST = CONFIG.get("keep_last", 4000)
TOOL_DEFS = CONFIG["tool_defs"]


def get_client():
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = OpenAI(
            api_key=os.environ.get("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com",
        )
    return _CLIENT


def estimate_tokens(text: str) -> int:
    return _estimate_tokens(text, MODEL)


def count_message_tokens(message: dict) -> int:
    return _count_message_tokens(message, MODEL)


def trim_messages(messages, keep_last=KEEP_LAST, cached_total_tokens=None, return_total=False):
    return _trim_messages(
        messages,
        keep_last=keep_last,
        model=MODEL,
        cached_total_tokens=cached_total_tokens,
        return_total=return_total,
    )


def get_system_info():
    system_info = {
        "os": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }

    try:
        with open("/proc/version", "r", encoding="utf-8") as file:
            proc_version = file.read()
            if "microsoft" in proc_version.lower():
                system_info["is_wsl"] = True
                system_info["wsl_version"] = "WSL2" if "WSL2" in proc_version else "WSL1"
            else:
                system_info["is_wsl"] = False
    except Exception:
        system_info["is_wsl"] = False

    lines = [
        f"操作系统: {system_info['os']}",
        f"系统版本: {system_info['release']}",
        f"平台架构: {system_info['machine']}",
        f"处理器: {system_info['processor']}",
        f"Python版本: {system_info['python_version']}",
    ]
    if system_info.get("is_wsl"):
        lines.append(f"运行环境: {system_info['wsl_version']} (Windows Subsystem for Linux)")
    return "\n".join(lines)


def sanitize_text(s):
    if s is None:
        return ""
    if not isinstance(s, str):
        s = str(s)
    return s.encode("utf-8", "replace").decode("utf-8")


def run_agent(task: str, max_steps: int = 18, enable_thinking_stream: bool = True, session=None):
    logger = get_logger()

    messages = session.copy() if session else []
    total_tokens = calc_total_tokens(messages, MODEL)

    messages, total_tokens = append_message(
        messages,
        {"role": "user", "content": task},
        total_tokens,
        keep_last=KEEP_LAST,
        model=MODEL,
        auto_trim=True,
    )

    t0 = time.perf_counter()
    logger.info("开始处理任务: {}", task)

    step, content, messages, total_tokens = run_main_loop(
        client=get_client(),
        model=MODEL,
        tool_defs=TOOL_DEFS,
        keep_last=KEEP_LAST,
        max_steps=max_steps,
        enable_thinking_stream=enable_thinking_stream,
        messages=messages,
        total_tokens=total_tokens,
    )

    dt = time.perf_counter() - t0
    logger.info("[Step {}] {}  tot time: {:.3f}s", step, content if content else "(empty)", dt)
    logger.info("任务处理完成，总耗时: {:.3f}s", dt)

    return messages


if __name__ == "__main__":
    logger = get_logger()
    logger.info("Agent启动")

    system_info = get_system_info()
    full_system = BASE_SYSTEM + f"\n\n当前工作目录：{os.getcwd()}\n\n系统信息：\n{system_info}"

    session = [{"role": "system", "content": full_system}]

    while True:
        try:
            query = sanitize_text(str(input("> ").strip()))
            if not query:
                continue
            session = run_agent(query, session=session)
        except KeyboardInterrupt:
            logger.info("用户中断，退出程序")
            break
        except Exception as exc:
            logger.error("运行出错: {}", str(exc))
