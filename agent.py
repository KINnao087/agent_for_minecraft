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
API_KEY = os.environ.get("DEEPSEEK_API_KEY") or CONFIG.get("api_key")


def _preview_text(value, limit=300):
    text = "" if value is None else str(value)
    text = text.replace("\r", "\\r").replace("\n", "\\n")
    return text if len(text) <= limit else text[:limit] + "...(truncated)"


def get_client():
    global _CLIENT
    if _CLIENT is None:
        if not API_KEY:
            raise RuntimeError("Missing DeepSeek API key. Set DEEPSEEK_API_KEY or config.api_key.")
        _CLIENT = OpenAI(
            api_key=API_KEY,
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
        f"OS: {system_info['os']}",
        f"Release: {system_info['release']}",
        f"Machine: {system_info['machine']}",
        f"Processor: {system_info['processor']}",
        f"Python: {system_info['python_version']}",
    ]
    if system_info.get("is_wsl"):
        lines.append(f"Runtime: {system_info['wsl_version']} (Windows Subsystem for Linux)")
    return "\n".join(lines)


def build_initial_session():
    system_info = get_system_info()
    full_system = BASE_SYSTEM + f"\n\nCurrent working directory: {os.getcwd()}\n\nSystem info:\n{system_info}"
    return [{"role": "system", "content": full_system}]


def sanitize_text(s):
    if s is None:
        return ""
    if not isinstance(s, str):
        s = str(s)
    return s.encode("utf-8", "replace").decode("utf-8")


def get_last_assistant_text(messages):
    for message in reversed(messages):
        if message.get("role") != "assistant":
            continue
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()
    return ""


def run_agent(task: str, max_steps: int = 18, enable_thinking_stream: bool = True, session=None, echo_output=True):
    logger = get_logger()

    messages = session.copy() if session else build_initial_session()
    total_tokens = calc_total_tokens(messages, MODEL)
    logger.info(
        "Prepare agent run: max_steps={}, stream={}, existing_messages={}, total_tokens={}",
        max_steps,
        enable_thinking_stream,
        len(messages),
        total_tokens,
    )

    messages, total_tokens = append_message(
        messages,
        {"role": "user", "content": task},
        total_tokens,
        keep_last=KEEP_LAST,
        model=MODEL,
        auto_trim=True,
    )

    t0 = time.perf_counter()
    logger.info("Start task: {}", task)
    logger.info("User task preview: {}", _preview_text(task))

    step, content, messages, total_tokens = run_main_loop(
        client=get_client(),
        model=MODEL,
        tool_defs=TOOL_DEFS,
        keep_last=KEEP_LAST,
        max_steps=max_steps,
        enable_thinking_stream=enable_thinking_stream,
        messages=messages,
        total_tokens=total_tokens,
        echo_output=echo_output,
    )

    dt = time.perf_counter() - t0
    logger.info("Assistant final preview: {}", _preview_text(get_last_assistant_text(messages)))
    logger.info("[Step {}] {} total time: {:.3f}s", step, content if content else "(empty)", dt)
    logger.info("Task completed in {:.3f}s", dt)

    return messages


def run_agent_and_get_reply(task: str, max_steps: int = 18, enable_thinking_stream: bool = False, session=None):
    messages = run_agent(
        task=task,
        max_steps=max_steps,
        enable_thinking_stream=enable_thinking_stream,
        session=session,
        echo_output=False,
    )
    return get_last_assistant_text(messages), messages


if __name__ == "__main__":
    logger = get_logger()
    logger.info("Agent started")

    session = build_initial_session()

    while True:
        try:
            query = sanitize_text(str(input("> ").strip()))
            if not query:
                continue
            session = run_agent(query, session=session)
        except KeyboardInterrupt:
            logger.info("User interrupted, exiting")
            break
        except Exception as exc:
            logger.error("Runtime error: {}", str(exc))
