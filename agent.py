import json
import os
import platform
import re
import time
from types import SimpleNamespace

from openai import OpenAI

from config.config import load_config
from log import end_thinking_stream, get_logger, start_thinking_stream, stream_thinking
from tools import TOOLS

CONFIG = load_config()

client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)

MODEL = CONFIG["model"]
BASE_SYSTEM = CONFIG["base_system"]
TOOL_DEFS = CONFIG["tool_defs"]

CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)
FINAL_RE = re.compile(r"<final>\s*(.*?)\s*</final>", re.S)


def parse_tool_call(content: str):
    match = CALL_RE.search(content or "")
    if not match:
        return None
    obj = json.loads(match.group(1))
    return obj["name"], obj.get("arguments", {})


def parse_final(content: str):
    match = FINAL_RE.search(content or "")
    return match.group(1) if match else None


def trim_messages(messages, keep_last=20):
    system_message = None
    rest = []
    for message in messages:
        if message.get("role") == "system" and system_message is None:
            system_message = message
        else:
            rest.append(message)
    return ([system_message] if system_message else []) + rest[-keep_last:]


def deepseek_chat(messages, stream_thinking_callback=None):
    logger = get_logger()

    is_stream = bool(stream_thinking_callback)

    try:
        if not is_stream:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=TOOL_DEFS,
                tool_choice="auto",
                temperature=0.2,
                stream=False,
            )
            return resp.choices[0].message

        start_thinking_stream()
        resp = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOL_DEFS,
            tool_choice="auto",
            temperature=0.2,
            stream=True,
        )

        content_parts = []
        reasoning_parts = []
        tool_call_chunks = {}

        for chunk in resp:
            choice = chunk.choices[0]
            delta = choice.delta

            reasoning_chunk = getattr(delta, "reasoning_content", None)
            if reasoning_chunk:
                reasoning_parts.append(reasoning_chunk)
                stream_thinking(reasoning_chunk)
                stream_thinking_callback(reasoning_chunk)

            content_chunk = getattr(delta, "content", None)
            if content_chunk:
                content_parts.append(content_chunk)

            for tc in getattr(delta, "tool_calls", None) or []:
                index = getattr(tc, "index", 0)
                entry = tool_call_chunks.setdefault(
                    index,
                    {"id": getattr(tc, "id", None), "name": "", "arguments": ""},
                )
                if getattr(tc, "id", None):
                    entry["id"] = tc.id
                function = getattr(tc, "function", None)
                if not function:
                    continue
                if getattr(function, "name", None):
                    entry["name"] += function.name
                if getattr(function, "arguments", None):
                    entry["arguments"] += function.arguments

        tool_calls = [
            SimpleNamespace(
                id=entry["id"] or f"call_{index}",
                function=SimpleNamespace(
                    name=entry["name"],
                    arguments=entry["arguments"] or "{}",
                ),
            )
            for index, entry in sorted(tool_call_chunks.items())
        ]

        return SimpleNamespace(
            content="".join(content_parts),
            tool_calls=tool_calls,
            reasoning_content="".join(reasoning_parts),
        )
    except Exception as exc:
        logger.error("DeepSeek API调用失败: {}", str(exc))
        raise
    finally:
        if is_stream:
            end_thinking_stream()


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
    # 把 surrogate / 非法字符替换掉，确保能 utf-8 编码
    return s.encode("utf-8", "replace").decode("utf-8")

def run_agent(task: str, max_steps: int = 18, enable_thinking_stream: bool = True):
    logger = get_logger()
    system_info = get_system_info()
    full_system = BASE_SYSTEM + f"\n\n当前工作目录：{os.getcwd()}\n\n系统信息：\n{system_info}"

    messages = [
        {"role": "system", "content": full_system},
        {"role": "user", "content": task},
    ]

    t0 = time.perf_counter()
    logger.info("开始处理任务: {}", task)

    for step in range(1, max_steps + 1):
        thinking_callback = (lambda chunk: None) if enable_thinking_stream else None
        msg = deepseek_chat(messages, thinking_callback)

        tool_calls = getattr(msg, "tool_calls", None) or []
        if tool_calls:
            messages.append(
                {
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in tool_calls
                    ],
                }
            )

            tc = tool_calls[0]
            name = tc.function.name
            args = tc.function.arguments or "{}"
            if isinstance(args, str):
                args = json.loads(args)

            if name not in TOOLS:
                result = {"ok": False, "output": f"unknown tool: {name}"}
                logger.error("未知工具: {}", name)
            else:
                logger.info("调用工具: {}", name)
                result = TOOLS[name](**args)

            if result.get("ok"):
                logger.info("工具 {} 执行成功", name)
            else:
                logger.error("工具 {} 执行失败: {}", name, result.get("output", ""))

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": name,
                    "content": result.get("output", ""),
                }
            )
            messages = trim_messages(messages, keep_last=30)
            continue

        content = (getattr(msg, "content", None) or "").strip()
        messages.append({"role": "assistant", "content": content})

        dt = time.perf_counter() - t0
        logger.info("[Step {}] {}  tot time: {:.3f}s", step, content if content else "(empty)", dt)

        final_text = parse_final(content)
        if final_text is not None:
            logger.info("任务完成，收到最终回复")
            print(final_text)
            break

        maybe = parse_tool_call(content)
        if maybe:
            name, args = maybe
            if name not in TOOLS:
                result = {"ok": False, "output": f"unknown tool: {name}"}
                logger.error("未知工具(标签): {}", name)
            else:
                logger.info("调用工具(标签): {}", name)
                result = TOOLS[name](**args)

            if result.get("ok"):
                logger.info("工具 {} 执行成功(标签)", name)
            else:
                logger.error("工具 {} 执行失败(标签): {}", name, result.get("output", ""))

            messages.append({"role": "tool", "name": name, "content": result.get("output", "")})
            messages = trim_messages(messages, keep_last=30)
            continue

        if content:
            print(content)
            break

        logger.warning("模型未返回可用内容，结束本轮任务")
        break

    total_time = time.perf_counter() - t0
    logger.info("任务处理完成，总耗时: {:.3f}s", total_time)


if __name__ == "__main__":
    logger = get_logger()
    logger.info("Agent启动")

    while True:
        try:
            query = sanitize_text(str(input("> ").strip()))
            if not query:
                continue
            run_agent(query)
        except KeyboardInterrupt:
            logger.info("用户中断，退出程序")
            break
        except Exception as exc:
            logger.error("运行出错: {}", str(exc))
