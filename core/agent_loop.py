import json
import re

from log import get_logger
from core.tools import TOOLS
from core.llm_client import deepseek_chat
from core.message_store import append_message

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


def handle_tool_call(tc, name, args, messages, total_tokens, keep_last, model):
    logger = get_logger()

    if isinstance(args, str):
        args = json.loads(args)

    if name not in TOOLS:
        result = {"ok": False, "output": f"unknown tool: {name}"}
        logger.error("Unknown tool: {}", name)
    else:
        logger.info("Call tool: {}", name)
        result = TOOLS[name](**args)

    if result.get("ok"):
        logger.info("Tool {} succeeded", name)
    else:
        logger.error("Tool {} failed: {}", name, result.get("output", ""))

    messages, total_tokens = append_message(
        messages,
        {
            "role": "tool",
            "tool_call_id": tc.id,
            "name": name,
            "content": result.get("output", ""),
        },
        total_tokens,
        keep_last=keep_last,
        model=model,
        auto_trim=True,
    )
    return messages, total_tokens


def run_main_loop(
    client,
    model,
    tool_defs,
    keep_last,
    max_steps,
    enable_thinking_stream,
    messages,
    total_tokens,
    echo_output=True,
):
    logger = get_logger()

    for step in range(1, max_steps + 1):
        thinking_callback = (lambda chunk: None) if enable_thinking_stream else None
        msg = deepseek_chat(client, model, tool_defs, messages, thinking_callback)

        tool_calls = getattr(msg, "tool_calls", None) or []
        if tool_calls:
            messages, total_tokens = append_message(
                messages,
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
                },
                total_tokens,
                keep_last=keep_last,
                model=model,
                auto_trim=True,
            )

            tc = tool_calls[0]
            name = tc.function.name
            args = tc.function.arguments or "{}"
            messages, total_tokens = handle_tool_call(
                tc, name, args, messages, total_tokens, keep_last=keep_last, model=model
            )
            continue

        content = (getattr(msg, "content", None) or "").strip()
        messages, total_tokens = append_message(
            messages,
            {"role": "assistant", "content": content},
            total_tokens,
            keep_last=keep_last,
            model=model,
            auto_trim=True,
        )

        final_text = parse_final(content)
        if final_text is not None:
            logger.info("Task completed with final tag")
            if echo_output:
                print(final_text)
            return step, content, messages, total_tokens

        maybe = parse_tool_call(content)
        if maybe:
            name, args = maybe
            if name not in TOOLS:
                result = {"ok": False, "output": f"unknown tool: {name}"}
                logger.error("Unknown tagged tool: {}", name)
            else:
                logger.info("Call tagged tool: {}", name)
                result = TOOLS[name](**args)

            if result.get("ok"):
                logger.info("Tagged tool {} succeeded", name)
            else:
                logger.error("Tagged tool {} failed: {}", name, result.get("output", ""))

            messages, total_tokens = append_message(
                messages,
                {"role": "tool", "name": name, "content": result.get("output", "")},
                total_tokens,
                keep_last=keep_last,
                model=model,
                auto_trim=True,
            )
            continue

        if content:
            if echo_output:
                print(content)
            return step, content, messages, total_tokens

        logger.warning("Model returned no usable content; ending task")
        return step, content, messages, total_tokens

    return max_steps, "", messages, total_tokens
