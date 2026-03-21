import json
import re

from log import get_logger
from core.tools import TOOLS
from core.llm_client import deepseek_chat
from core.message_store import append_message

CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)
FINAL_RE = re.compile(r"<final>\s*(.*?)\s*</final>", re.S)
EDIT_INTENT_RE = re.compile(
    r"(修改|改(一)?下|编辑|更新|重构|修复|添加注释|增加注释|写回|回写|改代码|改文件|注释|"
    r"\bedit\b|\bmodify\b|\bchange\b|\bupdate\b|\brewrite\b|\brefactor\b|\bfix\b|\bcomment\b)",
    re.I,
)
QUESTION_INTENT_RE = re.compile(r"(怎么|如何|为什么|为啥|\?|？|\bhow\b|\bwhy\b)", re.I)
WRITE_ENFORCEMENT_TEXT = (
    "The user requested a file modification. You must use a tool_call to apply the change. "
    "If write_file is available, use write_file instead of pasting modified code directly."
)
INVALID_TOOL_CALL_JSON_TEXT = (
    "Your previous <tool_call> JSON was invalid. Return exactly one corrected <tool_call> tag with strict valid JSON only. "
    "Escape all string values correctly. If you call write_file, the arguments.content field must be a valid JSON string with escaped newlines and quotes."
)


# Build a compact text preview for loop logs.
def _preview_text(value, limit=300):
    text = "" if value is None else str(value)
    text = text.replace("\r", "\\r").replace("\n", "\\n")
    return text if len(text) <= limit else text[:limit] + "...(truncated)"

#解析工具调用
# Extract one tagged tool call from assistant content.
def parse_tool_call(content: str):
    match = CALL_RE.search(content or "")
    if not match:
        return None

    payload = match.group(1)
    try:
        obj = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "Invalid tool_call JSON at "
            f"line {exc.lineno} column {exc.colno}: {exc.msg}. "
            f"Payload preview: {_preview_text(payload, 500)}"
        ) from exc

    name = obj.get("name")
    if not name:
        raise ValueError(f"tool_call is missing name. Payload preview: {_preview_text(payload, 500)}")
    return name, obj.get("arguments", {})


# Extract the final tagged answer from assistant content.
def parse_final(content: str):
    match = FINAL_RE.search(content or "")
    return match.group(1) if match else None


# Return the most recent user message text.
def _latest_user_text(messages) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            content = message.get("content")
            return content if isinstance(content, str) else str(content or "")
    return ""


# Collect available tool names from tool definitions.
def _tool_names(tool_defs) -> set[str]:
    names = set()
    for tool in tool_defs or []:
        function = tool.get("function") if isinstance(tool, dict) else None
        name = function.get("name") if isinstance(function, dict) else None
        if name:
            names.add(str(name))
    return names


# Detect when the latest user request likely needs a file write.
def _user_request_requires_file_write(messages, tool_defs) -> bool:
    latest_user = _latest_user_text(messages).strip()
    if not latest_user:
        return False
    if not EDIT_INTENT_RE.search(latest_user):
        return False
    if QUESTION_INTENT_RE.search(latest_user):
        return False
    return "write_file" in _tool_names(tool_defs)


# Check whether a specific system reminder is already present.
def _has_system_note(messages, note: str) -> bool:
    return any(message.get("role") == "system" and message.get("content") == note for message in messages)


# Execute one tool call and append its result to the history.
def handle_tool_call(tc, name, args, messages, total_tokens, keep_last, model):
    logger = get_logger()

    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError as exc:
            result = {
                "ok": False,
                "output": (
                    f"invalid tool call arguments for {name}: line {exc.lineno} column {exc.colno}: {exc.msg}. "
                    f"Arguments preview: {_preview_text(args, 500)}"
                ),
            }
            logger.error("Malformed tool call args for {}: {}", name, result["output"])
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

    logger.info("Tool call args for {}: {}", name, _preview_text(json.dumps(args, ensure_ascii=False)))

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
    logger.info("Tool {} output preview: {}", name, _preview_text(result.get("output", "")))

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


# Drive the agent loop until completion or step limit.
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
        logger.info("Agent loop step {} start, message_count={}", step, len(messages))
        thinking_callback = (lambda chunk: None) if enable_thinking_stream else None
        msg = deepseek_chat(client, model, tool_defs, messages, thinking_callback)

        tool_calls = getattr(msg, "tool_calls", None) or []
        if tool_calls:
            logger.info(
                "Model requested {} tool call(s), assistant content preview: {}",
                len(tool_calls),
                _preview_text(getattr(msg, "content", "") or ""),
            )

            #将消息加入历史消息列表
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
            logger.info("Executing first tool call: {}", name)
            messages, total_tokens = handle_tool_call(
                tc, name, args, messages, total_tokens, keep_last=keep_last, model=model
            )
            continue

        content = (getattr(msg, "content", None) or "").strip()
        logger.info("Model content preview at step {}: {}", step, _preview_text(content))

        if content and _user_request_requires_file_write(messages, tool_defs):
            logger.warning("Edit request returned direct content instead of a write tool call; enforcing retry")
            if not _has_system_note(messages, WRITE_ENFORCEMENT_TEXT):
                messages, total_tokens = append_message(
                    messages,
                    {"role": "system", "content": WRITE_ENFORCEMENT_TEXT},
                    total_tokens,
                    keep_last=keep_last,
                    model=model,
                    auto_trim=True,
                )
                continue

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
            logger.info("Final tag content preview: {}", _preview_text(final_text))
            if echo_output:
                print(final_text)
            return step, content, messages, total_tokens

        try:
            maybe = parse_tool_call(content)
        except ValueError as exc:
            logger.error("Malformed tagged tool call: {}", str(exc))
            messages, total_tokens = append_message(
                messages,
                {"role": "system", "content": INVALID_TOOL_CALL_JSON_TEXT},
                total_tokens,
                keep_last=keep_last,
                model=model,
                auto_trim=True,
            )
            continue

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
            logger.info("Tagged tool {} output preview: {}", name, _preview_text(result.get("output", "")))

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
            logger.info("Returning assistant content without final tag: {}", _preview_text(content))
            if echo_output:
                print(content)
            return step, content, messages, total_tokens

        logger.warning("Model returned no usable content; ending task")
        return step, content, messages, total_tokens

    return max_steps, "", messages, total_tokens
