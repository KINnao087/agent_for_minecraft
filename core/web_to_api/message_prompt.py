from __future__ import annotations

import json


# 将结构化值转换为适合 prompt 的文本。
def _stringify(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, indent=2)
    except Exception:
        return str(value)


# 序列化工具定义供网页 prompt 使用。
def _format_tool_defs(tool_defs: list[dict]) -> str:
    if not tool_defs:
        return "[]"
    return json.dumps(tool_defs, ensure_ascii=False, indent=2)


# 序列化一条历史消息供网页 prompt 使用。
def _format_message(message: dict, index: int) -> str:
    role = message.get("role", "unknown")
    header = [f"## Message {index}", f"role: {role}"]

    name = message.get("name")
    if name:
        header.append(f"name: {name}")

    tool_call_id = message.get("tool_call_id")
    if tool_call_id:
        header.append(f"tool_call_id: {tool_call_id}")

    tool_calls = message.get("tool_calls")
    body_parts = []
    if tool_calls:
        body_parts.append("tool_calls:")
        body_parts.append(_stringify(tool_calls))

    body_parts.append("content:")
    body_parts.append(_stringify(message.get("content", "")))

    return "\n".join(header + [""] + body_parts)


# 根据历史消息和工具构建网页模式 prompt。
def build_web_chat_prompt(messages: list[dict], tool_defs: list[dict], model: str) -> str:
    history = "\n\n".join(_format_message(message, index) for index, message in enumerate(messages, start=1))
    tools_text = _format_tool_defs(tool_defs)

    return (
        "You are continuing an existing agent conversation inside a browser chat UI.\n"
        "The conversation history is serialized below with explicit role labels.\n"
        "System messages have the highest priority and must be followed.\n"
        "Return exactly one assistant turn for the latest state of the conversation.\n"
        "\n"
        "Output rules:\n"
        "1. If a tool is needed, output exactly one XML-like tag in this format:\n"
        '<tool_call>{"name":"tool_name","arguments":{}}</tool_call>\n'
        "2. If you want to answer the user, output exactly one XML-like tag in this format:\n"
        "<final>your answer</final>\n"
        "3. Do not wrap the result in Markdown code fences.\n"
        "4. Do not explain the protocol.\n"
        "5. If the user asks to create, modify, rename, move, or delete project files, you must output a tool_call first.\n"
        "6. If write_file is available and a file must be changed, use write_file instead of pasting the modified code directly in <final>.\n"
        "7. The JSON inside <tool_call> must be strictly valid. Escape quotes, backslashes, and newlines inside string arguments. This is especially important for write_file.arguments.content.\n"
        "\n"
        f"Target model label: {model}\n"
        "\n"
        "Available tools JSON:\n"
        f"{tools_text}\n"
        "\n"
        "Conversation history:\n"
        f"{history}\n"
        "\n"
        "Now produce the next assistant message only."
    )


