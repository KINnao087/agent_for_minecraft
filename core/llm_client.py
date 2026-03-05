from types import SimpleNamespace

from log import end_thinking_stream, get_logger, start_thinking_stream, stream_thinking


def deepseek_chat(client, model, tool_defs, messages, stream_thinking_callback=None):
    logger = get_logger()
    is_stream = bool(stream_thinking_callback)

    try:
        if not is_stream:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tool_defs,
                tool_choice="auto",
                temperature=0.2,
                stream=False,
            )
            return resp.choices[0].message

        start_thinking_stream()
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tool_defs,
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

