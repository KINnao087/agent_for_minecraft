from __future__ import annotations

from typing import Callable, Optional

from log import end_thinking_stream, get_logger, start_thinking_stream, stream_thinking

from core.web_to_api.base import BaseChatProvider, ChatResult, ToolCall, ToolFunction


class DeepSeekApiProvider(BaseChatProvider):
    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com", temperature: float = 0.2):
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._temperature = temperature

    def chat(
        self,
        model: str,
        tool_defs: list[dict],
        messages: list[dict],
        stream_thinking_callback: Optional[Callable[[str], None]] = None,
    ) -> ChatResult:
        logger = get_logger()
        is_stream = bool(stream_thinking_callback)

        try:
            if not is_stream:
                resp = self._client.chat.completions.create(
                    model=model,
                    messages=messages,
                    tools=tool_defs,
                    tool_choice="auto",
                    temperature=self._temperature,
                    stream=False,
                )
                msg = resp.choices[0].message
                tool_calls = [
                    ToolCall(
                        id=tc.id,
                        function=ToolFunction(
                            name=tc.function.name,
                            arguments=tc.function.arguments or "{}",
                        ),
                    )
                    for tc in (getattr(msg, "tool_calls", None) or [])
                ]
                return ChatResult(
                    content=getattr(msg, "content", "") or "",
                    tool_calls=tool_calls,
                    reasoning_content=getattr(msg, "reasoning_content", "") or "",
                )

            start_thinking_stream()
            resp = self._client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tool_defs,
                tool_choice="auto",
                temperature=self._temperature,
                stream=True,
            )

            content_parts: list[str] = []
            reasoning_parts: list[str] = []
            tool_call_chunks: dict[int, dict[str, str | None]] = {}

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
                ToolCall(
                    id=(entry["id"] or f"call_{index}"),
                    function=ToolFunction(
                        name=str(entry["name"] or ""),
                        arguments=str(entry["arguments"] or "{}"),
                    ),
                )
                for index, entry in sorted(tool_call_chunks.items())
            ]

            return ChatResult(
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
