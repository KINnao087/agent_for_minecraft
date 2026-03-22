from __future__ import annotations

from typing import Callable, Optional

from log import get_logger

from core.web_to_api.base import BaseChatProvider, ChatResult
from core.web_to_api.deepseek_web_client import DeepSeekWebClient
from core.web_to_api.message_prompt import build_web_chat_prompt


class DeepSeekWebProvider(BaseChatProvider):
# 将浏览器客户端封装为对话 provider。
    def __init__(self, web_client: DeepSeekWebClient):
        self._web_client = web_client

# 通过浏览器工作流发送一次对话请求。
    def chat(
        self,
        model: str,
        tool_defs: list[dict],
        messages: list[dict],
        stream_thinking_callback: Optional[Callable[[str], None]] = None,
    ) -> ChatResult:
        logger = get_logger()
        if stream_thinking_callback:
            logger.warning("DeepSeekWebProvider does not support thinking stream; callback ignored")

        prompt = build_web_chat_prompt(messages=messages, tool_defs=tool_defs, model=model)
        logger.info("DeepSeek web prompt preview: {}", self._preview_text(prompt))
        reply_text = self._web_client.ask(prompt)
        return ChatResult(content=reply_text.strip(), tool_calls=[], reasoning_content="")

# 生成网页 provider 日志用的精简预览。
    @staticmethod
    def _preview_text(value: str, limit: int = 300) -> str:
        text = "" if value is None else str(value)
        text = text.replace("\r", "\\r").replace("\n", "\\n")
        return text if len(text) <= limit else text[:limit] + "...(truncated)"
