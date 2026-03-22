from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class ToolFunction:
    name: str
    arguments: str


@dataclass
class ToolCall:
    id: str
    function: ToolFunction


@dataclass
class ChatResult:
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    reasoning_content: str = ""


class BaseChatProvider(ABC):
# 定义 provider 必须实现的对话接口。
    @abstractmethod
    def chat(
        self,
        model: str,
        tool_defs: list[dict],
        messages: list[dict],
        stream_thinking_callback: Optional[Callable[[str], None]] = None,
    ) -> ChatResult:
        raise NotImplementedError
