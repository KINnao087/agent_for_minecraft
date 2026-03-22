#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lightweight colored logger with optional streamed thinking output.
"""

from __future__ import annotations

import inspect
import sys
import time
from enum import Enum
from pathlib import Path
from typing import Callable, Optional


class LogLevel(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    AGENT = "AGENT"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
    THINKING = "THINKING"


class Color:
    RESET = "\033[0m"
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    BRIGHT_BLACK = "\033[90m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"

    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"
    BG_MAGENTA = "\033[45m"
    BG_CYAN = "\033[46m"
    BG_WHITE = "\033[47m"


class Logger:
# 初始化 logger 状态和流式缓冲区。
    def __init__(self, name: str = "agent"):
        self.name = name
        self.level = LogLevel.INFO
        self.enable_color = sys.stdout.isatty()
        self.thinking_callback: Optional[Callable[[str], None]] = None
        self.thinking_buffer = ""

# 返回指定日志级别对应的 ANSI 颜色。
    def _get_color(self, level: LogLevel) -> str:
        if not self.enable_color:
            return ""

        color_map = {
            LogLevel.DEBUG: Color.BRIGHT_BLACK,
            LogLevel.INFO: Color.BRIGHT_BLUE,
            LogLevel.AGENT: Color.GREEN,
            LogLevel.WARNING: Color.BRIGHT_YELLOW,
            LogLevel.ERROR: Color.BRIGHT_RED,
            LogLevel.CRITICAL: Color.RED + Color.BG_WHITE,
            LogLevel.THINKING: Color.BRIGHT_MAGENTA,
        }
        return color_map.get(level, Color.RESET)

# 组装包含时间、级别和调用位置的日志行。
    def _format_message(self, level: LogLevel, message: str) -> str:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        color = self._get_color(level)
        reset = Color.RESET if self.enable_color else ""
        location = self._caller_location()
        return f"{color}[{timestamp}] [{level.value}] [{self.name}] [{location}] {message}{reset}"

# 判断当前级别是否需要输出日志。
    def _should_log(self, level: LogLevel) -> bool:
        level_order = {
            LogLevel.DEBUG: 0,
            LogLevel.INFO: 1,
            LogLevel.AGENT: 1,
            LogLevel.WARNING: 2,
            LogLevel.ERROR: 3,
            LogLevel.CRITICAL: 4,
            LogLevel.THINKING: 5,
        }
        return level_order.get(level, 0) >= level_order.get(self.level, 0)

# 渲染并输出一条日志消息。
    def log(self, level: LogLevel, message: str, *args, **kwargs):
        if not self._should_log(level):
            return

        rendered_message = self._render_message(message, *args, **kwargs)
        formatted_msg = self._format_message(level, rendered_message)
        output = sys.stderr if level in [LogLevel.ERROR, LogLevel.CRITICAL] else sys.stdout
        print(formatted_msg, file=output)

# 安全地格式化带参数的日志消息。
    def _render_message(self, message: object, *args, **kwargs) -> str:
        if message is None:
            text = ""
        elif isinstance(message, str):
            text = message
        else:
            text = str(message)

        if not args and not kwargs:
            return text

        try:
            return text.format(*args, **kwargs)
        except Exception as exc:
            return (
                f"{text} [log format error: {exc}; "
                f"args={args!r}; kwargs={kwargs!r}]"
            )

# 返回第一个非 logger 内部的调用位置。
    def _caller_location(self) -> str:
        current_file = Path(__file__).resolve()
        frame = inspect.currentframe()
        try:
            if frame is not None:
                frame = frame.f_back
            while frame is not None:
                filename = Path(frame.f_code.co_filename).resolve()
                if filename != current_file:
                    return f"{filename.name}:{frame.f_lineno}"
                frame = frame.f_back
            return "unknown:0"
        finally:
            del frame

# 输出调试日志。
    def debug(self, message: str, *args, **kwargs):
        self.log(LogLevel.DEBUG, message, *args, **kwargs)

# 输出普通信息日志。
    def info(self, message: str, *args, **kwargs):
        self.log(LogLevel.INFO, message, *args, **kwargs)

# 输出 agent 专用日志。
    def agent(self, message: str, *args, **kwargs):
        self.log(LogLevel.AGENT, message, *args, **kwargs)

# 输出警告日志。
    def warning(self, message: str, *args, **kwargs):
        self.log(LogLevel.WARNING, message, *args, **kwargs)

# 输出错误日志。
    def error(self, message: str, *args, **kwargs):
        self.log(LogLevel.ERROR, message, *args, **kwargs)

# 输出严重错误日志。
    def critical(self, message: str, *args, **kwargs):
        self.log(LogLevel.CRITICAL, message, *args, **kwargs)

# 输出思考流日志。
    def thinking(self, message: str, *args, **kwargs):
        self.log(LogLevel.THINKING, message, *args, **kwargs)

# 开始输出单行思考流。
    def start_thinking_stream(self):
        self.thinking_buffer = ""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        color = self._get_color(LogLevel.THINKING)
        prefix = f"{color}[{timestamp}] [{LogLevel.THINKING.value}] [{self.name}] "
        print(prefix, end="", flush=True)

# 追加一段思考流内容。
    def stream_thinking(self, chunk: str):
        if not chunk:
            return

        self.thinking_buffer += chunk
        print(chunk, end="", flush=True)

        if self.thinking_callback:
            self.thinking_callback(chunk)

# 结束当前思考流输出。
    def end_thinking_stream(self):
        reset = Color.RESET if self.enable_color else ""
        print(reset, flush=True)

        if self.thinking_callback and self.thinking_buffer:
            self.thinking_callback(self.thinking_buffer)

        self.thinking_buffer = ""

# 设置最小日志级别。
    def set_level(self, level: LogLevel):
        self.level = level

# 注册思考流回调函数。
    def set_thinking_callback(self, callback: Callable[[str], None]):
        self.thinking_callback = callback


_default_logger = Logger()


# 返回默认 logger 或创建具名 logger。
def get_logger(name: str = "agent") -> Logger:
    if name == "agent":
        return _default_logger
    return Logger(name)


# 将 debug 调用转发到默认 logger。
def debug(message: str, *args, **kwargs):
    _default_logger.debug(message, *args, **kwargs)


# 将 info 调用转发到默认 logger。
def info(message: str, *args, **kwargs):
    _default_logger.info(message, *args, **kwargs)


# 将 agent 调用转发到默认 logger。
def agent(message: str, *args, **kwargs):
    _default_logger.agent(message, *args, **kwargs)


# 将 warning 调用转发到默认 logger。
def warning(message: str, *args, **kwargs):
    _default_logger.warning(message, *args, **kwargs)


# 将 error 调用转发到默认 logger。
def error(message: str, *args, **kwargs):
    _default_logger.error(message, *args, **kwargs)


# 将 critical 调用转发到默认 logger。
def critical(message: str, *args, **kwargs):
    _default_logger.critical(message, *args, **kwargs)


# 将 thinking 调用转发到默认 logger。
def thinking(message: str, *args, **kwargs):
    _default_logger.thinking(message, *args, **kwargs)


# 启动默认 logger 的思考流。
def start_thinking_stream():
    _default_logger.start_thinking_stream()


# 向默认 logger 转发一段思考流。
def stream_thinking(chunk: str):
    _default_logger.stream_thinking(chunk)


# 结束默认 logger 的思考流。
def end_thinking_stream():
    _default_logger.end_thinking_stream()


# 设置默认 logger 的级别。
def set_log_level(level: LogLevel):
    _default_logger.set_level(level)


# 设置默认 logger 的思考流回调。
def set_thinking_callback(callback: Callable[[str], None]):
    _default_logger.set_thinking_callback(callback)


__all__ = [
    "Logger",
    "LogLevel",
    "Color",
    "get_logger",
    "debug",
    "info",
    "agent",
    "warning",
    "error",
    "critical",
    "thinking",
    "start_thinking_stream",
    "stream_thinking",
    "end_thinking_stream",
    "set_log_level",
    "set_thinking_callback",
]
