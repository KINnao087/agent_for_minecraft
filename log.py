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
    # Initialize logger state and stream buffers.
    def __init__(self, name: str = "agent"):
        self.name = name
        self.level = LogLevel.INFO
        self.enable_color = sys.stdout.isatty()
        self.thinking_callback: Optional[Callable[[str], None]] = None
        self.thinking_buffer = ""

    # Return the ANSI color for a log level.
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

    # Format one log line with time, level, and caller.
    def _format_message(self, level: LogLevel, message: str) -> str:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        color = self._get_color(level)
        reset = Color.RESET if self.enable_color else ""
        location = self._caller_location()
        return f"{color}[{timestamp}] [{level.value}] [{self.name}] [{location}] {message}{reset}"

    # Check whether the current level allows this message.
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

    # Render and print one log message.
    def log(self, level: LogLevel, message: str, *args, **kwargs):
        if not self._should_log(level):
            return

        rendered_message = self._render_message(message, *args, **kwargs)
        formatted_msg = self._format_message(level, rendered_message)
        output = sys.stderr if level in [LogLevel.ERROR, LogLevel.CRITICAL] else sys.stdout
        print(formatted_msg, file=output)

    # Safely format a log message with optional arguments.
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

    # Return the first non-logger caller location.
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

    # Log a debug message.
    def debug(self, message: str, *args, **kwargs):
        self.log(LogLevel.DEBUG, message, *args, **kwargs)

    # Log an info message.
    def info(self, message: str, *args, **kwargs):
        self.log(LogLevel.INFO, message, *args, **kwargs)

    # Log an agent-facing message.
    def agent(self, message: str, *args, **kwargs):
        self.log(LogLevel.AGENT, message, *args, **kwargs)

    # Log a warning message.
    def warning(self, message: str, *args, **kwargs):
        self.log(LogLevel.WARNING, message, *args, **kwargs)

    # Log an error message.
    def error(self, message: str, *args, **kwargs):
        self.log(LogLevel.ERROR, message, *args, **kwargs)

    # Log a critical message.
    def critical(self, message: str, *args, **kwargs):
        self.log(LogLevel.CRITICAL, message, *args, **kwargs)

    # Log a thinking message.
    def thinking(self, message: str, *args, **kwargs):
        self.log(LogLevel.THINKING, message, *args, **kwargs)

    # Start streaming thinking output on one line.
    def start_thinking_stream(self):
        self.thinking_buffer = ""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        color = self._get_color(LogLevel.THINKING)
        prefix = f"{color}[{timestamp}] [{LogLevel.THINKING.value}] [{self.name}] "
        print(prefix, end="", flush=True)

    # Append one thinking chunk to the live stream.
    def stream_thinking(self, chunk: str):
        if not chunk:
            return

        self.thinking_buffer += chunk
        print(chunk, end="", flush=True)

        if self.thinking_callback:
            self.thinking_callback(chunk)

    # Finish the live thinking stream.
    def end_thinking_stream(self):
        reset = Color.RESET if self.enable_color else ""
        print(reset, flush=True)

        if self.thinking_callback and self.thinking_buffer:
            self.thinking_callback(self.thinking_buffer)

        self.thinking_buffer = ""

    # Set the minimum log level.
    def set_level(self, level: LogLevel):
        self.level = level

    # Register a callback for thinking output.
    def set_thinking_callback(self, callback: Callable[[str], None]):
        self.thinking_callback = callback


_default_logger = Logger()


# Return the shared default logger or a new named logger.
def get_logger(name: str = "agent") -> Logger:
    if name == "agent":
        return _default_logger
    return Logger(name)


# Proxy debug logging to the default logger.
def debug(message: str, *args, **kwargs):
    _default_logger.debug(message, *args, **kwargs)


# Proxy info logging to the default logger.
def info(message: str, *args, **kwargs):
    _default_logger.info(message, *args, **kwargs)


# Proxy agent logging to the default logger.
def agent(message: str, *args, **kwargs):
    _default_logger.agent(message, *args, **kwargs)


# Proxy warning logging to the default logger.
def warning(message: str, *args, **kwargs):
    _default_logger.warning(message, *args, **kwargs)


# Proxy error logging to the default logger.
def error(message: str, *args, **kwargs):
    _default_logger.error(message, *args, **kwargs)


# Proxy critical logging to the default logger.
def critical(message: str, *args, **kwargs):
    _default_logger.critical(message, *args, **kwargs)


# Proxy thinking logging to the default logger.
def thinking(message: str, *args, **kwargs):
    _default_logger.thinking(message, *args, **kwargs)


# Start the default logger thinking stream.
def start_thinking_stream():
    _default_logger.start_thinking_stream()


# Forward a thinking chunk to the default logger.
def stream_thinking(chunk: str):
    _default_logger.stream_thinking(chunk)


# Finish the default logger thinking stream.
def end_thinking_stream():
    _default_logger.end_thinking_stream()


# Set the level on the default logger.
def set_log_level(level: LogLevel):
    _default_logger.set_level(level)


# Set the thinking callback on the default logger.
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
