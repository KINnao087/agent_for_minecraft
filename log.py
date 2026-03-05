#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
日志模块，提供带颜色的日志输出和AI思考过程流式传输功能
"""

import sys
import time
from typing import Optional, Callable, Any
from enum import Enum


class LogLevel(Enum):
    """日志级别枚举"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
    THINKING = "THINKING"  # AI思考过程


class Color:
    """ANSI颜色代码"""
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
    
    # 背景色
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"
    BG_MAGENTA = "\033[45m"
    BG_CYAN = "\033[46m"
    BG_WHITE = "\033[47m"


class Logger:
    """日志记录器"""
    
    def __init__(self, name: str = "agent"):
        self.name = name
        self.level = LogLevel.INFO
        self.enable_color = sys.stdout.isatty()
        self.thinking_callback: Optional[Callable[[str], None]] = None
        self.thinking_buffer = ""
        
    def _get_color(self, level: LogLevel) -> str:
        """根据日志级别获取颜色"""
        if not self.enable_color:
            return ""
            
        color_map = {
            LogLevel.DEBUG: Color.BRIGHT_BLACK,
            LogLevel.INFO: Color.BRIGHT_BLUE,
            LogLevel.WARNING: Color.BRIGHT_YELLOW,
            LogLevel.ERROR: Color.BRIGHT_RED,
            LogLevel.CRITICAL: Color.RED + Color.BG_WHITE,
            LogLevel.THINKING: Color.BRIGHT_MAGENTA,
        }
        return color_map.get(level, Color.RESET)
    
    def _format_message(self, level: LogLevel, message: str) -> str:
        """格式化日志消息"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        color = self._get_color(level)
        reset = Color.RESET if self.enable_color else ""
        
        return f"{color}[{timestamp}] [{level.value}] [{self.name}] {message}{reset}"
    
    def log(self, level: LogLevel, message: str, *args, **kwargs):
        """通用日志方法"""
        if not self._should_log(level):
            return
            
        formatted_msg = self._format_message(level, message.format(*args, **kwargs))
        print(formatted_msg, file=sys.stderr if level in [LogLevel.ERROR, LogLevel.CRITICAL] else sys.stdout)
    
    def _should_log(self, level: LogLevel) -> bool:
        """检查是否应该记录该级别的日志"""
        level_order = {
            LogLevel.DEBUG: 0,
            LogLevel.INFO: 1,
            LogLevel.WARNING: 2,
            LogLevel.ERROR: 3,
            LogLevel.CRITICAL: 4,
            LogLevel.THINKING: 5,
        }
        return level_order.get(level, 0) >= level_order.get(self.level, 0)
    
    def debug(self, message: str, *args, **kwargs):
        """调试日志"""
        self.log(LogLevel.DEBUG, message, *args, **kwargs)
    
    def info(self, message: str, *args, **kwargs):
        """信息日志"""
        self.log(LogLevel.INFO, message, *args, **kwargs)
    
    def warning(self, message: str, *args, **kwargs):
        """警告日志"""
        self.log(LogLevel.WARNING, message, *args, **kwargs)
    
    def error(self, message: str, *args, **kwargs):
        """错误日志"""
        self.log(LogLevel.ERROR, message, *args, **kwargs)
    
    def critical(self, message: str, *args, **kwargs):
        """严重错误日志"""
        self.log(LogLevel.CRITICAL, message, *args, **kwargs)
    
    def thinking(self, message: str, *args, **kwargs):
        """AI思考过程日志"""
        self.log(LogLevel.THINKING, message, *args, **kwargs)
    
    def start_thinking_stream(self):
        """开始流式思考过程"""
        self.thinking_buffer = ""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        color = self._get_color(LogLevel.THINKING)
        reset = Color.RESET if self.enable_color else ""
        
        prefix = f"{color}[{timestamp}] [{LogLevel.THINKING.value}] [{self.name}] "
        print(prefix, end="", flush=True)
    
    def stream_thinking(self, chunk: str):
        """流式输出思考过程"""
        if not chunk:
            return
            
        self.thinking_buffer += chunk
        # 输出到控制台
        print(chunk, end="", flush=True)
        
        # 如果有回调函数，调用它
        if self.thinking_callback:
            self.thinking_callback(chunk)
    
    def end_thinking_stream(self):
        """结束流式思考过程"""
        reset = Color.RESET if self.enable_color else ""
        print(reset, flush=True)
        
        # 如果有回调函数，通知思考结束
        if self.thinking_callback and self.thinking_buffer:
            self.thinking_callback(self.thinking_buffer)
        
        self.thinking_buffer = ""
    
    def set_level(self, level: LogLevel):
        """设置日志级别"""
        self.level = level
    
    def set_thinking_callback(self, callback: Callable[[str], None]):
        """设置思考过程回调函数"""
        self.thinking_callback = callback


# 全局日志实例
_default_logger = Logger()


def get_logger(name: str = "agent") -> Logger:
    """获取日志记录器"""
    if name == "agent":
        return _default_logger
    return Logger(name)


def debug(message: str, *args, **kwargs):
    """调试日志（全局函数）"""
    _default_logger.debug(message, *args, **kwargs)


def info(message: str, *args, **kwargs):
    """信息日志（全局函数）"""
    _default_logger.info(message, *args, **kwargs)


def warning(message: str, *args, **kwargs):
    """警告日志（全局函数）"""
    _default_logger.warning(message, *args, **kwargs)


def error(message: str, *args, **kwargs):
    """错误日志（全局函数）"""
    _default_logger.error(message, *args, **kwargs)


def critical(message: str, *args, **kwargs):
    """严重错误日志（全局函数）"""
    _default_logger.critical(message, *args, **kwargs)


def thinking(message: str, *args, **kwargs):
    """AI思考过程日志（全局函数）"""
    _default_logger.thinking(message, *args, **kwargs)


def start_thinking_stream():
    """开始流式思考过程（全局函数）"""
    _default_logger.start_thinking_stream()


def stream_thinking(chunk: str):
    """流式输出思考过程（全局函数）"""
    _default_logger.stream_thinking(chunk)


def end_thinking_stream():
    """结束流式思考过程（全局函数）"""
    _default_logger.end_thinking_stream()


def set_log_level(level: LogLevel):
    """设置日志级别（全局函数）"""
    _default_logger.set_level(level)


def set_thinking_callback(callback: Callable[[str], None]):
    """设置思考过程回调函数（全局函数）"""
    _default_logger.set_thinking_callback(callback)


# 导出常用函数
__all__ = [
    'Logger',
    'LogLevel',
    'Color',
    'get_logger',
    'debug',
    'info',
    'warning',
    'error',
    'critical',
    'thinking',
    'start_thinking_stream',
    'stream_thinking',
    'end_thinking_stream',
    'set_log_level',
    'set_thinking_callback',
]