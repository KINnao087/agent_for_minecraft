#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
示例：如何使用新的日志系统
"""

import sys
import time
from log import *

def example_agent_workflow():
    """模拟Agent工作流程"""
    logger = get_logger("example_agent")
    
    # 1. 任务开始
    logger.info("开始处理用户请求: 创建一个简单的Python脚本")
    
    # 2. AI思考过程（流式）
    logger.thinking("用户想要创建一个简单的Python脚本...")
    time.sleep(0.5)
    logger.thinking("我需要考虑脚本的功能和结构...")
    time.sleep(0.5)
    
    # 3. 工具调用
    logger.info("调用工具: list_dir")
    logger.info("调用工具: write_file")
    
    # 4. 处理结果
    logger.info("工具执行成功")
    logger.warning("注意：文件已存在，需要确认是否覆盖")
    
    # 5. 最终回复
    logger.info("任务完成，准备回复用户")
    
    # 6. 模拟流式思考输出
    print("\n" + "=" * 60)
    print("模拟AI流式思考过程:")
    print("=" * 60)
    
    start_thinking_stream()
    thinking_chunks = [
        "让我思考一下这个问题...\n",
        "用户想要一个简单的Python脚本，我需要考虑几个方面：\n",
        "1. 脚本的功能是什么？\n",
        "2. 需要哪些输入和输出？\n",
        "3. 如何处理错误情况？\n",
        "4. 如何让脚本更健壮？\n",
        "基于这些考虑，我将创建一个简单的文件处理脚本。\n"
    ]
    
    for chunk in thinking_chunks:
        stream_thinking(chunk)
        time.sleep(0.3)
    
    end_thinking_stream()

def example_with_custom_logger():
    """使用自定义日志记录器"""
    print("\n" + "=" * 60)
    print("使用自定义日志记录器:")
    print("=" * 60)
    
    # 创建自定义日志记录器
    custom_logger = Logger("my_module")
    custom_logger.set_level(LogLevel.DEBUG)
    
    # 记录不同级别的日志
    custom_logger.debug("详细的调试信息")
    custom_logger.info("模块初始化完成")
    custom_logger.warning("配置文件中缺少可选参数")
    custom_logger.error("无法连接到数据库")
    custom_logger.critical("系统资源耗尽，需要立即处理")

def example_color_demo():
    """颜色演示"""
    print("\n" + "=" * 60)
    print("日志颜色演示:")
    print("=" * 60)
    
    # 使用全局函数
    info("信息级别 - 蓝色")
    warning("警告级别 - 黄色")
    error("错误级别 - 红色")
    critical("严重级别 - 红底白字")
    thinking("思考级别 - 紫色")

def main():
    """主函数"""
    print("日志系统使用示例")
    print("=" * 60)
    
    # 设置日志级别为DEBUG以显示所有日志
    set_log_level(LogLevel.DEBUG)
    
    # 运行示例
    example_agent_workflow()
    example_with_custom_logger()
    example_color_demo()
    
    print("\n" + "=" * 60)
    print("示例完成！")
    print("=" * 60)
    
    # 最后显示一些提示
    print("\n使用提示:")
    print("1. 导入: from log import get_logger, info, warning, error, etc.")
    print("2. 获取日志记录器: logger = get_logger('module_name')")
    print("3. 设置日志级别: logger.set_level(LogLevel.DEBUG)")
    print("4. 流式思考: start_thinking_stream(), stream_thinking(), end_thinking_stream()")
    print("5. 全局函数: info('message'), error('message'), etc.")

if __name__ == "__main__":
    main()