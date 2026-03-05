#!/usr/bin/env python3
"""检查agent.py中的TOOL_DEFS是否包含新工具"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import agent

# 检查TOOL_DEFS中的工具名称
tool_names = []
for tool_def in agent.TOOL_DEFS:
    name = tool_def["function"]["name"]
    tool_names.append(name)

print("agent.py中的TOOL_DEFS包含以下工具:")
for name in sorted(tool_names):
    print(f"  - {name}")

# 检查是否包含新添加的工具
new_tools = ["copy_file", "move_file", "delete_file", "mkdir_p"]
missing_tools = []

for tool in new_tools:
    if tool in tool_names:
        print(f"✓ {tool} 已包含在TOOL_DEFS中")
    else:
        print(f"✗ {tool} 未包含在TOOL_DEFS中")
        missing_tools.append(tool)

if missing_tools:
    print(f"\n错误: 以下工具未包含在TOOL_DEFS中: {', '.join(missing_tools)}")
    sys.exit(1)
else:
    print("\n所有新工具都已成功添加到TOOL_DEFS中！")
    sys.exit(0)