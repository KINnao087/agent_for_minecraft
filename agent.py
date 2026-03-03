import os, json, requests, re

from tools import TOOLS

OLLAMA = os.getenv("OLLAMA", "http://127.0.0.1:11434")
MODEL = os.getenv("OLLAMA_MODEL", "huihui_ai/qwen3-vl-abliterated:8b")

SYSTEM = """你是代码工具Agent。

只允许两种输出：
1) <tool_call>{"name":"...","arguments":{...}}</tool_call>
2) <final>...</final>

规则：
- 只要需要文件/目录/命令结果：立刻输出 tool_call，禁止解释/猜测。
- 不确定内容先 read_file；修改后 write_file 写回完整文件；关键修改后 run_cmd 验证。
- 每轮最多调用一个工具；拿到结果再继续。
"""

# 工具“说明书”：发给模型看，让它知道有哪些工具、每个工具要什么参数
TOOL_DEFS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a text file from the project",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "max_bytes": {"type": "integer", "default": 120000}
                },
                "required": ["path"]
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write full content to a text file in the project",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"}
                },
                "required": ["path", "content"]
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_cmd",
            "description": "Run a shell command in the project directory",
            "parameters": {
                "type": "object",
                "properties": {
                    "cmd": {"type": "string"},
                    "timeout_sec": {"type": "integer", "default": 60}
                },
                "required": ["cmd"]
            },
        }
    },
{
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List directory entries under a path",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "default": "."}},
                "required": []
            },
        }
    },
]

def ollama_chat(messages):
    """调用 Ollama 的 /api/chat 接口，返回模型的 message（可能包含 tool_calls 或 content）。"""
    url = f"{OLLAMA}/api/chat"
    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0.2},
        "tools": TOOL_DEFS,
    }
    r = requests.post(url, json=payload, timeout=120)
    r.raise_for_status()
    return r.json()["message"]  # {"role": "...", "content": "...", "tool_calls": [...]}

CALL_RE  = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)
FINAL_RE = re.compile(r"<final>\s*(.*?)\s*</final>", re.S)

def parse_tool_call(content: str):
    m = CALL_RE.search(content or "")
    if not m:
        return None
    obj = json.loads(m.group(1))
    return obj["name"], obj.get("arguments", {})

def parse_final(content: str):
    m = FINAL_RE.search(content or "")
    return m.group(1) if m else None

# import xml.etree.ElementTree as ET
# def print_xml(xml_msg: str):
#     final = ET.fromstring(xml_msg)
#     print(final.text.strip())

import time
def run_agent(task: str, max_steps: int = 8):
    """Agent 主循环：模型要工具→你执行→回喂→继续。"""
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": task},
    ]
    t0 = time.perf_counter()

    for step in range(1, max_steps + 1):

        msg = ollama_chat(messages)

        content = (msg.get("content") or "").strip()

        # print(msg)

        # 1) 标准 tool_calls 格式
        tool_calls = msg.get("tool_calls") or []
        if tool_calls:
            print(f"tool_calls: {msg.get('tool_calls')}")

            messages.append(msg)  # 把“我要调用工具”的意图也记进历史

            tc = tool_calls[0]  # 我们要求每次最多一个工具
            name = tc["function"]["name"]
            args = tc["function"].get("arguments") or {}
            if isinstance(args, str):
                args = json.loads(args)

            if name not in TOOLS:
                result = {"ok": False, "output": f"unknown tool: {name}"}
            else:
                result = TOOLS[name](**args)
            print(f"tool: {name} result: {result.get('ok')}")
            messages.append({
                "role": "tool",
                "name": name,
                "content": json.dumps(result, ensure_ascii=False),
            })
            continue

        # 2) 普通输出
        dt = time.perf_counter() - t0
        messages.append({"role": "assistant", "content": content})
        print(f"[Step {step}] {content if content else '(empty)'}  tot time: {dt:.3f}s\n")

        # 简单停机：模型说“完成/结束”
        if "<final>" in content or "</final>" in content:
            break

if __name__ == "__main__":
    # Windows: 用 dir；Linux/mac: 用 ls
    run_agent("调用 list_dir 列出当前目录。然后用一句话告诉我有哪些文件。")