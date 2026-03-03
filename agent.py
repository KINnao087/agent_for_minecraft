import os, json, requests, re

from tools import TOOLS

OLLAMA = os.getenv("OLLAMA", "http://127.0.0.1:11434")
MODEL = os.getenv("OLLAMA_MODEL", "huihui_ai/qwen3-vl-abliterated:8b")

SYSTEM = """你是一个代码Agent。优先用工具获得事实（读文件/跑命令）。
规则：
- 不要凭空猜文件内容；不确定就read_file。
- 修改文件前先read_file，修改后用write_file写回完整文件。
- 每次关键修改后都run_cmd验证。
- 少废话，输出可执行的修改。

你是一个严格的工具调用Agent。

你只能输出两种东西之一：
(1) 工具调用：必须使用 <tool_call>{...}</tool_call>，JSON里只能包含 name 和 arguments 两个字段。
(2) 最终答案：必须以 <final>...</final> 包裹。

硬规则：
- 需要任何文件内容/目录/命令输出时，必须先调用工具，禁止猜测。
- 每次最多调用一个工具；拿到结果后再决定下一步。
- 工具调用 JSON 示例：
<tool_call>{"name":"read_file","arguments":{"path":"foo.py","max_bytes":120000}}</tool_call>

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


def run_agent(task: str, max_steps: int = 8):
    """Agent 主循环：模型要工具→你执行→回喂→继续。"""
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": task},
    ]

    for step in range(1, max_steps + 1):
        msg = ollama_chat(messages)
        content = (msg.get("content") or "").strip()

        # 1) 标准 tool_calls 格式
        tool_calls = msg.get("tool_calls") or []
        if tool_calls:
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

            messages.append({
                "role": "tool",
                "name": name,
                "content": json.dumps(result, ensure_ascii=False),
            })
            continue

        # 2) 普通输出
        messages.append({"role": "assistant", "content": content})
        print(f"\n[Step {step}] {content if content else '(empty)'}\n")

        # 简单停机：模型说“完成/结束”
        if "完成" in content or "结束" in content:
            break

if __name__ == "__main__":
    # Windows: 用 dir；Linux/mac: 用 ls
    run_agent("先用 run_cmd 运行 `dir`，告诉我当前目录有哪些文件。")