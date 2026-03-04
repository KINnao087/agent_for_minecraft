import os, json, re, time
from tools import TOOLS
from openai import OpenAI

# ===== DeepSeek(OpenAI-compatible) client =====
client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)

MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

SYSTEM = """你是代码工具Agent。

只允许两种输出：(最高优先级)
1) <tool_call>{"name":"...","arguments":{...}}</tool_call>  (这个用来调用工具)
2) <final>...</final> (这个用来回复用户)

规则：
- 你需要对用户的需求加以分析，然后决定是否调用工具还是直接回复用户。并不是所有需求都需要依据代码
- 只要需要文件/目录/命令结果：立刻输出 tool_call，禁止解释/猜测。
- 不确定内容先 read_file；修改后 write_file 写回完整文件；关键修改后 run_cmd 验证。
- 每轮最多调用一个工具；拿到结果再继续。
"""

# 工具定义（OpenAI tools schema，DeepSeek 兼容）
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
                    "max_bytes": {"type": "integer", "default": 20000},  # 建议先别喂太大
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write full content to a text file in the project",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                "required": ["path", "content"],
            },
        },
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
                    "timeout_sec": {"type": "integer", "default": 60},
                },
                "required": ["cmd"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List directory entries under a path",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "default": "."}},
                "required": [],
            },
        },
    },
]

CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)
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


def trim_messages(messages, keep_last=20):
    # 永远保留第一条 system，其余保留最近 keep_last 条
    sys_msg = None
    rest = []
    for m in messages:
        if m.get("role") == "system" and sys_msg is None:
            sys_msg = m
        else:
            rest.append(m)
    return ([sys_msg] if sys_msg else []) + rest[-keep_last:]


def deepseek_chat(messages):
    """
    返回 assistant message（兼容 tool_calls / content）
    """
    resp = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=TOOL_DEFS,
        tool_choice="auto",
        temperature=0.2,
        stream=False,
    )
    return resp.choices[0].message  # .content / .tool_calls


def run_agent(task: str, max_steps: int = 8):
    messages = [
        {"role": "system", "content": SYSTEM + f"\n\n当前工作目录是：{os.getcwd()}"},
        {"role": "user", "content": task},
    ]

    t0 = time.perf_counter()

    for step in range(1, max_steps + 1):
        msg = deepseek_chat(messages)

        # ===== 1) SDK tool_calls =====
        tool_calls = getattr(msg, "tool_calls", None) or []
        if tool_calls:
            # 记录 assistant 的 tool_call 意图
            messages.append(
                {
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in tool_calls
                    ],
                }
            )

            tc = tool_calls[0]  # 你要求每轮最多一个
            name = tc.function.name
            args = tc.function.arguments or "{}"
            if isinstance(args, str):
                args = json.loads(args)

            if name not in TOOLS:
                result = {"ok": False, "output": f"unknown tool: {name}"}
            else:
                result = TOOLS[name](**args)

            # 你本地看日志可以，但别指望模型看 print
            print(f"[tool] {name} ok={result.get('ok')} output: {result.get('output', '')}")

            # ===== 关键：把工具结果喂回模型 =====
            # 更稳：直接喂 output（不要再包一层 JSON，尤其 output 很长时）
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,  # 很重要：对应到这次 tool call
                    "name": name,
                    "content": result.get("output", ""),
                }
            )

            messages = trim_messages(messages, keep_last=30)
            continue

        # ===== 2) 普通输出 =====
        content = (msg.content or "").strip()
        messages.append({"role": "assistant", "content": content})

        dt = time.perf_counter() - t0
        print(f"[Step {step}] {content if content else '(empty)'}  tot time: {dt:.3f}s\n")

        # 看到 <final> 就停
        if parse_final(content) is not None or ("<final>" in content and "</final>" in content):
            break

        # 如果模型没用 tool_calls 机制而是走你自定义 <tool_call> 标签，也能兜底
        maybe = parse_tool_call(content)
        if maybe:
            name, args = maybe
            if name not in TOOLS:
                result = {"ok": False, "output": f"unknown tool: {name}"}
            else:
                result = TOOLS[name](**args)

            print(f"[tool(tag)] {name} ok={result.get('ok')}")
            messages.append({"role": "tool", "name": name, "content": result.get("output", "")})
            messages = trim_messages(messages, keep_last=30)
            continue

# ....
if __name__ == "__main__":
    while True:
        query = str(input("> ").strip())
        if not query:
            continue
        run_agent(query)