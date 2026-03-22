# Agent for Minecraft (CLI)

A lightweight, tool-enabled **CLI agent** for Minecraft-related development workflows.  
It supports **tool calling**, **multi-turn sessions** (so it doesn't forget after one task), **streaming reasoning output**, and a **token-based sliding context window** to keep prompts under control.

> Built for practical tasks like: scanning project files, searching code, running commands, generating patches, and iterating on a mod/dev repo.

---

## Features

-  **Tool Calling**: the model can call your registered `TOOLS` (file ops, search, commands, etc.)
-  **Session Memory**: keeps conversation context across multiple tasks in the same run
-  **Streaming Reasoning (optional)**: stream the model's thinking output while it works
-  **Token-based Context Trimming**: sliding window based on estimated tokens, while:
  - always keeping `system` messages
  - avoiding breaking `assistant(tool_call) + tool(result)` pairs
-  **Config-driven**: settings live in `config.json` and are loaded via `config.py`

---

## Requirements

- Python **3.10+** (3.11 recommended)
- A configured provider: `api` or `web`

Python dependencies (adjust to your repo):
- `openai` (OpenAI SDK, used with DeepSeek `base_url`)
- any local modules in this repo (`tools`, `log`, `config`, etc.)

Provider notes:
- `provider: "api"` uses the DeepSeek API and requires `DEEPSEEK_API_KEY` or `config.api_key`
- `provider: "web"` uses the browser-backed web adapter in `core/web_to_api`

---

## Quick Start

### 1) Clone
```bash
git clone <YOUR_REPO_URL>
cd agent_for_minecraft
```

### 2) Set environment variable
```bash
set DEEPSEEK_API_KEY=your_api_key
```

### 3) Run CLI mode
```bash
python agent.py
```

### 4) Run socket API mode
```bash
python socket_server.py --host 127.0.0.1 --port 5050
```

---

## Socket API

This project also provides a TCP socket API for external software.

Protocol rules:
- TCP long connection
- UTF-8 encoding
- JSON Lines protocol: one JSON request per line, one JSON response per line

Default address:
- Host: `127.0.0.1`
- Port: `5050`

### Request format

```json
{
  "action": "chat",
  "session_id": "demo",
  "task": "Analyze the current project entry point",
  "max_steps": 12,
  "stream": false
}
```

Fields:
- `action`: request type, currently supports `chat`, `reset`, and `beat`
- `session_id`: conversation id, used to keep multi-turn context for one external client
- `task`: user input passed to the agent
- `max_steps`: optional, max internal reasoning/tool-call steps
- `stream`: optional, whether to enable internal thinking stream

### Chat response

```json
{
  "ok": true,
  "action": "chat",
  "session_id": "demo",
  "reply": "This is the agent response"
}
```

### Reset session

Request:
```json
{
  "action": "reset",
  "session_id": "demo"
}
```

### Heartbeat

Request:
```json
{
  "action": "beat",
  "session_id": "demo"
}
```

Response:
```json
{
  "ok": true,
  "action": "beat",
  "session_id": "demo"
}
```

Response:
```json
{
  "ok": true,
  "action": "reset",
  "session_id": "demo"
}
```

### Python client example

```python
import json
import socket

request = {
    "action": "chat",
    "session_id": "demo",
    "task": "List the main modules in the current project",
    "max_steps": 10,
    "stream": False,
}

with socket.create_connection(("127.0.0.1", 5050)) as sock:
    sock.sendall((json.dumps(request, ensure_ascii=False) + "\n").encode("utf-8"))
    response = sock.recv(65535).decode("utf-8").strip()
    print(json.loads(response))
```

### Error response example

```json
{
  "ok": false,
  "error": "task must be a non-empty string"
}
```

---

## Notes

- `session_id` should be unique per external client or conversation
- If you do not need multi-turn memory, always send a new `session_id`
- If you need to clear context, call `reset`
- Current API is synchronous: one request returns one final response
