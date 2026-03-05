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
- A **DeepSeek API Key** in environment variables

Python dependencies (adjust to your repo):
- `openai` (OpenAI SDK, used with DeepSeek `base_url`)
- any local modules in this repo (`tools`, `log`, `config`, etc.)

---

## Quick Start

### 1) Clone
```bash
git clone <YOUR_REPO_URL>
cd agent_for_minecraft
