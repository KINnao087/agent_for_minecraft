import os
import subprocess
from typing import Any, Callable, Dict

WORKDIR = os.getcwd()

def _err(tool: str, e: Exception, **extra) -> Dict[str, Any]:
    # 统一错误返回结构：不抛异常，返回给 agent
    msg = f"{type(e).__name__}: {e}"
    if extra:
        msg += " | " + ", ".join(f"{k}={v!r}" for k, v in extra.items())
    return {"ok": False, "output": f"[{tool}] {msg}"}

def _sanitize_text(s: Any) -> str:
    # 防止 surrogate/非法字符导致 UnicodeEncodeError
    if s is None:
        return ""
    if not isinstance(s, str):
        s = str(s)
    return s.encode("utf-8", "replace").decode("utf-8")

def _safe(path: str, base: str = WORKDIR) -> str:
    """把相对路径转成绝对路径，并阻止路径穿越（例如 ../../windows/system32）。"""
    try:
        # 注意：你的原版 _safe 形参写错了（query/path），这里修成真正的 path
        p = os.path.abspath(os.path.join(base, path))
        if not p.startswith(base):
            raise ValueError("Path escapes workdir")
        return p
    except Exception as e:
        # 让调用者拿到统一错误（调用者会 catch）
        raise

def read_file(path: str, max_bytes: int = 120_000):
    try:
        p = _safe(path)
        with open(p, "rb") as f:
            data = f.read(max_bytes + 1)
        if len(data) > max_bytes:
            return {"ok": False, "output": f"too large >{max_bytes}"}
        text = data.decode("utf-8", errors="replace")
        text = _sanitize_text(text)
        return {"ok": True, "output": text}
    except Exception as e:
        return _err("read_file", e, path=path, max_bytes=max_bytes)

def write_file(path: str, content: str):
    try:
        p = _safe(path)
        os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
        content = _sanitize_text(content)
        with open(p, "w", encoding="utf-8", errors="replace") as f:
            f.write(content)
        return {"ok": True, "output": f"wrote {len(content)} chars to {path}"}
    except Exception as e:
        return _err("write_file", e, path=path)

def run_cmd(cmd: str, timeout_sec: int = 60):
    try:
        proc = subprocess.run(
            cmd,
            shell=True,
            cwd=WORKDIR,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
        out = (proc.stdout or "") + (proc.stderr or "")
        out = _sanitize_text(out.strip())
        return {"ok": proc.returncode == 0, "output": out}
    except subprocess.TimeoutExpired as e:
        return {"ok": False, "output": f"[run_cmd] TimeoutExpired after {timeout_sec}s: {e.cmd!r}"}
    except Exception as e:
        return _err("run_cmd", e, cmd=cmd, timeout_sec=timeout_sec)

def rg_search(query: str, path: str = ".", max_lines: int = 200):
    try:
        # 先确保 path 合法（防穿越）
        _ = _safe(path)
        cmd = f'rg -n --no-heading --color never "{query}" "{path}"'
        r = run_cmd(cmd)
        if not r.get("ok"):
            return r

        out = r.get("output", "")
        lines = out.splitlines()
        if len(lines) > max_lines:
            out = "\n".join(lines[:max_lines]) + f"\n... (truncated, {len(lines)} lines total)"
        return {"ok": True, "output": _sanitize_text(out)}
    except Exception as e:
        return _err("rg_search", e, query=query, path=path, max_lines=max_lines)

def list_dir(path: str = "."):
    """列出某个目录下的文件/文件夹（返回按字母排序的列表）。"""
    try:
        p = _safe(path)
        names = os.listdir(p)
        names.sort()
        return {"ok": True, "output": "\n".join(names)}
    except Exception as e:
        return _err("list_dir", e, path=path)

def grep_text(pattern: str, path: str = ".", max_matches: int = 50):
    """
    在 path 目录下递归搜索文本 pattern，返回匹配的文件与行号（最多 max_matches 条）。
    """
    try:
        import re

        root = _safe(path)
        rx = re.compile(pattern)
        hits = []

        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                file_path = os.path.join(dirpath, fn)

                if ".git" in file_path or "__pycache__" in file_path or "node_modules" in file_path:
                    continue

                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        for i, line in enumerate(f, 1):
                            if rx.search(line):
                                rel = os.path.relpath(file_path, WORKDIR)
                                hits.append(f"{rel}:{i}: {line.rstrip()}")
                                if len(hits) >= max_matches:
                                    return {"ok": True, "output": "\n".join(hits)}
                except Exception:
                    # 单文件读不了就跳过
                    continue

        return {"ok": True, "output": "\n".join(hits) if hits else "(no matches)"}
    except Exception as e:
        return _err("grep_text", e, pattern=pattern, path=path, max_matches=max_matches)

TOOLS = {
    "read_file": read_file,
    "write_file": write_file,
    "run_cmd": run_cmd,
    "list_dir": list_dir,
    "rg_search": rg_search,
    "grep_text": grep_text,
}