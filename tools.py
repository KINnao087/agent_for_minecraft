import os, subprocess

WORKDIR = os.getcwd()

def read_file(path: str, max_bytes: int = 120_000_00):
    p = _safe(path)
    with open(p, "rb") as f:
        data = f.read(max_bytes + 1)
    if len(data) > max_bytes:
        return {"ok": False, "output": f"too large >{max_bytes}"}
    return {"ok": True, "output": data.decode("utf-8", errors="replace")}

def write_file(path: str, content: str):
    p = _safe(path)
    os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        f.write(content)
    return {"ok": True, "output": f"wrote {len(content)} chars to {path}"}

def run_cmd(cmd: str, timeout_sec: int = 60):
    proc = subprocess.run(cmd, shell=True, cwd=WORKDIR,
                          capture_output=True, text=True, timeout=timeout_sec)
    out = (proc.stdout or "") + (proc.stderr or "")
    return {"ok": proc.returncode == 0, "output": out.strip()}

def _safe(query: str, path: str = '.', max_lines: int = 200) -> str:
    """把相对路径转成绝对路径，并阻止路径穿越（例如 ../../windows/system32）。"""
    p = os.path.abspath(os.path.join(WORKDIR, path))
    if not p.startswith(WORKDIR):
        raise ValueError("Path escapes workdir")
    return p


def rg_search(path: str):
    return run_cmd(f"rg {path}")


def list_dir(path: str = "."):
    """列出某个目录下的文件/文件夹（返回按字母排序的列表）。"""
    p = _safe(path)
    names = os.listdir(p)
    names.sort()
    return {"ok": True, "output": "\n".join(names)}

def grep_text(pattern: str, path: str = ".", max_matches: int = 50):
    """
    在 path 目录下递归搜索文本 pattern，返回匹配的文件与行号（最多 max_matches 条）。
    只读文本文件，遇到二进制/编码问题就跳过。
    """
    import re

    root = _safe(path)
    rx = re.compile(pattern)
    hits = []

    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            file_path = os.path.join(dirpath, fn)

            # 简单跳过常见大目录/缓存（你按需加）
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
                # 读不了就跳过
                continue

    return {"ok": True, "output": "\n".join(hits) if hits else "(no matches)"}


TOOLS = {"read_file": read_file, "write_file": write_file, "run_cmd": run_cmd, "list_dir": list_dir}
