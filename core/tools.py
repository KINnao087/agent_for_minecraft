import os, platform
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict

from log import get_logger

WORKDIR = os.getcwd()
logger = get_logger("tools")

def _err(tool: str, e: Exception, **extra) -> Dict[str, Any]:
    # 统一错误返回结构：不抛异常，返回给 agent
    msg = f"{type(e).__name__}: {e}"
    if extra:
        msg += " | " + ", ".join(f"{k}={v!r}" for k, v in extra.items())
    
    # 记录错误日志
    logger.error("工具 {} 执行出错: {}", tool, msg)
    
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
        # 记录安全错误
        logger.error("路径安全检查失败: {}, base={}", str(e), base)
        raise

def read_file(path: str, max_bytes: int = 120_000):
    try:
        p = _safe(path)
        logger.debug("读取文件: {}, 最大字节数: {}", path, max_bytes)
        
        with open(p, "rb") as f:
            data = f.read(max_bytes + 1)
        
        if len(data) > max_bytes:
            logger.warning("文件过大: {} > {}", len(data), max_bytes)
            return {"ok": False, "output": f"too large >{max_bytes}"}
        
        text = data.decode("utf-8", errors="replace")
        text = _sanitize_text(text)
        
        logger.info("成功读取文件: {}, 大小: {} 字节", path, len(data))
        return {"ok": True, "output": text}
    except Exception as e:
        return _err("read_file", e, path=path, max_bytes=max_bytes)

def write_file(path: str, content: str):
    try:
        p = _safe(path)
        logger.debug("写入文件: {}, 内容长度: {}", path, len(content))
        
        os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
        content = _sanitize_text(content)
        
        with open(p, "w", encoding="utf-8", errors="replace") as f:
            f.write(content)
        
        logger.info("成功写入文件: {}, 大小: {} 字符", path, len(content))
        return {"ok": True, "output": f"wrote {len(content)} chars to {path}"}
    except Exception as e:
        return _err("write_file", e, path=path)

def run_cmd(cmd: str, timeout_sec: int = 60):
    try:
        logger.debug("执行命令: {}, 超时: {} 秒", cmd, timeout_sec)
        
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
        
        if proc.returncode == 0:
            logger.info("命令执行成功: {}, 返回码: {}", cmd, proc.returncode)
        else:
            logger.warning("命令执行失败: {}, 返回码: {}", cmd, proc.returncode)
        
        return {"ok": proc.returncode == 0, "output": out}
    except subprocess.TimeoutExpired as e:
        logger.error("命令执行超时: {}, 超时时间: {} 秒", e.cmd, timeout_sec)
        return {"ok": False, "output": f"[run_cmd] TimeoutExpired after {timeout_sec}s: {e.cmd!r}"}
    except Exception as e:
        return _err("run_cmd", e, cmd=cmd, timeout_sec=timeout_sec)

def rg_search(query: str, path: str = ".", max_lines: int = 200):
    try:
        # 先确保 path 合法（防穿越）
        _ = _safe(path)
        logger.debug("搜索文本: {}, 路径: {}, 最大行数: {}", query, path, max_lines)
        
        cmd = f'rg -n --no-heading --color never "{query}" "{path}"'
        r = run_cmd(cmd)
        
        if not r.get("ok"):
            logger.warning("rg搜索失败: {}", query)
            return r

        out = r.get("output", "")
        lines = out.splitlines()
        
        if len(lines) > max_lines:
            logger.info("rg搜索结果过多: {} 行，截断至 {} 行", len(lines), max_lines)
            out = "\n".join(lines[:max_lines]) + f"\n... (truncated, {len(lines)} lines total)"
        
        logger.info("rg搜索完成: {}, 找到 {} 行结果", query, len(lines))
        return {"ok": True, "output": _sanitize_text(out)}
    except Exception as e:
        return _err("rg_search", e, query=query, path=path, max_lines=max_lines)

def read_file_lines(path: str, start_line: int = 0, max_lines: int = 200):
    try:
        p = _safe(path)
        logger.debug("读取文件行: {}, 起始行: {}, 最大行数: {}", path, start_line, max_lines)
        
        # 读取文件内容
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        
        # 处理起始行索引（支持从0开始计数）
        if start_line < 0:
            start_line = max(0, len(lines) + start_line)  # 支持负数索引，从末尾开始
        
        # 确保起始行不超过文件行数
        if start_line >= len(lines):
            logger.warning("起始行超出文件范围: {} >= {}", start_line, len(lines))
            return {"ok": True, "output": ""}
        
        # 计算结束行
        end_line = min(start_line + max_lines, len(lines))
        
        # 提取指定行范围
        selected_lines = lines[start_line:end_line]
        
        # 构建输出，包含行号信息
        output_lines = []
        for i, line in enumerate(selected_lines, start=start_line + 1):
            output_lines.append(f"{i}: {line.rstrip()}")
        
        output = "\n".join(output_lines)
        output = _sanitize_text(output)
        
        logger.info("成功读取文件行: {}, 行范围: {}-{}, 共 {} 行", 
                   path, start_line + 1, end_line, len(selected_lines))
        
        return {"ok": True, "output": output}
    except Exception as e:
        return _err("read_file_lines", e, path=path, start_line=start_line, max_lines=max_lines)

def copy_file(src: str, dst: str, overwrite: bool = False):
    src_p = Path(src)
    dst_p = Path(dst)

    if not src_p.exists():
        return {"ok": False, "error": f"src not found: {src}"}

    if (dst_p.exists() and dst_p.is_dir()):
        dst_p = dst_p / src_p.name

    # 处理 overwrite
    if dst_p.exists():
        if not overwrite:
            return {"ok": False, "error": f"dst already exists: {str(dst_p)}"}
        # overwrite=True: 先删除目标
        if dst_p.is_dir():
            shutil.rmtree(dst_p)
        else:
            dst_p.unlink()

    try:
        if src_p.is_dir():
            # Python 3.8+：dirs_exist_ok 允许覆盖，但我们前面已经清理了
            shutil.copytree(src_p, dst_p)
        else:
            # copy2 会尽量保留时间戳等元数据
            dst_p.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_p, dst_p)

        return {"ok": True, "dst": str(dst_p)}
    except Exception as e:
        return {"ok": False, "error": repr(e)}

def move_file(src: str, dst: str, overwrite: bool = False):
    src_p = Path(src)
    dst_p = Path(dst)

    if not src_p.exists():
        return {"ok": False, "error": f"src not found: {src}"}

    if (dst_p.exists() and dst_p.is_dir()):
        dst_p = dst_p / src_p.name

    # 处理 overwrite
    if dst_p.exists():
        if not overwrite:
            return {"ok": False, "error": f"dst already exists: {str(dst_p)}"}
        # overwrite=True: 先删除目标
        if dst_p.is_dir():
            shutil.rmtree(dst_p)
        else:
            dst_p.unlink()

    try:
        # 移动文件或目录
        if src_p.is_dir():
            # 对于目录，使用shutil.move
            shutil.move(str(src_p), str(dst_p))
        else:
            # 对于文件，确保目标目录存在
            dst_p.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src_p), str(dst_p))

        return {"ok": True, "dst": str(dst_p)}
    except Exception as e:
        return {"ok": False, "error": repr(e)}

def delete_file(path: str):
    """删除文件或目录"""
    try:
        path_p = Path(path)
        
        if not path_p.exists():
            return {"ok": False, "error": f"path not found: {path}"}
        
        logger.debug("删除文件/目录: {}", path)
        
        if path_p.is_dir():
            shutil.rmtree(path_p)
            logger.info("成功删除目录: {}", path)
        else:
            path_p.unlink()
            logger.info("成功删除文件: {}", path)
        
        return {"ok": True, "output": f"deleted {path}"}
    except Exception as e:
        return {"ok": False, "error": repr(e)}

def mkdir_p(path: str):
    """创建目录，如果父目录不存在也会创建（类似 mkdir -p）"""
    try:
        path_p = Path(path)
        
        logger.debug("创建目录: {}", path)
        
        # 使用mkdir创建目录，parents=True表示创建父目录，exist_ok=True表示如果目录已存在也不报错
        path_p.mkdir(parents=True, exist_ok=True)
        
        logger.info("成功创建目录: {}", path)
        return {"ok": True, "output": f"created directory {path}"}
    except Exception as e:
        return {"ok": False, "error": repr(e)}

def list_dir(path: str = "."):
    """列出某个目录下的文件/文件夹（返回按字母排序的列表）。"""
    try:
        p = _safe(path)
        logger.debug("列出目录: {}", path)
        
        names = os.listdir(p)
        names.sort()
        
        logger.info("目录列表完成: {}, 找到 {} 个条目", path, len(names))
        return {"ok": True, "output": "\n".join(names)}
    except Exception as e:
        return _err("list_dir", e, path=path)

def grep_text(pattern: str, path: str = ".", max_matches: int = 50):
    """
    在 path 目录下递归搜索文本 pattern，返回匹配的文件与行号（最多 max_matches 条）。
    """
    try:
        import re

        logger.debug("文本搜索: {}, 路径: {}, 最大匹配数: {}", pattern, path, max_matches)
        
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
                                    logger.info("文本搜索达到最大匹配数: {}", max_matches)
                                    return {"ok": True, "output": "\n".join(hits)}
                except Exception:
                    # 单文件读不了就跳过
                    continue

        logger.info("文本搜索完成: {}, 找到 {} 个匹配", pattern, len(hits))
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
    "read_file_lines": read_file_lines,
    "copy_file": copy_file,
    "move_file": move_file,
    "delete_file": delete_file,
    "mkdir_p": mkdir_p,
}