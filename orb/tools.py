"""
Obscuro Tool Registry — the action space of the unified autonomous agent.

Tools are the bridge between language generation and real-world effect.
Each tool takes structured JSON args and returns a plain-text Observation.
The agent loop calls these; the model never touches them directly.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
import traceback
import urllib.request
from dataclasses import dataclass
from typing import Callable

_WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MAX_OUTPUT = 6000  # max chars returned from any single tool call


@dataclass
class ToolResult:
    tool: str
    args: dict
    output: str
    success: bool
    elapsed_ms: int = 0

    def format(self) -> str:
        status = "✓" if self.success else "✗"
        out = self.output
        if len(out) > _MAX_OUTPUT:
            out = out[:_MAX_OUTPUT] + f"\n... (truncated — {len(self.output):,} total chars)"
        return f"[{status} {self.tool} | {self.elapsed_ms}ms]\n{out}"


class ToolRegistry:
    """
    Central registry of all tools available to Obscuro.
    Tools are pluggable — register any callable via register(name, fn).

    Built-in tools:
      shell       — execute any bash command in the workspace
      python      — execute Python code via subprocess
      file_read   — read a file (up to 8 000 chars)
      file_write  — write / create a file (creates parent dirs automatically)
      file_delete — delete a file or directory tree
      file_list   — list directory contents
      web_fetch   — fetch a URL as plain text (HTML stripped)
      think       — record extended internal reasoning (no side effect)
    """

    SCHEMA = (
        "TOOLS — emit as single-line JSON with \"tool\" and \"args\" keys:\n"
        '  shell       {"tool":"shell","args":{"cmd":"ls -la"}}\n'
        '  python      {"tool":"python","args":{"code":"print(2**10)"}}\n'
        '  file_read   {"tool":"file_read","args":{"path":"src/main.py"}}\n'
        '  file_write  {"tool":"file_write","args":{"path":"out.py","content":"..."}}\n'
        '  file_delete {"tool":"file_delete","args":{"path":"old.txt"}}\n'
        '  file_list   {"tool":"file_list","args":{"path":"."}}\n'
        '  web_fetch   {"tool":"web_fetch","args":{"url":"https://..."}}\n'
        '  think       {"tool":"think","args":{"content":"extended reasoning..."}}'
    )

    def __init__(self) -> None:
        self._tools: dict[str, Callable[..., str]] = {}
        self._register_builtins()

    def _register_builtins(self) -> None:
        self.register("shell",       self._shell)
        self.register("python",      self._python)
        self.register("file_read",   self._file_read)
        self.register("file_write",  self._file_write)
        self.register("file_delete", self._file_delete)
        self.register("file_list",   self._file_list)
        self.register("web_fetch",   self._web_fetch)
        self.register("think",       self._think)

    def register(self, name: str, fn: Callable[..., str]) -> None:
        self._tools[name] = fn

    def call(self, name: str, args: dict) -> ToolResult:
        t0 = time.monotonic()
        if name not in self._tools:
            available = ", ".join(sorted(self._tools))
            return ToolResult(
                name, args,
                f"Unknown tool '{name}'. Available: {available}",
                False,
            )
        try:
            output = self._tools[name](**args)
            elapsed = int((time.monotonic() - t0) * 1000)
            return ToolResult(name, args, str(output), True, elapsed)
        except TypeError as exc:
            return ToolResult(
                name, args,
                f"Bad arguments for '{name}': {exc}",
                False,
                int((time.monotonic() - t0) * 1000),
            )
        except Exception:
            return ToolResult(
                name, args,
                f"Tool error:\n{traceback.format_exc()[-900:]}",
                False,
                int((time.monotonic() - t0) * 1000),
            )

    @property
    def names(self) -> list[str]:
        return list(self._tools)

    # ── Built-in tool implementations ─────────────────────────────────────────

    def _shell(self, cmd: str, timeout: int = 45) -> str:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True,
            timeout=timeout, cwd=_WORKSPACE,
        )
        out = (result.stdout + result.stderr).strip()
        if result.returncode != 0:
            out = f"[exit {result.returncode}]\n{out}"
        return out[:_MAX_OUTPUT] if out else "(no output)"

    def _python(self, code: str, timeout: int = 30) -> str:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True,
            timeout=timeout, cwd=_WORKSPACE,
        )
        out = (result.stdout + result.stderr).strip()
        if result.returncode != 0:
            out = f"[exit {result.returncode}]\n{out}"
        return out[:_MAX_OUTPUT] if out else "(executed, no output)"

    def _file_read(self, path: str, max_bytes: int = 8000) -> str:
        p = path if os.path.isabs(path) else os.path.join(_WORKSPACE, path)
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            content = f.read(max_bytes)
        if len(content) == max_bytes:
            content += f"\n\n... (truncated at {max_bytes:,} chars)"
        return content

    def _file_write(self, path: str, content: str) -> str:
        p = path if os.path.isabs(path) else os.path.join(_WORKSPACE, path)
        parent = os.path.dirname(p)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Written {len(content):,} chars → {p}"

    def _file_delete(self, path: str) -> str:
        import shutil
        p = path if os.path.isabs(path) else os.path.join(_WORKSPACE, path)
        if os.path.isfile(p):
            os.remove(p)
            return f"Deleted file: {p}"
        elif os.path.isdir(p):
            shutil.rmtree(p)
            return f"Deleted directory tree: {p}"
        return f"Not found: {p}"

    def _file_list(self, path: str = ".") -> str:
        p = path if os.path.isabs(path) else os.path.join(_WORKSPACE, path)
        try:
            entries = sorted(os.scandir(p), key=lambda e: (e.is_file(), e.name))
        except NotADirectoryError:
            return f"Not a directory: {p}"
        lines = []
        for e in entries[:200]:
            kind = "FILE" if e.is_file() else "DIR "
            size = f"  ({e.stat().st_size:,} B)" if e.is_file() else ""
            lines.append(f"{kind}  {e.name}{size}")
        if len(entries) > 200:
            lines.append(f"... ({len(entries) - 200} more entries not shown)")
        return "\n".join(lines) if lines else "(empty directory)"

    def _web_fetch(self, url: str, max_bytes: int = 8000) -> str:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Obscuro/2.0 (autonomous-intelligence)"},
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            raw = resp.read(max_bytes).decode("utf-8", errors="replace")
        # Strip style/script blocks then all tags
        clean = re.sub(r"<style[^>]*>.*?</style>", " ", raw, flags=re.DOTALL)
        clean = re.sub(r"<script[^>]*>.*?</script>", " ", clean, flags=re.DOTALL)
        clean = re.sub(r"<[^>]+>", " ", clean)
        clean = re.sub(r"\s+", " ", clean).strip()
        return clean[:max_bytes]

    def _think(self, content: str) -> str:
        """Record extended internal reasoning — no external side effects."""
        word_count = len(content.split())
        return f"[Internal reasoning recorded: {word_count} words — no external action taken]"
