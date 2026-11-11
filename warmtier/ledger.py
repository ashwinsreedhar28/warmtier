"""Edit-aware context ledger: know which file views in an agent's context are stale.

A harness calls this on its message list (OpenAI chat format) before each
model call. The ledger finds every tool result that shows file content, tracks
a version per file path that advances on every successful edit, and reports
which views show a version that is no longer current.

    led = FileLedger("claudecode")
    views = led.scan(messages)                    # every file view, with staleness
    idx = led.plan(messages, mode="batch")        # message indices worth dropping now
    messages = led.apply(messages, idx)           # replace them with one-line stubs

    tag = led.tag("parser lives in a.py", ["/repo/a.py"], messages)
    led.valid(tag, later_messages)                # False once /repo/a.py was edited

Two staleness levels, as in the measurement code:
  file    the file was edited after the view
  window  an edit replaced text that the view showed (falls back to file level
          when the edit does not say what it replaced)

Modes for `plan`:
  splice  return every stale view; for an engine that can remove a span from
          cached KV without re-prefilling what follows
  batch   return stale views only once they add up to `threshold` of the
          context, so one cache invalidation pays for several drops

The ledger sees edits made through the harness's own tools and simple shell
rewrites (sed -i, redirects). A harness that can diff its workspace should
pass `changed_paths` to catch everything else.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from .claudecode import _bash_edit, _bash_view, _strip_numbers

DIALECTS = {
    "claudecode": {"read": {"Read"}, "edit": {"Edit"}, "write": {"Write"}, "bash": {"Bash"},
                   "path": ("file_path",), "old": ("old_string",),
                   "edit_ok": lambda t: t.startswith("The file ") and "updated" in t[:400]},
    "shwin": {"read": {"read_file"}, "edit": {"edit_file"}, "write": {"write_file"}, "bash": {"bash"},
              "path": ("path",), "old": ("old", "old_string", "old_text", "search"),
              "edit_ok": lambda t: not t.lstrip().startswith("ERROR")},
}
_RESULT = re.compile(r"^<tool_result name='([^']+)'>\n(.*)\n</tool_result>", re.DOTALL)
STUB = "[stale view of {path}: the file was edited later in this session; read it again if you need it]"


@dataclass
class View:
    index: int                    # position in the message list
    path: str
    chars: int
    call: int                     # model calls made before this view arrived
    version: int                  # file version the view shows
    stale_file: int | None = None      # call index from which it is stale, file level
    stale_window: int | None = None    # same, window level
    body: str = field(default="", repr=False)


def _calls_of(m: dict) -> list[tuple[str, dict]]:
    out = []
    for tc in m.get("tool_calls") or []:
        fn = tc.get("function", {})
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except (ValueError, TypeError):
            args = {}
        out.append((fn.get("name", ""), args if isinstance(args, dict) else {}))
    if not out:                                        # text protocol: one JSON envelope
        c = m.get("content") or ""
        i = c.find('{"type"')
        if i >= 0:
            try:
                env = json.JSONDecoder().raw_decode(c[i:])[0]
                if env.get("type") == "tool_call":
                    out.append((env.get("name", ""), env.get("arguments") or {}))
            except ValueError:
                pass
    return out


def _result_text(m: dict) -> str | None:
    c = m.get("content")
    if not isinstance(c, str):
        return None
    if m.get("role") == "tool":
        return c
    if m.get("role") == "user":
        mm = _RESULT.match(c)
        if mm:
            return mm.group(2)
    return None


class FileLedger:
    def __init__(self, dialect: str = "claudecode"):
        self.d = DIALECTS[dialect]

    def _first(self, args: dict, keys: tuple) -> str | None:
        for k in keys:
            if isinstance(args.get(k), str) and args[k]:
                return args[k]
        return None

    def scan(self, messages: list[dict], changed_paths: set[str] | None = None) -> list[View]:
        d = self.d
        views: list[View] = []
        version: dict[str, int] = {}
        call = 0
        pending: list[tuple[str, dict]] = []

        def edited(path: str, old: str | None) -> None:
            version[path] = version.get(path, 0) + 1
            for v in views:
                if v.path != path or v.call >= call:
                    continue
                if v.stale_file is None:
                    v.stale_file = call
                if v.stale_window is None and (old is None or old in v.body):
                    v.stale_window = call

        for i, m in enumerate(messages):
            if m.get("role") == "assistant":
                pending = _calls_of(m)
                call += 1
                continue
            text = _result_text(m)
            if text is None or not pending:
                continue
            name, args = pending.pop(0)
            path = self._first(args, d["path"])
            if name in d["read"] and path and not text.lstrip().startswith(("ERROR", "<tool_use_error>")):
                views.append(View(i, path, len(text), call, version.get(path, 0), body=_strip_numbers(text)))
            elif name in d["edit"] and path and d["edit_ok"](text):
                edited(path, self._first(args, d["old"]) or None)
            elif name in d["write"] and path and d["edit_ok"](text):
                edited(path, None)
            elif name in d["bash"]:
                cmd = args.get("command") or ""
                target = _bash_edit(cmd)
                if target:
                    edited(target, None)
                else:
                    bv = _bash_view(cmd)
                    if bv and len(text) >= 200:
                        views.append(View(i, bv[0], len(text), call, version.get(bv[0], 0), body=_strip_numbers(text)))
        for path in changed_paths or ():                 # edits the transcript did not show
            for v in views:
                if v.path == path and v.stale_file is None:
                    v.stale_file = v.stale_window = call
        self._version = version
        return views

    def stale(self, messages: list[dict], level: str = "window", changed_paths: set[str] | None = None) -> list[View]:
        key = "stale_window" if level == "window" else "stale_file"
        return [v for v in self.scan(messages, changed_paths) if getattr(v, key) is not None]

    def plan(self, messages: list[dict], mode: str = "batch", threshold: float = 0.25, level: str = "window",
             changed_paths: set[str] | None = None) -> list[int]:
        """Message indices to drop now. Views already stubbed are ignored."""
        s = [v for v in self.stale(messages, level, changed_paths)
             if "[stale view of " not in (messages[v.index].get("content") or "")[:200]]
        if not s:
            return []
        if mode == "splice":
            return [v.index for v in s]
        total = sum(len(m.get("content") or "") for m in messages if isinstance(m.get("content"), str))
        return [v.index for v in s] if sum(v.chars for v in s) >= threshold * max(total, 1) else []

    def apply(self, messages: list[dict], indices: list[int]) -> list[dict]:
        paths = {v.index: v.path for v in self.scan(messages)}
        out = list(messages)
        for i in indices:
            m = out[i]
            stub = STUB.format(path=paths.get(i, "a file"))
            if m.get("role") == "tool":
                out[i] = {**m, "content": stub}
            else:
                name = (_RESULT.match(m.get("content") or "") or [None, "tool"])[1]
                out[i] = {**m, "content": f"<tool_result name='{name}'>\n{stub}\n</tool_result>"}
        return out

    def tag(self, note: str, paths: list[str], messages: list[dict]) -> dict:
        self.scan(messages)
        return {"note": note, "deps": {p: self._version.get(p, 0) for p in paths}}

    def valid(self, tagged: dict, messages: list[dict], changed_paths: set[str] | None = None) -> bool:
        self.scan(messages)
        changed = changed_paths or set()
        return all(self._version.get(p, 0) == v and p not in changed for p, v in tagged["deps"].items())
