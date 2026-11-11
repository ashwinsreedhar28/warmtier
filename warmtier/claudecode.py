"""Silver tables for Claude Code style traces in OpenAI message format
(nvidia/Open-SWE-Traces, claudecode split). Same three tables and columns as
sweagent.py, so gold.py and cost.py run unchanged.

A view is a tool result that shows file content:
  Read    the Read tool (file_path, optional offset/limit)
  Bash    a single shell command that only prints one file: cat, head, tail,
          or sed -n 'A,Bp' (best effort; pipes and compound commands are not
          counted)
An edit is a successful Edit tool call, or a Bash command that rewrites a file
in place (sed -i, or a redirect into a path). Bash edits are best effort.

Staleness:
  file    any later edit to the same file
  window  a later Edit whose old_string occurs in the text the view showed
          (Bash edits never count at window level)

The prompt model is append-only: system prompt, tool schemas, then every
message. reasoning_content is not counted as carried context; the runs table
records its size and what carrying it would add (reasoning_char_calls).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shlex
from multiprocessing import Pool

import pyarrow.parquet as pq

from .sweagent import CHUNK_COLS, RUN_COLS, VIEW_COLS, _sha, _write

LINE_NO = re.compile(r"^\s*(\d+)[\t→]", re.M)
SED_RANGE = re.compile(r"^(\d+),(\d+)p$")
REDIRECT = re.compile(r">>?\s*([^\s;&|<>]+)")


def _bash_view(cmd: str):
    """(path, lo, hi) if the command only prints one file, else None."""
    if any(t in cmd for t in ("|", "&&", ";", ">", "$(", "`", "\n")):
        return None
    try:
        tok = shlex.split(cmd)
    except ValueError:
        return None
    if not tok:
        return None
    paths = [t for t in tok[1:] if "/" in t or "." in t and not t.startswith("-")]
    paths = [t for t in paths if not SED_RANGE.match(t) and not t.startswith("-")]
    if len(paths) != 1:
        return None
    if tok[0] == "cat":
        return paths[0], 0, 0
    if tok[0] in ("head", "tail"):
        return paths[0], 0, 0
    if tok[0] == "sed" and "-n" in tok and "-i" not in tok:
        for t in tok[1:]:
            m = SED_RANGE.match(t)
            if m:
                return paths[0], int(m.group(1)), int(m.group(2))
    return None


def _bash_edit(cmd: str):
    """Path rewritten by the command, if it is recognisably an in-place edit."""
    m = re.search(r"\bsed\s+(?:-[a-zA-Z]*\s+)*-i\b.*?\s([^\s'\";|&]+)\s*(?:$|[;&|])", cmd, re.S)
    if m:
        return m.group(1)
    if re.match(r"\s*(cat|echo|printf|tee)\b", cmd):
        r = REDIRECT.search(cmd)
        if r and not r.group(1).startswith(("/dev/", "&")):
            return r.group(1)
    return None


def _strip_numbers(text: str) -> str:
    return re.sub(r"^\s*\d+[\t→]", "", text, flags=re.M)


def process_run(row: dict, run_id: int):
    msgs = row["messages"]
    iid, repo = row["instance_id"], row.get("repo") or ""
    n_calls = sum(1 for m in msgs if m["role"] == "assistant")
    chunks, views, edits = [], [], []        # edits: (call, file, old_string or None)
    created = set()
    tot = {"system": 0, "task": 0, "assistant": 0, "observation": 0, "tools": 0}
    prefill = 0
    call = 0
    first_user = True
    pending: list[tuple[str, dict]] = []
    reasoning = reasoning_cc = 0             # reasoning text, and what carrying it would add

    tools_text = json.dumps(row.get("tools") or [], sort_keys=True)
    tot["tools"] = len(tools_text)
    prefill += len(tools_text) * n_calls
    chunks.append((run_id, iid, repo, -1, 0, "tools", "", len(tools_text), _sha(tools_text), n_calls))

    for idx, m in enumerate(msgs):
        role = m["role"]
        text = m.get("content") or ""
        cmd_word = ""
        if role == "assistant":
            kind = "assistant"
            pending = []
            for tc in m.get("tool_calls") or []:
                fn = tc["function"]
                try:
                    args = json.loads(fn["arguments"])
                except (ValueError, TypeError):
                    args = {}
                pending.append((fn["name"], args if isinstance(args, dict) else {}))
                text += fn["name"] + (fn["arguments"] or "")
            cmd_word = pending[0][0] if pending else ""
            call += 1
            r = len(m.get("reasoning_content") or "")
            reasoning += r
            reasoning_cc += r * (n_calls - call)
        elif role == "system":
            kind = "system"
        elif role == "user" and first_user:
            kind, first_user = "task", False
        elif role == "tool":
            kind = "observation"
        else:
            kind = "observation"                 # later user turns (harness reminders)
        carried = n_calls - call
        tot[kind] += len(text)
        prefill += len(text) * carried
        if role == "tool" and pending:
            name, args = pending.pop(0)
            cmd_word = name
            view = None
            if name == "Read" and args.get("file_path") and LINE_NO.match(text):
                nums = LINE_NO.findall(text)
                view = (args["file_path"], int(nums[0]), int(nums[-1]))
            elif name == "Bash":
                c = args.get("command") or ""
                cmd_word = "Bash:" + (c.split()[0] if c.split() else "")
                ed = _bash_edit(c)
                if ed:
                    if not any(v[3] == ed for v in views):
                        created.add(ed)
                    edits.append((call, ed, None))
                else:
                    bv = _bash_view(c)
                    if bv and len(text) >= 200:
                        view = bv
            elif name == "Edit" and args.get("file_path") and text.startswith("The file ") and "updated" in text[:400]:
                edits.append((call, args["file_path"], args.get("old_string") or ""))
            if view:
                views.append([run_id, iid, repo, view[0], call, len(text), _sha(text), view[1], view[2],
                              carried, None, None, view[0] in created, text if name == "Read" else _strip_numbers(text)])
        chunks.append((run_id, iid, repo, idx, call, kind, cmd_word[:24], len(text), _sha(text), carried))

    for v in views:
        f, c, body = v[3], v[4], _strip_numbers(v[13])
        for (ec, ef, old) in edits:
            if ec > c and ef == f:
                if v[10] is None:
                    v[10] = ec
                if v[11] is None and old and old in body:
                    v[11] = ec
                if v[10] is not None and v[11] is not None:
                    break
        v.pop()                                   # drop the text
    view_chars = sum(v[5] for v in views)
    view_cc = sum(v[5] * v[9] for v in views)
    stale_file = sum(v[5] * (n_calls - v[10]) for v in views if v[10] is not None)
    stale_win = sum(v[5] * (n_calls - v[11]) for v in views if v[11] is not None)
    model = ((row.get("metadata") or {}).get("teacher_model") or {}).get("name") or ""
    run = (run_id, iid, repo, model, row.get("resolved") == 1, "", n_calls,
           tot["system"] + tot["tools"], tot["task"], tot["assistant"], tot["observation"], prefill,
           len(views), len({v[3] for v in views}), len(edits), view_chars, view_cc, stale_file, stale_win,
           reasoning, reasoning_cc)
    return chunks, [tuple(v) for v in views], run


def process_file(args) -> int:
    path, file_no, out, force = args
    stem = f"{file_no:05d}.parquet"
    if not force and os.path.exists(os.path.join(out, "runs", stem)):
        return 0
    chunks, views, runs = [], [], []
    n = 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=8):
        for row in batch.to_pylist():
            c, v, r = process_run(row, file_no * 1_000_000 + n)
            chunks += c; views += v; runs.append(r)
            n += 1
    _write(os.path.join(out, "chunks", stem), CHUNK_COLS, chunks)
    _write(os.path.join(out, "views", stem), VIEW_COLS, views)
    _write(os.path.join(out, "runs", stem), RUN_COLS, runs)
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("parquet_glob")
    ap.add_argument("--out", default="data/silver/claudecode")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--only", type=int, nargs="*", help="process only these file numbers")
    ap.add_argument("--force", action="store_true", help="rewrite outputs that already exist")
    a = ap.parse_args()
    for t in ("chunks", "views", "runs"):
        os.makedirs(os.path.join(a.out, t), exist_ok=True)
    paths = sorted(glob.glob(a.parquet_glob, recursive=True))
    jobs = [(p, i, a.out, a.force) for i, p in enumerate(paths) if a.only is None or i in a.only]
    with Pool(a.workers) as pool:
        total = sum(pool.imap_unordered(process_file, jobs))
    print(f"{total} runs from {len(jobs)} of {len(paths)} files -> {a.out}")


if __name__ == "__main__":
    main()
