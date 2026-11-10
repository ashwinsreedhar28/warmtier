"""Silver tables for SWE-agent trajectories (nebius/SWE-agent-trajectories).

Bronze row -> three silver tables, written as parquet per source file:

  chunks  one row per message: kind, command, size, content hash, and the
          index of the first LLM call whose prompt carries it
  views   one row per file view (an observation that starts with
          "[File: <path> (<n> lines total)]"): which file, which lines, and
          the call at which a later successful edit to that file made it stale
  runs    one row per run with totals

Model of the prompt: append-only. Call k's prompt is every message before the
k-th assistant message, so a message first carried at call c in a run of n
calls is prefilled (n - c) times without a cache. SWE-agent itself elides old
observations when it renders the prompt; this dataset stores the trajectory,
not the rendered prompts, so numbers here describe an append-only harness
replaying the same trajectory.

Staleness is counted two ways:
  file    any later successful edit to the same file (upper bound)
  window  a later successful edit whose line range intersects the lines the
          view showed (lower bound; ignores line-number shifts)
Edits made through shell commands (sed, python, rm) are not detected, so both
are undercounts of the true edit rate.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import os
import re
from multiprocessing import Pool

import pyarrow as pa
import pyarrow.parquet as pq

FILE_HDR = re.compile(r"\[File: (.+?) \((\d+) lines total\)\]")
NUMBERED = re.compile(r"^(\d+):", re.M)
BLOCK = re.compile(r"```[a-z]*\n?(.*?)```", re.S)
EDIT_RANGE = re.compile(r"^edit\s+(\d+):(\d+)")
VIEW_CMDS = {"open", "goto", "scroll_down", "scroll_up", "edit", "create"}


def _sha(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", "replace")).hexdigest()[:16]


def _command(ai_text: str) -> str:
    blocks = BLOCK.findall(ai_text or "")
    body = blocks[-1].strip() if blocks else ""
    return body


def repo_of(instance_id: str) -> str:
    return re.sub(r"-\d+$", "", instance_id)


def process_run(row: dict, run_id: int):
    traj = row["trajectory"]
    iid = row["instance_id"]
    repo = repo_of(iid)
    n_calls = sum(1 for m in traj if m["role"] == "ai")
    chunks, views, edits = [], [], []          # edits: (call_idx, file, start, end)
    created = set()                            # files the agent created in this run
    call = 0                                   # assistant messages seen so far
    last_cmd = ""
    first_user = True
    tot = {"system": 0, "task": 0, "assistant": 0, "observation": 0}
    prefill = 0
    for idx, m in enumerate(traj):
        role = m["role"]
        text = m.get("text") or m.get("system_prompt") or ""
        cmd_word = ""
        if role == "ai":
            kind = "assistant"
            last_cmd = _command(text)
            cmd_word = last_cmd.split()[0] if last_cmd else ""
            call += 1
        elif role == "system":
            kind = "system"
        elif first_user:
            kind, first_user = "task", False
        else:
            kind = "observation"
            cmd_word = last_cmd.split()[0] if last_cmd else ""
        carried = n_calls - call               # calls whose prompt holds this message
        tot[kind] += len(text)
        prefill += len(text) * carried
        chunks.append((run_id, iid, repo, idx, call, kind, cmd_word[:24], len(text), _sha(text), carried))
        if kind == "observation":
            h = FILE_HDR.search(text[:400])
            if h and cmd_word in VIEW_CMDS:
                nums = NUMBERED.findall(text)
                lo, hi = (int(nums[0]), int(nums[-1])) if nums else (0, 0)
                ok_edit = cmd_word == "edit" and "File updated" in text
                if cmd_word == "create":
                    created.add(h.group(1))
                if ok_edit or cmd_word == "create":
                    r = EDIT_RANGE.match(last_cmd)
                    s, e = (int(r.group(1)), int(r.group(2))) if r else (1, 1)
                    edits.append((call, h.group(1), s, e))
                views.append([run_id, iid, repo, h.group(1), call, len(text), _sha(text), lo, hi, carried, None, None,
                              h.group(1) in created])
    # staleness: first later edit to the same file / intersecting the same lines
    for v in views:
        f, c, lo, hi = v[3], v[4], v[7], v[8]
        for (ec, ef, s, e) in edits:
            if ec > c and ef == f:
                if v[10] is None:
                    v[10] = ec
                if v[11] is None and s <= hi and e >= lo:
                    v[11] = ec
                if v[10] is not None and v[11] is not None:
                    break
    view_chars = sum(v[5] for v in views)
    view_cc = sum(v[5] * v[9] for v in views)
    stale_file = sum(v[5] * (n_calls - v[10]) for v in views if v[10] is not None)
    stale_win = sum(v[5] * (n_calls - v[11]) for v in views if v[11] is not None)
    run = (run_id, iid, repo, row["model_name"], bool(row["target"]), row["exit_status"], n_calls,
           tot["system"], tot["task"], tot["assistant"], tot["observation"], prefill,
           len(views), len({v[3] for v in views}), len(edits), view_chars, view_cc, stale_file, stale_win)
    return chunks, [tuple(v) for v in views], run


CHUNK_COLS = ["run", "instance", "repo", "idx", "call", "kind", "cmd", "chars", "sha", "carried"]
VIEW_COLS = ["run", "instance", "repo", "file", "call", "chars", "sha", "line_lo", "line_hi", "carried",
             "stale_file_call", "stale_window_call", "scratch"]
RUN_COLS = ["run", "instance", "repo", "model", "resolved", "exit", "n_calls", "system_chars", "task_chars",
            "assistant_chars", "observation_chars", "prefill_chars_no_cache", "n_views", "n_files_viewed",
            "n_edits", "view_chars", "view_char_calls", "stale_file_char_calls", "stale_window_char_calls"]


def _write(path: str, cols: list[str], rows: list[tuple]) -> None:
    arrays = [pa.array([r[i] for r in rows]) for i in range(len(cols))]
    pq.write_table(pa.table(arrays, names=cols), path, compression="zstd")


def process_file(args) -> int:
    path, file_no, out = args
    chunks, views, runs = [], [], []
    n = 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=32):
        for row in batch.to_pylist():
            c, v, r = process_run(row, file_no * 1_000_000 + n)
            chunks += c; views += v; runs.append(r)
            n += 1
    stem = f"{file_no:05d}.parquet"
    _write(os.path.join(out, "chunks", stem), CHUNK_COLS, chunks)
    _write(os.path.join(out, "views", stem), VIEW_COLS, views)
    _write(os.path.join(out, "runs", stem), RUN_COLS, runs)
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("parquet_glob")
    ap.add_argument("--out", default="data/silver/nebius")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    a = ap.parse_args()
    for t in ("chunks", "views", "runs"):
        os.makedirs(os.path.join(a.out, t), exist_ok=True)
    paths = sorted(glob.glob(a.parquet_glob, recursive=True))
    with Pool(a.workers) as pool:
        total = sum(pool.imap_unordered(process_file, [(p, i, a.out) for i, p in enumerate(paths)]))
    print(f"{total} runs from {len(paths)} files -> {a.out}")


if __name__ == "__main__":
    main()
