"""Reuse of context in agent trajectories.

Bronze: raw rows from a public trajectory dataset (one row = one agent run).
Silver: one record per message ("chunk"): role, kind, content hash, size, and
        how many LLM calls in its own run carried it.
Gold:   per content hash across runs: how many runs, instances and repos share
        it, and what an ideal cache at each scope would save.

Sizes are in characters. Converting to tokens needs the serving model's
tokenizer and is done in the cost model, not here.

Supported row shapes (normalised by `messages_of`):
  nebius/SWE-agent-trajectories   row["trajectory"]: list (or JSON string) of
                                  {role: system|user|ai, text|system_prompt}
  nvidia/Open-SWE-Traces          row["messages"]: list of {role, content}
Field names were taken from the dataset cards, not yet from the files; the
normaliser fails loudly on a shape it does not recognise.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import re
from collections import Counter, defaultdict
from typing import Iterable, Iterator

ASSISTANT = {"ai", "assistant"}


def _text(m: dict) -> str:
    for k in ("content", "text", "system_prompt"):
        v = m.get(k)
        if isinstance(v, str) and v:
            return v
        if isinstance(v, list):                      # OpenAI content parts
            return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in v)
    calls = m.get("tool_calls")
    return json.dumps(calls, sort_keys=True) if calls else ""


def messages_of(row: dict) -> list[tuple[str, str]]:
    raw = row.get("messages", row.get("trajectory"))
    if isinstance(raw, str):
        raw = json.loads(raw)
    if not isinstance(raw, list) or not raw or not isinstance(raw[0], dict):
        raise ValueError(f"unrecognised trajectory shape; keys={sorted(row)}")
    out = []
    for m in raw:
        role = str(m.get("role", "")).lower()
        text = _text(m)
        if m.get("tool_calls") and m.get("content"):
            text += json.dumps(m["tool_calls"], sort_keys=True)
        out.append((role, text))
    return out


def repo_of(row: dict) -> str:
    if row.get("repo"):
        return str(row["repo"])
    iid = str(row.get("instance_id", ""))
    return re.sub(r"[-_](pr)?\d+$", "", iid) or "unknown"


def chunks(row: dict, traj_id: str) -> Iterator[dict]:
    """Silver records for one run."""
    msgs = messages_of(row)
    n_calls = sum(1 for r, _ in msgs if r in ASSISTANT)
    calls_seen = 0
    first_user = True
    for i, (role, text) in enumerate(msgs):
        if role in ASSISTANT:
            calls_seen += 1
            kind = "assistant"
        elif role == "system":
            kind = "system"
        elif first_user and role == "user":
            kind, first_user = "task", False
        else:
            kind = "observation"
        yield {
            "traj": traj_id, "instance": str(row.get("instance_id", "")), "repo": repo_of(row),
            "idx": i, "kind": kind, "chars": len(text),
            "sha": hashlib.sha1(text.encode("utf-8", "replace")).hexdigest(),
            # LLM calls in this run whose prompt contains this message
            "calls_carrying": n_calls - calls_seen,
        }
    if row.get("tools"):
        t = json.dumps(row["tools"], sort_keys=True, default=str)
        yield {"traj": traj_id, "instance": str(row.get("instance_id", "")), "repo": repo_of(row),
               "idx": -1, "kind": "tools", "chars": len(t),
               "sha": hashlib.sha1(t.encode()).hexdigest(), "calls_carrying": n_calls}


def summarize(rows: Iterable[dict]) -> dict:
    """Gold metrics over a stream of bronze rows."""
    n_traj = 0
    by_kind = defaultdict(lambda: {"chars": 0, "char_calls": 0, "chunks": 0})
    sha_chars: dict[str, int] = {}
    sha_kind: dict[str, str] = {}
    sha_trajs: dict[str, set] = defaultdict(set)
    sha_inst: dict[str, set] = defaultdict(set)
    sha_repo: dict[str, set] = defaultdict(set)
    unique_chars = uncached = 0
    amps = []
    for n_traj, row in enumerate(rows, 1):
        t_unique = t_uncached = 0
        for c in chunks(row, str(n_traj)):
            k = by_kind[c["kind"]]
            k["chars"] += c["chars"]; k["chunks"] += 1
            k["char_calls"] += c["chars"] * c["calls_carrying"]
            t_unique += c["chars"]
            t_uncached += c["chars"] * c["calls_carrying"]
            if c["chars"] >= 200:                    # ignore trivial strings
                s = c["sha"]
                sha_chars[s] = c["chars"]; sha_kind[s] = c["kind"]
                sha_trajs[s].add(c["traj"]); sha_inst[s].add(c["instance"]); sha_repo[s].add(c["repo"])
        unique_chars += t_unique; uncached += t_uncached
        if t_unique:
            amps.append(t_uncached / t_unique)
    # cross-run reuse: chars that an ideal shared cache would not have to prefill again
    scope = Counter(); scope_kind = defaultdict(Counter)
    for s, trajs in sha_trajs.items():
        extra = (len(trajs) - 1) * sha_chars[s]
        if extra <= 0:
            continue
        sc = ("cross_repo" if len(sha_repo[s]) > 1 else
              "cross_instance_same_repo" if len(sha_inst[s]) > 1 else "same_instance")
        scope[sc] += extra; scope_kind[sc][sha_kind[s]] += extra
    amps.sort()
    return {
        "trajectories": n_traj,
        "unique_input_chars": unique_chars,
        "prefill_chars_no_cache": uncached,
        "amplification_no_cache_median": round(amps[len(amps) // 2], 2) if amps else None,
        "by_kind": {k: v for k, v in sorted(by_kind.items())},
        "cross_run_reusable_chars": dict(scope),
        "cross_run_reusable_share_of_unique": {k: round(v / unique_chars, 4) for k, v in scope.items()} if unique_chars else {},
        "cross_run_by_kind": {k: dict(v) for k, v in scope_kind.items()},
    }


def parquet_rows(paths: list[str], limit: int | None = None) -> Iterator[dict]:
    import pyarrow.parquet as pq
    n = 0
    for p in paths:
        f = pq.ParquetFile(p)
        for b in f.iter_batches(batch_size=64):
            for row in b.to_pylist():
                yield row
                n += 1
                if limit and n >= limit:
                    return


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("parquet", nargs="+", help="parquet files or globs")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out")
    a = ap.parse_args()
    paths = sorted(p for g in a.parquet for p in glob.glob(g))
    res = summarize(parquet_rows(paths, a.limit))
    res["source_files"] = len(paths)
    text = json.dumps(res, indent=2)
    if a.out:
        open(a.out, "w").write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
