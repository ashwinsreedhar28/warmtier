"""Stale-view eviction: what it saves in context versus what it costs in cache.

    python -m warmtier.evict 'data/silver/nebius/chunks/*.parquet' 'data/silver/nebius/views/*.parquet' \\
        --out results/evict_nebius.json

Replays each run call by call under a prefix cache and compares policies for
file views that an edit has made stale (see sweagent.py / claudecode.py for
how staleness is detected):

  append      never evict (today's append-only harness)
  evict_now   drop a stale view at the first call after the edit. Everything
              after the earliest dropped view loses its cached KV and is
              prefilled again
  lazy_10/25  the same, but only once stale content reaches 10% / 25% of the
              context, so several drops share one re-prefill
  splice      drop stale views with no re-prefill (an engine that can edit
              cached KV in place; the upper bound)

Two totals per policy, both relative to append:
  prefill   characters the engine actually has to prefill. With a prefix cache
            that is new tool output and user text, plus anything re-prefilled
            after an eviction. Assistant output is written by decode and is
            free unless it has to be redone
  context   sum over calls of the context length: what decode attends over and
            what KV memory holds, call after call

Staleness level: "window" (the edit touched the lines shown; conservative) or
"file" (any edit to that file).
"""
from __future__ import annotations

import argparse
import glob
import json
from multiprocessing import Pool

import pyarrow.parquet as pq

POLICIES = ("append", "evict_now", "lazy_10", "lazy_25", "splice")
_VIEWS: dict = {}


def simulate(msgs: list[tuple], policy: str) -> tuple[int, int, int]:
    """msgs: (call, chars, is_assistant, stale_call or None), in prompt order.
    Returns (prefill_chars, context_integral, eviction_events)."""
    n_calls = sum(1 for m in msgs if m[2])
    m = len(msgs)
    present = [True] * m
    cached = [False] * m
    arrived = 0                      # msgs[:arrived] have call <= c
    L = prefill = ctx = events = 0
    theta = {"lazy_10": 0.10, "lazy_25": 0.25}.get(policy, 0.0)
    stale_idx = sorted((i for i in range(m) if msgs[i][3] is not None), key=lambda i: msgs[i][3])
    sp = 0
    pending: list[int] = []
    pending_chars = 0
    for c in range(n_calls):
        new = 0
        while arrived < m and msgs[arrived][0] <= c:
            call, chars, is_asst, _ = msgs[arrived]
            L += chars
            if is_asst:
                cached[arrived] = True          # decode already wrote its KV
            else:
                new += chars
                cached[arrived] = True
            arrived += 1
        if policy != "append":
            while sp < len(stale_idx) and msgs[stale_idx[sp]][3] <= c:
                i = stale_idx[sp]; sp += 1
                if i < arrived and present[i]:
                    pending.append(i); pending_chars += msgs[i][1]
            if pending and pending_chars >= theta * L:
                first = min(pending)
                for i in pending:
                    present[i] = False
                    L -= msgs[i][1]
                if policy != "splice":
                    for i in range(first + 1, arrived):
                        if present[i]:
                            new += msgs[i][1] if (msgs[i][0] < c or msgs[i][2]) else 0
                events += 1
                pending, pending_chars = [], 0
        prefill += new
        ctx += L
    return prefill, ctx, events


def run_file(args):
    path, level = args
    t = pq.read_table(path, columns=["run", "idx", "call", "kind", "chars", "sha"]).sort_by([("run", "ascending"), ("idx", "ascending")])
    cols = [t.column(c).to_pylist() for c in ("run", "call", "kind", "chars", "sha")]
    tot = {p: [0, 0, 0] for p in POLICIES}
    runs = 0
    cur, msgs = None, []

    def flush():
        nonlocal runs
        if not msgs:
            return
        runs += 1
        for p in POLICIES:
            a, b, e = simulate(msgs, p)
            tot[p][0] += a; tot[p][1] += b; tot[p][2] += e

    for run, call, kind, chars, sha in zip(*cols):
        if run != cur:
            flush()
            cur, msgs = run, []
        st = _VIEWS.get((run, call, sha))
        stale = None if st is None else st[level]
        msgs.append((call, chars, kind == "assistant", stale))
    flush()
    return runs, tot


def load_views(paths: list[str]) -> None:
    for p in paths:
        t = pq.read_table(p, columns=["run", "call", "sha", "stale_file_call", "stale_window_call"])
        for run, call, sha, sf, sw in zip(*(t.column(c).to_pylist() for c in t.column_names)):
            if sf is not None or sw is not None:
                _VIEWS[(run, call, sha)] = (sf, sw)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("chunks_glob")
    ap.add_argument("views_glob")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out")
    a = ap.parse_args()
    load_views(sorted(glob.glob(a.views_glob)))
    paths = sorted(glob.glob(a.chunks_glob))
    res = {}
    for level, name in ((1, "window"), (0, "file")):
        with Pool(a.workers) as pool:
            parts = pool.map(run_file, [(p, level) for p in paths])
        runs = sum(r for r, _ in parts)
        tot = {p: [sum(t[p][k] for _, t in parts) for k in range(3)] for p in POLICIES}
        base = tot["append"]
        res[name] = {"runs": runs, "append_prefill_chars": base[0], "append_context_integral": base[1],
                     "policies": {p: {"prefill_vs_append": round(tot[p][0] / base[0], 4),
                                      "context_vs_append": round(tot[p][1] / base[1], 4),
                                      "evictions_per_run": round(tot[p][2] / runs, 2)} for p in POLICIES}}
    text = json.dumps(res, indent=2)
    if a.out:
        open(a.out, "w").write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
