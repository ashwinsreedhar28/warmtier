"""Which signal should decide what leaves an agent's context? A replay of eviction policies.

    python -m warmtier.policy data/silver/claudecode --parts results/policy_parts_claudecode --budget-s 150
    python -m warmtier.policy data/silver/claudecode --parts results/policy_parts_claudecode --merge results/policy_claudecode.json

Engines can now edit cached KV in place (Leyline, arXiv 2606.01065) and the open question is what
should drive the edits. This replays each recorded run call by call and applies one policy for
choosing what to drop, under each kind of engine:

policies (what is dropped, and when)
  age2        every tool output over 400 characters, two calls after it arrived, replaced by a
              200-character stub. Leyline's evaluated policy: it needs no knowledge of the tools
  edit        a file view, at the call where an edit touched the lines it shows (window level)
  edit_file   a file view, at the call where any edit touched its file
  dup         a file view, when a later view of the same file covers the same lines or has the
              same bytes: the older copy is redundant
  composite_file   edit_file or dup, whichever comes first
  age5, age10 age2 with a longer wait
  semantic2, semantic5   tool-aware: file views leave only when stale or duplicated (composite_file);
              every other tool output (test runs, greps, listings) ages out after 2 or 5 calls
Views dropped by the tool-aware policies leave an 80-character stub.

engines (what a drop costs)
  prefix      prefix cache only: everything after the first dropped message is prefilled again
  prefix_lazy the same, but drops wait until they add up to 25% of the context
  splice      cached KV is edited in place: only the stubs are prefilled

Per policy and engine, relative to never evicting: prefill characters and the context integral
(sum over calls of context length: what KV memory holds and decode attends over).

Per policy, what was dropped, by what it was at the moment it was dropped:
  stale   a file view already invalidated by an edit (file level)
  dup     a file view already covered by a later view
  live    a file view that was neither: still current and present only once
  other   a tool output that is not a file view (test run, grep, listing); the replay cannot tell
          whether it was still needed
and `blind_first_edits`: file views dropped before the first edit the agent made to that file,
that is, cases where the recorded agent went on to edit a file whose text the policy had removed.

Limits: the agent's behaviour is replayed, not re-run. A policy that removes text the agent later
needs would make a real agent re-read it; that cost is not in the prefill numbers, only counted
(live share, blind first edits). Later edits to an already-edited file are not tracked.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time
from collections import defaultdict

import pyarrow.parquet as pq

POLICIES = ("age2", "age5", "age10", "edit", "edit_file", "dup", "composite_file", "semantic2", "semantic5")
ENGINES = ("prefix", "prefix_lazy", "splice")
AGE_MIN_CHARS, AGE_STUB, VIEW_STUB = 400, 200, 80


def simulate(msgs: list[tuple], engine: str) -> tuple[int, int, int]:
    """msgs: (call, chars, is_assistant, evict_call or None, stub_chars), in prompt order.
    Returns (prefill_chars, context_integral, eviction_events). With stub 0 and engine "prefix",
    "prefix_lazy" (theta 0.25) or "splice" this is evict.simulate's evict_now, lazy_25 and splice."""
    n_calls = sum(1 for m in msgs if m[2])
    m = len(msgs)
    size = [x[1] for x in msgs]                 # current size: full, then stub once dropped
    dropped = [False] * m
    arrived = 0
    L = prefill = ctx = events = 0
    theta = 0.25 if engine == "prefix_lazy" else 0.0
    order = sorted((i for i in range(m) if msgs[i][3] is not None), key=lambda i: msgs[i][3])
    sp = 0
    pending: list[int] = []
    pending_chars = 0
    for c in range(n_calls):
        new = 0
        while arrived < m and msgs[arrived][0] <= c:
            L += msgs[arrived][1]
            if not msgs[arrived][2]:
                new += msgs[arrived][1]
            arrived += 1
        while sp < len(order) and msgs[order[sp]][3] <= c:
            i = order[sp]; sp += 1
            if i < arrived and not dropped[i]:
                pending.append(i); pending_chars += msgs[i][1] - msgs[i][4]
        if pending and pending_chars >= theta * L:
            first = min(pending)
            for i in pending:
                dropped[i] = True
                L -= msgs[i][1] - msgs[i][4]
                size[i] = msgs[i][4]
            if engine == "splice":
                new += sum(msgs[i][4] for i in pending)
            else:
                new += sum(msgs[i][4] for i in pending if i == first)
                for i in range(first + 1, arrived):
                    if msgs[i][0] < c or msgs[i][2] or dropped[i]:
                        new += size[i]
            events += 1
            pending, pending_chars = [], 0
        prefill += new
        ctx += L
    return prefill, ctx, events


def append_only(msgs: list[tuple]) -> tuple[int, int]:
    return simulate([(a, b, c, None, 0) for a, b, c, *_ in msgs], "splice")[:2]


def view_signals(views: list[dict]) -> dict[tuple, dict]:
    """(call, sha) -> {'edit', 'edit_file', 'dup'} drop calls for one run's views."""
    by_file = defaultdict(list)
    for v in views:
        by_file[v["file"]].append(v)
    out = {}
    for vs in by_file.values():
        vs.sort(key=lambda v: v["call"])
        for i, a in enumerate(vs):
            dup = None
            for b in vs[i + 1:]:
                same_lines = a["line_lo"] is not None and b["line_lo"] is not None and \
                    b["line_lo"] <= a["line_lo"] and b["line_hi"] >= a["line_hi"]
                if b["call"] > a["call"] and (b["sha"] == a["sha"] or same_lines):
                    dup = b["call"]
                    break
            out[(a["call"], a["sha"])] = {"edit": a["stale_window_call"], "edit_file": a["stale_file_call"], "dup": dup}
    return out


def _min(*xs):
    xs = [x for x in xs if x is not None]
    return min(xs) if xs else None


def plan(rows: list[tuple], sig: dict[tuple, dict], policy: str) -> list[tuple]:
    """rows: (call, kind, chars, sha) in prompt order -> msgs for simulate()."""
    msgs = []
    for call, kind, chars, sha in rows:
        s = sig.get((call, sha)) if kind == "observation" else None
        ev, stub = None, VIEW_STUB
        if policy.startswith("age"):
            if kind == "observation" and chars > AGE_MIN_CHARS:
                ev, stub = call + int(policy[3:]), AGE_STUB
        elif policy.startswith("semantic"):
            if s is not None:
                ev = _min(s["edit_file"], s["dup"])
            elif kind == "observation" and chars > AGE_MIN_CHARS:
                ev, stub = call + int(policy[8:]), AGE_STUB
        elif s is not None:
            ev = {"edit": s["edit"], "edit_file": s["edit_file"], "dup": s["dup"],
                  "composite_file": _min(s["edit_file"], s["dup"])}[policy]
        if ev is not None and chars <= stub:
            ev = None
        msgs.append((call, chars, kind == "assistant", ev, stub))
    return msgs


def classify(rows, sig, msgs, n_calls) -> dict:
    """What a policy dropped, by what it was when dropped."""
    d = dict(stale=0, dup=0, live=0, live_other=0, blind_first_edits=0, views_dropped=0)
    for (call, kind, chars, sha), m in zip(rows, msgs):
        ev = m[3]
        if ev is None or ev >= n_calls:
            continue
        gone = chars - m[4]
        s = sig.get((call, sha)) if kind == "observation" else None
        if s is None:
            d["live_other"] += gone
            continue
        d["views_dropped"] += 1
        if s["edit_file"] is not None and s["edit_file"] <= ev:
            d["stale"] += gone
        elif s["dup"] is not None and s["dup"] <= ev:
            d["dup"] += gone
        else:
            d["live"] += gone
            if s["edit_file"] is not None and s["edit_file"] > ev:
                d["blind_first_edits"] += 1
    return d


def run_part(chunk_path: str, view_path: str) -> dict:
    vt = pq.read_table(view_path, columns=["run", "file", "call", "sha", "line_lo", "line_hi", "stale_file_call",
                                           "stale_window_call", "scratch"]).to_pylist()
    views = defaultdict(list)
    for v in vt:
        if not v["scratch"]:
            views[v["run"]].append(v)
    t = pq.read_table(chunk_path, columns=["run", "idx", "call", "kind", "chars", "sha"]).sort_by(
        [("run", "ascending"), ("idx", "ascending")])
    cols = [t.column(c).to_pylist() for c in ("run", "call", "kind", "chars", "sha")]
    tot = {"runs": 0, "append": [0, 0], "views": 0, "first_edits_of_viewed_files": 0,
           "sim": {p: {e: [0, 0, 0] for e in ENGINES} for p in POLICIES},
           "dropped": {p: dict(stale=0, dup=0, live=0, live_other=0, blind_first_edits=0, views_dropped=0) for p in POLICIES}}

    def flush(run, rows):
        if not rows:
            return
        sig = view_signals(views.get(run, []))
        n_calls = sum(1 for r in rows if r[1] == "assistant")
        tot["runs"] += 1
        tot["views"] += len(sig)
        tot["first_edits_of_viewed_files"] += sum(1 for s in sig.values() if s["edit_file"] is not None)
        a = append_only([(c, ch, k == "assistant") for c, k, ch, _ in rows])
        tot["append"][0] += a[0]; tot["append"][1] += a[1]
        for p in POLICIES:
            msgs = plan(rows, sig, p)
            for e in ENGINES:
                r = simulate(msgs, e)
                for k in range(3):
                    tot["sim"][p][e][k] += r[k]
            for k, v in classify(rows, sig, msgs, n_calls).items():
                tot["dropped"][p][k] += v

    cur, rows = None, []
    for run, call, kind, chars, sha in zip(*cols):
        if run != cur:
            flush(cur, rows)
            cur, rows = run, []
        rows.append((call, kind, chars, sha))
    flush(cur, rows)
    return tot


def merge(parts: list[dict]) -> dict:
    runs = sum(p["runs"] for p in parts)
    ap = [sum(p["append"][k] for p in parts) for k in range(2)]
    out = {"runs": runs, "append_prefill_chars": ap[0], "append_context_integral": ap[1],
           "views": sum(p["views"] for p in parts),
           "first_edits_of_viewed_files": sum(p["first_edits_of_viewed_files"] for p in parts), "policies": {}}
    for pol in POLICIES:
        d = {k: sum(p["dropped"][pol][k] for p in parts) for k in ("stale", "dup", "live", "live_other", "blind_first_edits", "views_dropped")}
        gone = d["stale"] + d["dup"] + d["live"] + d["live_other"]
        row = {"dropped_chars": gone,
               "dropped_stale_pct": round(100 * d["stale"] / gone, 1) if gone else None,
               "dropped_dup_pct": round(100 * d["dup"] / gone, 1) if gone else None,
               "dropped_live_view_pct": round(100 * d["live"] / gone, 1) if gone else None,
               "dropped_other_output_pct": round(100 * d["live_other"] / gone, 1) if gone else None,
               "views_dropped": d["views_dropped"], "blind_first_edits": d["blind_first_edits"],
               "blind_first_edits_pct_of_first_edits": round(100 * d["blind_first_edits"] / max(out["first_edits_of_viewed_files"], 1), 1)}
        for e in ENGINES:
            s = [sum(p["sim"][pol][e][k] for p in parts) for k in range(3)]
            row[e] = {"prefill_vs_append": round(s[0] / ap[0], 4), "context_vs_append": round(s[1] / ap[1], 4),
                      "evictions_per_run": round(s[2] / runs, 2)}
        out["policies"][pol] = row
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("silver", help="directory with chunks/ and views/")
    ap.add_argument("--parts", required=True, help="directory for per-file results (resumable)")
    ap.add_argument("--budget-s", type=float, default=0, help="stop starting new files after this many seconds")
    ap.add_argument("--merge", help="write the merged result here and print it")
    a = ap.parse_args()
    os.makedirs(a.parts, exist_ok=True)
    chunks = sorted(glob.glob(f"{a.silver}/chunks/*.parquet"))
    t0 = time.time()
    if not a.merge:
        for c in chunks:
            name = os.path.basename(c)
            out = f"{a.parts}/{name}.json"
            if os.path.exists(out):
                continue
            if a.budget_s and time.time() - t0 > a.budget_s:
                break
            json.dump(run_part(c, f"{a.silver}/views/{name}"), open(out + ".tmp", "w"))
            os.replace(out + ".tmp", out)
    done = sorted(glob.glob(f"{a.parts}/*.json"))
    print(f"{len(done)}/{len(chunks)} files done")
    if a.merge and done:
        res = merge([json.load(open(f)) for f in done])
        res["files"] = f"{len(done)}/{len(chunks)}"
        open(a.merge, "w").write(json.dumps(res, indent=2) + "\n")
        print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
