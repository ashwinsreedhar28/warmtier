"""Reuse among runs that work on the same task: a stand-in for an agent team.

    python -m warmtier.team 'data/silver/nebius/chunks/*.parquet' \
        'data/silver/nebius/views/*.parquet' --out results/team_nebius.json

A "team" here is k runs of the same task instance (same repo, same base
commit, same issue). These runs never saw each other, so this is a proxy: it
shows how much content k agents on one checkout read in common, not how a
coordinated team behaves.

For each team size k, teams are the first k runs (random order, fixed seed) of
every instance that has at least k runs. Reuse is "characters a shared store
would not prefill again" over all characters the team prefilled once each:

  header            system prompt, tool list and task text (shared by
                    construction)
  body              everything after the header
  prefix_chain      reusable only if every earlier message matches too
  content_addressed reusable if the same message text appears in another
                    run of the team at any position
  view_content      file views only: same bytes shown in another run
  view_file         file views only: same file shown in another run (bytes
                    may differ); an upper bound for file-level reuse
"""
from __future__ import annotations

import argparse
import glob
import json
import random
from collections import defaultdict

import pyarrow.parquet as pq

SIZES = (2, 4, 8, 16)
HEADER = ("system", "tools", "task")


def load_runs(paths: list[str]) -> dict[str, dict[int, list[tuple[bool, int, int]]]]:
    """instance -> run -> [(is_header, chars, sha)] in prompt order."""
    out: dict[str, dict[int, list]] = defaultdict(lambda: defaultdict(list))
    for p in paths:
        t = pq.read_table(p, columns=["run", "instance", "idx", "kind", "chars", "sha"])
        t = t.sort_by([("run", "ascending"), ("idx", "ascending")])
        cols = [t.column(c).to_pylist() for c in ("run", "instance", "kind", "chars", "sha")]
        for r, inst, kind, c, s in zip(*cols):
            out[inst][r].append((kind in HEADER, c, int(s, 16)))
    return out


def load_views(paths: list[str]) -> dict[int, list[tuple[str, int, int]]]:
    """run -> [(file, chars, sha)]; scratch files (created by the run) dropped."""
    out: dict[int, list] = defaultdict(list)
    for p in paths:
        t = pq.read_table(p)
        has_file = "file" in t.column_names
        cols = [t.column(c).to_pylist() for c in ("run", "chars", "sha", "scratch")]
        files = t.column("file").to_pylist() if has_file else [None] * t.num_rows
        for r, c, s, scratch, f in zip(*cols, files):
            if not scratch:
                out[r].append((f, c, int(s, 16)))
    return out


def team_stats(runs: list[list[tuple[bool, int, int]]], views: list[list[tuple[str, int, int]]]) -> dict:
    """Reuse inside one team, in the order the runs are given.

    Only reuse a teammate provides is counted: text a run repeats from its own
    earlier context counts once, on first sight, and only if an earlier run of
    the team had it."""
    tot = {"header": 0, "body": 0}
    chain_seen: set[int] = set()
    msg_seen: set[int] = set()
    pc = {"header": 0, "body": 0}
    ca = {"header": 0, "body": 0}
    for msgs in runs:
        chain = 0
        new_chain, new_msg = [], set()
        for is_header, c, h in msgs:
            part = "header" if is_header else "body"
            tot[part] += c
            chain = hash((chain, h))
            if chain in chain_seen:
                pc[part] += c
            new_chain.append(chain)
            if h in msg_seen and h not in new_msg:
                ca[part] += c
            new_msg.add(h)
        chain_seen.update(new_chain)
        msg_seen |= new_msg
    v_tot = v_content = v_file = 0
    sha_seen: set[int] = set()
    file_seen: set[str] = set()
    for vs in views:
        mine: set[int] = set()
        for f, c, h in vs:
            v_tot += c
            if h in sha_seen and h not in mine:
                v_content += c
            if f is not None and f in file_seen and h not in mine:
                v_file += c
            mine.add(h)
        sha_seen |= {h for _, _, h in vs}
        file_seen |= {f for f, _, _ in vs if f is not None}
    return {"tot": tot, "pc": pc, "ca": ca, "v_tot": v_tot, "v_content": v_content,
            "v_file": v_file if file_seen else None}


def _pct(a: int, b: int) -> float | None:
    return round(100 * a / b, 2) if b else None


def compute(chunk_paths: list[str], view_paths: list[str], seed: int = 0, sizes=SIZES) -> dict:
    by_inst = load_runs(chunk_paths)
    views = load_views(view_paths)
    rng = random.Random(seed)
    order = {inst: rng.sample(sorted(rs), len(rs)) for inst, rs in sorted(by_inst.items())}
    res = {"instances": len(by_inst), "runs": sum(len(r) for r in by_inst.values()), "seed": seed, "sizes": {}}
    for k in sizes:
        agg = defaultdict(int)
        teams = 0
        for inst, ids in order.items():
            if len(ids) < k:
                continue
            ids = ids[:k]
            s = team_stats([by_inst[inst][r] for r in ids], [views.get(r, []) for r in ids])
            teams += 1
            for part in ("header", "body"):
                agg["tot_" + part] += s["tot"][part]
                agg["pc_" + part] += s["pc"][part]
                agg["ca_" + part] += s["ca"][part]
            for key in ("v_tot", "v_content"):
                agg[key] += s[key]
            if s["v_file"] is not None:
                agg["v_file"] += s["v_file"]
                agg["has_file"] = 1
        if not teams:
            continue
        total = agg["tot_header"] + agg["tot_body"]
        res["sizes"][str(k)] = {
            "teams": teams,
            "header_share_of_prefill_pct": _pct(agg["tot_header"], total),
            "all_prefix_chain_pct": _pct(agg["pc_header"] + agg["pc_body"], total),
            "all_content_addressed_pct": _pct(agg["ca_header"] + agg["ca_body"], total),
            "body_prefix_chain_pct": _pct(agg["pc_body"], agg["tot_body"]),
            "body_content_addressed_pct": _pct(agg["ca_body"], agg["tot_body"]),
            "view_share_of_prefill_pct": _pct(agg["v_tot"], total),
            "view_content_reuse_pct": _pct(agg["v_content"], agg["v_tot"]),
            "view_file_reuse_pct": _pct(agg["v_file"], agg["v_tot"]) if agg["has_file"] else None,
        }
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("chunks_glob")
    ap.add_argument("views_glob")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out")
    a = ap.parse_args()
    res = compute(sorted(glob.glob(a.chunks_glob)), sorted(glob.glob(a.views_glob)), a.seed)
    text = json.dumps(res, indent=2)
    if a.out:
        open(a.out, "w").write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
