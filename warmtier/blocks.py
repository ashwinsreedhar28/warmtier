"""Would a store keyed on file line ranges catch what whole-view hashing misses?

    python -m warmtier.blocks 'data/nebius/**/*.parquet' --out results/blocks_nebius.json

team.py found that SWE-agent teammates open the same files far more often
than they see the same bytes (62% vs 24% of view text at 16 agents). SWE-agent
shows a file as a 100-line window, so two agents rarely land on the same
window. This measures reuse when the unit is smaller than a view:

  view     the whole window's numbered lines, as one unit
  B lines  blocks of B consecutive lines aligned to line numbers
           (lines 1..B, B+1..2B, ...). A block counts only if the view shows
           all of it (or all that exists up to the end of the file).
  1 line   single lines: the upper bound for any line-aligned scheme

A unit is reusable when an earlier teammate saw the same file path, the same
position and the same text. Only the first sight in a run counts, and only
text a teammate (not the run itself) supplied. The denominator is all numbered
line text in the run's file views. Files a run created itself are skipped.
Teams are the first k runs of a task in a fixed random order, as in team.py.

Raw SWE-agent trajectories only (the line numbers come from its viewer).
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import pickle
import random
import re
import zlib
from collections import defaultdict
from multiprocessing import Pool

FILE_HDR = re.compile(r"\[File: (.+?) \((\d+) lines total\)\]")
LINE = re.compile(r"^(\d+):(.*)$", re.M)
BLOCK = re.compile(r"```[a-z]*\n?(.*?)```", re.S)
VIEW_CMDS = {"open", "goto", "scroll_down", "scroll_up", "edit", "create"}
SIZES = (1, 10, 25, 50, 100)
TEAM = (2, 4, 8, 16)


def _h(*parts) -> int:
    return int.from_bytes(hashlib.blake2b("\x00".join(map(str, parts)).encode("utf-8", "replace"), digest_size=8).digest(), "big")


def _command(ai_text: str) -> str:
    m = BLOCK.findall(ai_text or "")
    return (m[-1] if m else "").strip()


def view_units(path: str, total: int, lines: list[tuple[int, str]]) -> dict[str, list[tuple[int, int]]]:
    """(key, chars) for each unit of one view, per granularity."""
    by_no = dict(lines)
    out = {"view": [(_h("v", path, lines[0][0], *[t for _, t in lines]), sum(len(t) for _, t in lines))]}
    for b in SIZES:
        units = []
        for blk in sorted({(n - 1) // b for n in by_no}):
            lo, hi = blk * b + 1, min((blk + 1) * b, total or 10**9)
            if all(n in by_no for n in range(lo, hi + 1)):
                text = [by_no[n] for n in range(lo, hi + 1)]
                units.append((_h(b, path, lo, *text), sum(map(len, text))))
        out[str(b)] = units
    return out


def run_units(traj: list[dict]):
    """Per granularity: [(key, chars)] in order of first sight; plus total line chars."""
    seen = {k: set() for k in ["view", *map(str, SIZES)]}
    units = {k: [] for k in seen}
    total = 0
    created, last = set(), ""
    for m in traj:
        if m["role"] == "ai":
            last = _command(m.get("text") or "")
            continue
        text = m.get("text") or ""
        word = last.split()[0] if last else ""
        h = FILE_HDR.search(text[:400])
        if not h or word not in VIEW_CMDS:
            continue
        path = h.group(1)
        if word == "create":
            created.add(path)
        if path in created:
            continue
        lines = [(int(n), t) for n, t in LINE.findall(text)]
        if not lines:
            continue
        total += sum(len(t) for _, t in lines)
        for g, us in view_units(path, int(h.group(2)), lines).items():
            for key, chars in us:
                if key not in seen[g]:
                    seen[g].add(key)
                    units[g].append((key, chars))
    return units, total


def process_file(args):
    path, keep_mod, out = args
    import pyarrow.parquet as pq
    res = []
    pf = pq.ParquetFile(path)
    for g in range(pf.metadata.num_row_groups):
        for row in pf.read_row_group(g, columns=["instance_id", "trajectory"]).to_pylist():
            if zlib.crc32(row["instance_id"].encode()) % keep_mod:
                continue
            units, total = run_units(row["trajectory"])
            if total:
                res.append((row["instance_id"], units, total))
    pickle.dump(res, open(out, "wb"))
    return len(res)


def reduce(parts: list[str], seed: int = 0) -> dict:
    by_inst = defaultdict(list)
    for p in sorted(parts):
        for inst, units, total in pickle.load(open(p, "rb")):
            by_inst[inst].append((units, total))
    rng = random.Random(seed)
    res = {"instances": len(by_inst), "runs": sum(map(len, by_inst.values())), "seed": seed, "sizes": {}}
    grains = ["view", *map(str, SIZES)]
    for k in TEAM:
        tot = 0
        reuse = dict.fromkeys(grains, 0)
        covered = dict.fromkeys(grains, 0)
        teams = 0
        for inst in sorted(by_inst):
            runs = by_inst[inst]
            if len(runs) < k:
                continue
            team = rng.sample(runs, k)
            teams += 1
            pool = {g: set() for g in grains}
            for units, total in team:
                tot += total
                for g in grains:
                    covered[g] += sum(c for _, c in units[g])
                    reuse[g] += sum(c for key, c in units[g] if key in pool[g])
                for g in grains:
                    pool[g].update(key for key, _ in units[g])
        if not teams:
            continue
        res["sizes"][str(k)] = {"teams": teams, "view_line_chars": tot,
                                "reuse_pct": {g: round(100 * reuse[g] / tot, 2) for g in grains},
                                "text_in_whole_units_pct": {g: round(100 * covered[g] / tot, 2) for g in grains}}
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("parquet_glob")
    ap.add_argument("--keep-mod", type=int, default=3, help="keep instances whose id hash is 0 mod this (1 = all)")
    ap.add_argument("--tmp", default="/tmp/warmtier_blocks")
    ap.add_argument("--procs", type=int, default=16)
    ap.add_argument("--out")
    a = ap.parse_args()
    import os
    os.makedirs(a.tmp, exist_ok=True)
    paths = sorted(glob.glob(a.parquet_glob, recursive=True))
    jobs = [(p, a.keep_mod, f"{a.tmp}/{i:05d}.pkl") for i, p in enumerate(paths)]
    with Pool(a.procs) as pool:
        n = sum(pool.imap_unordered(process_file, jobs))
    print(f"{len(paths)} files, {n} runs kept", flush=True)
    res = reduce([j[2] for j in jobs])
    res["keep_mod"] = a.keep_mod
    text = json.dumps(res, indent=2)
    if a.out:
        open(a.out, "w").write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
