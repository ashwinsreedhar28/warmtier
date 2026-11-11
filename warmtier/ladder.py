"""How much cross-run reuse does each sharing scheme see?

    python -m warmtier.ladder 'data/silver/nebius/chunks/*.parquet' --out results/ladder_nebius.json

Three schemes, each scored as "extra copies it would not have to prefill
again" over all unique characters in the dataset:

  prefix_chain      a message is reusable only if every message before it in
                    the prompt is identical too. This is what hash-chained
                    block caching (vLLM prefix cache, SGLang radix tree) can
                    reuse across runs.
  content_addressed a message is reusable if the same text appears in another
                    run at any position. An upper bound for position-free KV
                    reuse at message granularity; reusing KV out of position
                    is not free (cross-attention to different earlier tokens
                    has to be repaired).
File-identity reuse (same file, different bytes) is in gold.py.

Message granularity only: two messages that differ by one character count as
different under both schemes.
"""
from __future__ import annotations

import argparse
import glob
import json

import pyarrow.parquet as pq

SHIFT = 1 << 34


def compute(paths: list[str]) -> dict:
    node: dict[int, int] = {}      # prefix-chain hash -> count*SHIFT + chars
    msg: dict[int, int] = {}       # message hash      -> runs*SHIFT + chars
    total = 0
    runs = 0
    for p in paths:
        t = pq.read_table(p, columns=["run", "idx", "chars", "sha"]).sort_by([("run", "ascending"), ("idx", "ascending")])
        run_col, chars_col, sha_col = (t.column(c).to_pylist() for c in ("run", "chars", "sha"))
        prev_run, chain, seen = None, 0, set()
        for r, c, s in zip(run_col, chars_col, sha_col):
            if r != prev_run:
                prev_run, chain, seen = r, 0, set()
                runs += 1
            h = int(s, 16)
            chain = hash((chain, h))
            total += c
            node[chain] = node.get(chain, c) + SHIFT
            if h not in seen:
                seen.add(h)
                msg[h] = msg.get(h, c) + SHIFT
    def extra(d):
        return sum((v // SHIFT - 1) * (v % SHIFT) for v in d.values())
    pc, ca = extra(node), extra(msg)
    return {"runs": runs, "unique_chars": total,
            "prefix_chain_pct": round(100 * pc / total, 2),
            "content_addressed_pct": round(100 * ca / total, 2),
            "content_addressed_over_prefix_chain": round(ca / pc, 2) if pc else None}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("chunks_glob")
    ap.add_argument("--out")
    a = ap.parse_args()
    res = compute(sorted(glob.glob(a.chunks_glob)))
    text = json.dumps(res, indent=2)
    if a.out:
        open(a.out, "w").write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
