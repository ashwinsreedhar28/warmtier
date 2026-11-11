"""Characters per token by message kind, to restate the trace shares in tokens.

    python -m warmtier.tokens --sweagent 'data/nebius/**/*.parquet' \\
        --claudecode 'data/open-swe-traces/data/claudecode/**/*.parquet' --model Qwen/Qwen3-4B --runs 1500

The trace statistics are in characters. Code, prose and tool output tokenise at
different rates, so a share of characters is not the same share of tokens.
This samples runs, tokenises every message with one model's tokenizer and
reports characters and tokens per kind of message. Shares can then be
re-weighted: tokens_k = chars_k / (chars per token)_k.
"""
from __future__ import annotations

import argparse
import glob
import json
import random
from collections import Counter

from .ledger import FileLedger
from .sweagent import FILE_HDR, VIEW_CMDS, _command


def sweagent_kinds(traj: list[dict]):
    last, first_user = "", True
    for m in traj:
        text = m.get("text") or m.get("system_prompt") or ""
        if m["role"] == "ai":
            last = _command(text)
            yield "assistant", text
        elif m["role"] == "system":
            yield "system", text
        elif first_user:
            first_user = False
            yield "task", text
        else:
            word = last.split()[0] if last else ""
            if FILE_HDR.search(text[:400]) and word in VIEW_CMDS:
                if word == "edit":
                    yield ("edit_echo" if "File updated" in text else "edit_rejected"), text
                elif word == "create":
                    yield "create", text
                else:
                    yield "read", text
            else:
                yield "other_observation", text


def claudecode_kinds(msgs: list[dict]):
    views = {v.index for v in FileLedger("claudecode").scan(msgs)}
    seen_user = False
    for i, m in enumerate(msgs):
        role = m.get("role")
        text = m.get("content") or ""
        if role == "assistant":
            calls = "".join(json.dumps(tc.get("function", {})) for tc in (m.get("tool_calls") or []))
            yield "assistant", text + (m.get("reasoning_content") or "") + calls
        elif role == "system":
            yield "system", text
        elif role == "user" and not seen_user:
            seen_user = True
            yield "task", text
        elif i in views:
            yield "read", text
        else:
            yield "other_observation", text


def measure(paths: list[str], column: str, kinds, tok, n_runs: int, seed: int) -> dict:
    import pyarrow.parquet as pq
    rng = random.Random(seed)
    paths = paths[:]
    rng.shuffle(paths)
    chars, toks, runs = Counter(), Counter(), 0
    per_file = max(1, -(-n_runs // len(paths)))
    for p in paths:
        pf = pq.ParquetFile(p)
        took = 0
        for g in rng.sample(range(pf.metadata.num_row_groups), pf.metadata.num_row_groups):
            rows = pf.read_row_group(g, columns=[column]).to_pylist()
            rng.shuffle(rows)
            for row in rows:
                items = [(k, t) for k, t in kinds(row[column]) if t]
                enc = tok([t for _, t in items], add_special_tokens=False)["input_ids"]
                for (k, t), ids in zip(items, enc):
                    chars[k] += len(t)
                    toks[k] += len(ids)
                runs += 1
                took += 1
                if took >= per_file or runs >= n_runs:
                    break
            if took >= per_file or runs >= n_runs:
                break
        if runs >= n_runs:
            break
    return {"runs": runs, "kinds": {k: {"chars": chars[k], "tokens": toks[k], "chars_per_token": round(chars[k] / toks[k], 3)}
                                    for k in sorted(chars)},
            "all": {"chars": sum(chars.values()), "tokens": sum(toks.values()),
                    "chars_per_token": round(sum(chars.values()) / sum(toks.values()), 3)}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sweagent")
    ap.add_argument("--claudecode")
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--runs", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    res = {"model": a.model, "seed": a.seed}
    if a.sweagent:
        res["sweagent"] = measure(sorted(glob.glob(a.sweagent, recursive=True)), "trajectory", sweagent_kinds, tok, a.runs, a.seed)
    if a.claudecode:
        res["claudecode"] = measure(sorted(glob.glob(a.claudecode, recursive=True)), "messages", claudecode_kinds, tok, a.runs, a.seed)
    text = json.dumps(res, indent=2)
    if a.out:
        open(a.out, "w").write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
