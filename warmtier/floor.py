"""Noise floor for the gate metrics: how far does full prefill move under a trivial change?

    python -m warmtier.floor '<traces glob>' --model Qwen/Qwen3-4B --n 60 --gen 64 --out results/floor_qwen3_4b.json

gate.py compares each reuse scheme with full prefill by KL and by whether the
greedily generated next call is identical. "Identical" is fragile: one changed
token anywhere in 64 breaks it. This script measures how often it breaks when
nothing about reuse is involved, on the same samples and seed as the gate runs:

  pre_nl   one newline token added in front of the prefix (every position shifts by 1)
  drop8    the prefix starts 8 tokens later (3,000-token prefix; 0.3% of it removed)
  drop64   the prefix starts 64 tokens later (2% of it removed)

Each is still full prefill of everything shown. Their distance from the
unperturbed run is the floor that reuse schemes should be read against.
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics
import time

import torch

from .gate import allowed_matrix, call_name, generate, hidden, parts, samples, self_check

CONDS = ("pre_nl", "drop8", "drop64")


def variant(c: str, P: list[int], nl: int) -> list[int]:
    return {"pre_nl": [nl] + P, "drop8": P[8:], "drop64": P[64:]}[c]


@torch.no_grad()
def logp(model, P, F, S) -> torch.Tensor:
    _, head = parts(model)
    device = next(model.parameters()).device
    ids = torch.tensor(P + F + S, device=device)
    h = hidden(model, ids, allowed_matrix("full", len(P), len(F), len(S)))[len(P) + len(F) - 1:-1]
    return torch.log_softmax(head(h).float(), dim=-1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("parquet_glob")
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-prefix", type=int, default=3000)
    ap.add_argument("--max-file", type=int, default=2000)
    ap.add_argument("--max-suffix", type=int, default=200)
    ap.add_argument("--gen", type=int, default=64)
    ap.add_argument("--out")
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    try:
        model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float16)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float16)
    model = model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
    self_check(model, len(tok))
    nl = tok("\n", add_special_tokens=False)["input_ids"][0]
    stop_ids = {i for i in (tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id) if isinstance(i, int) and i >= 0}
    norm = lambda t: t.split("<|im_end|>")[0].strip()
    rows, t0 = [], time.time()
    for i, s in enumerate(samples(sorted(glob.glob(a.parquet_glob, recursive=True)), tok, a.n, a.seed,
                                  a.max_prefix, a.max_file, a.max_suffix), 1):
        P, F, S, T = s["P"], s["F"], s["S"], s["T"]
        full = logp(model, P, F, S)
        row = {"instance": s["instance"], "nP": len(P), "m": {}, "g": {}}
        gen_full = norm(tok.decode(generate(model, P, F, T, "full", None, a.gen, stop_ids))) if s["ref"] else None
        for c in CONDS:
            P2 = variant(c, P, nl)
            x = logp(model, P2, F, S)
            row["m"][c] = {"kl": float((full.exp() * (full - x)).sum(-1).mean()),
                           "top1": float((x.argmax(-1) == full.argmax(-1)).float().mean())}
            if gen_full is not None:
                g = norm(tok.decode(generate(model, P2, F, T, "full", None, a.gen, stop_ids)))
                row["g"][c] = {"name_full": call_name(g) == call_name(gen_full), "exact_full": g == gen_full}
        rows.append(row)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        el = time.time() - t0
        print(f"[{i}/{a.n}] " + " ".join(f"{c} kl={row['m'][c]['kl']:.3f}" for c in CONDS) + f" | {el / i:.1f}s each", flush=True)
    agg = {}
    for c in CONDS:
        g = [r["g"][c] for r in rows if r["g"]]
        agg[c] = {"kl_mean": round(statistics.mean(r["m"][c]["kl"] for r in rows), 4),
                  "top1_mean": round(statistics.mean(r["m"][c]["top1"] for r in rows), 4),
                  "gen_n": len(g),
                  "name_full": round(sum(x["name_full"] for x in g) / max(len(g), 1), 4),
                  "exact_full": round(sum(x["exact_full"] for x in g) / max(len(g), 1), 4)}
    res = {"model": a.model, "n": len(rows), "seed": a.seed, "aggregate": agg, "rows": rows}
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(agg, indent=1))


if __name__ == "__main__":
    main()
