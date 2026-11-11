"""Why does the writer adapter score the real next turn above full prefill?

    python -m warmtier.writer_diag '<traces glob>' --adapter out/writer_qwen3_4b.pt --n 60 --out diag.json

Audit tool for results/gate3_qwen3_4b_writer.json. For each sample it scores
the real next turn S under:
  full          plain model, full attention
  iso           file isolated from the prefix, no adapter (plain reuse)
  writer        file isolated, adapter on the file's tokens (the reported condition)
  adapter_full  adapter on the file's tokens, full attention (no isolation)
and reports things that separate "the adapter recovers lost context" from
"the adapter sharpens or restyles the model":
  nll, entropy, probability of the top token
  nll split by whether the real token is full's top choice
  nll by position in the turn
  the temperature that would make full's own predictions best fit the real
  tokens, and how much nll that alone buys
"""
from __future__ import annotations

import argparse
import glob
import json
import math

import torch

from .gate import allowed_matrix, hidden, parts, samples, _set_gate

CONDS = ("full", "iso", "writer", "adapter_full")
BUCKETS = ((0, 8), (8, 32), (32, 10_000))
TEMPS = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2)


@torch.no_grad()
def logps(model, P, F, S) -> dict[str, torch.Tensor]:
    device = next(model.parameters()).device
    ids = torch.tensor(P + F + S, device=device)
    nP, nF, nS = len(P), len(F), len(S)
    s0 = nP + nF
    _, head = parts(model)
    out = {}
    for c in CONDS:
        mask = allowed_matrix("iso" if c in ("iso", "writer") else "full", nP, nF, nS)
        gate = (nP, s0) if c in ("writer", "adapter_full") else None
        h = hidden(model, ids, mask, gate_rows=gate)[s0 - 1:-1]
        _set_gate(model, None, 0, device)
        out[c] = torch.log_softmax(head(h).float(), dim=-1)
    return out


def sample_stats(lp: dict[str, torch.Tensor], S: list[int]) -> dict:
    tgt = torch.tensor(S, device=lp["full"].device)
    full = lp["full"]
    top_full = full.argmax(-1)
    hit = top_full == tgt
    res = {"n": len(S), "n_hit": int(hit.sum())}
    for c, x in lp.items():
        nll = -x.gather(1, tgt[:, None])[:, 0]
        ent = -(x.exp() * x).sum(-1)
        r = {"nll": float(nll.mean()), "entropy": float(ent.mean()), "top_prob": float(x.max(-1).values.exp().mean()),
             "logp_of_full_top": float(x.gather(1, top_full[:, None]).mean()),
             "nll_sum_hit": float(nll[hit].sum()), "nll_sum_miss": float(nll[~hit].sum()),
             "kl_from_full": float((full.exp() * (full - x)).sum(-1).mean())}
        for a, b in BUCKETS:
            seg = nll[a:b]
            r[f"nll_sum_{a}"] = float(seg.sum()); r[f"n_{a}"] = int(seg.numel())
        res[c] = r
    logits = full                               # log-probs work as logits for re-tempering
    res["temp_nll_sum"] = {str(t): float(-torch.log_softmax(logits / t, -1).gather(1, tgt[:, None]).sum()) for t in TEMPS}
    return res


def aggregate(rows: list[dict]) -> dict:
    n = sum(r["n"] for r in rows)
    n_hit = sum(r["n_hit"] for r in rows)
    out = {"tokens": n, "real_token_is_full_top": round(n_hit / n, 4)}
    for c in CONDS:
        g = lambda k: sum(r[c][k] for r in rows)
        out[c] = {"nll_per_sample_mean": round(sum(r[c]["nll"] for r in rows) / len(rows), 4),
                  "nll_per_token": round(sum(r[c]["nll"] * r["n"] for r in rows) / n, 4),
                  "entropy": round(sum(r[c]["entropy"] * r["n"] for r in rows) / n, 4),
                  "top_prob": round(sum(r[c]["top_prob"] * r["n"] for r in rows) / n, 4),
                  "logp_of_full_top": round(sum(r[c]["logp_of_full_top"] * r["n"] for r in rows) / n, 4),
                  "kl_from_full": round(sum(r[c]["kl_from_full"] for r in rows) / len(rows), 4),
                  "nll_when_real_is_full_top": round(g("nll_sum_hit") / max(n_hit, 1), 4),
                  "nll_when_real_is_not": round(g("nll_sum_miss") / max(n - n_hit, 1), 4)}
        for a, _ in BUCKETS:
            out[c][f"nll_pos_{a}"] = round(g(f"nll_sum_{a}") / max(sum(r[c][f"n_{a}"] for r in rows), 1), 4)
    out["full_at_temperature_nll"] = {t: round(sum(r["temp_nll_sum"][t] for r in rows) / n, 4) for t in map(str, TEMPS)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("parquet_glob")
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-prefix", type=int, default=3000)
    ap.add_argument("--max-file", type=int, default=2000)
    ap.add_argument("--max-suffix", type=int, default=200)
    ap.add_argument("--out")
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from .writer import load_adapter
    tok = AutoTokenizer.from_pretrained(a.model)
    try:
        model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float16)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float16)
    model = model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
    load_adapter(model, a.adapter)
    rows = []
    for i, s in enumerate(samples(sorted(glob.glob(a.parquet_glob, recursive=True)), tok, a.n, a.seed,
                                  a.max_prefix, a.max_file, a.max_suffix), 1):
        st = sample_stats(logps(model, s["P"], s["F"], s["S"]), s["S"])
        st["instance"] = s["instance"]
        rows.append(st)
        print(f"[{i}/{a.n}] nll full {st['full']['nll']:.3f} iso {st['iso']['nll']:.3f} writer {st['writer']['nll']:.3f} "
              f"adapter_full {st['adapter_full']['nll']:.3f}", flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    res = {"model": a.model, "n": len(rows), "glob": a.parquet_glob, "aggregate": aggregate(rows), "rows": rows}
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res["aggregate"], indent=1))


if __name__ == "__main__":
    main()
