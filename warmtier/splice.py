"""Splice safety: can an engine drop a stale file view from cached KV in place?

    python -m warmtier.splice 'data/open-swe-traces/data/claudecode/**/*.parquet' \\
        --model Qwen/Qwen3-4B --n 100 --gen 64 --out results/splice_qwen3_4b.json

evict.py shows that dropping stale views only pays if nothing after them has
to be prefilled again. That needs the engine to cut a span out of the cached
KV and keep going. The catch: everything the model wrote or read after the
stale view was encoded while attending to it, so its KV still carries traces
of the dropped text. This script measures whether that matters.

Each sample is a real trace cut at the first model call after an edit made an
earlier file view stale (found with ledger.FileLedger, window level):
  P  everything before the stale view (truncated on the left)
  F  the stale view
  M  everything between the view and the call (includes the edit)
  S  the assistant turn the real agent produced next

Conditions:
  full        nothing dropped (append-only harness)
  evict_text  F replaced by the ledger's one-line stub and everything after it
              prefilled again. What a text-level harness does; the reference
              for "the view is gone"
  splice      KV for M is computed with F visible, then F is cut: nothing from
              the new call onward can attend to F. Positions keep their gap.
              Zero re-prefill
  mask_all    M cannot attend to F either. Same as evict_text without the stub
              and with a position gap; separates those two effects from the
              stale-KV effect

A mask reproduces the spliced cache exactly (tests/test_splice.py checks it
against cutting the KV tensors by hand).

Reported per condition:
  nll            mean negative log-likelihood of the real next turn
  kl_evict       mean KL(evict_text || condition) per token of S, in nats
  top1_evict     share of S positions with the same top token as evict_text
  with --gen N, the next tool call generated greedily: how often its name and
  the whole call match the real agent's call (name_ref, exact_ref), the call
  generated under evict_text (name_evict, exact_evict) and under full
  (name_full, exact_full)

The real next turn was produced by an agent that had the stale view in
context, so *_ref leans toward full. The splice question is answered by the
*_evict columns: splice is safe if it lands as close to evict_text as mask_all
does.
"""
from __future__ import annotations

import argparse
import glob
import json
import random
import statistics
import time

import torch

from .gate import _mask4d, assistant_text, call_name, chatml, parts, self_check
from .ledger import STUB, FileLedger

CONDITIONS = ("full", "evict_text", "splice", "mask_all")
OPEN = "<|im_start|>assistant\n"
CALL = '\n{"name": "'


def allowed_matrix(cond: str, nP: int, nF: int, nM: int, nR: int) -> torch.Tensor:
    """Rows/cols over P+F+M+R, where R is everything from the new call onward."""
    L = nP + nF + nM + nR
    a = torch.tril(torch.ones(L, L, dtype=torch.bool))
    f0, f1, r0 = nP, nP + nF, nP + nF + nM
    if cond == "splice":
        a[r0:, f0:f1] = False
    elif cond == "mask_all":
        a[f1:, f0:f1] = False
    elif cond != "full":
        raise ValueError(cond)
    return a


@torch.no_grad()
def logp_tail(model, ids: list[int], allowed: torch.Tensor | None, n_tail: int) -> torch.Tensor:
    """Log-probs for the last n_tail tokens of ids (each predicted from what precedes it)."""
    base, head = parts(model)
    device = next(model.parameters()).device
    x = torch.tensor(ids, device=device)
    kw = {} if allowed is None else {"attention_mask": _mask4d(model, allowed, device)}
    h = base(input_ids=x[None], **kw).last_hidden_state[0][-n_tail - 1:-1]
    return torch.log_softmax(head(h).float(), dim=-1)


@torch.no_grad()
def score(model, s: dict) -> dict:
    P, F, M, S, n_open = s["P"], s["F"], s["M"], s["S"], s["n_open"]
    L = s.get("L") or []                     # encoded after the splice; never sees F under splice
    n = len(S) - n_open                      # score the turn body, not the role header
    lp = {}
    for c in ("full", "splice", "mask_all"):
        lp[c] = logp_tail(model, P + F + M + L + S, allowed_matrix(c, len(P), len(F), len(M), len(L) + len(S)), n)
    lp["evict_text"] = logp_tail(model, P + s["stub"] + M + L + S, None, n)
    tgt = torch.tensor(S[n_open:], device=lp["full"].device)
    ref, full = lp["evict_text"], lp["full"]
    out = {}
    for c in CONDITIONS:
        x = lp[c]
        out[c] = {"nll": float(-x.gather(1, tgt[:, None]).mean()),
                  "kl_evict": float((ref.exp() * (ref - x)).sum(-1).mean()),
                  "top1_evict": float((x.argmax(-1) == ref.argmax(-1)).float().mean()),
                  "kl_full": float((full.exp() * (full - x)).sum(-1).mean())}
    return out


@torch.no_grad()
def generate(model, ids: list[int], allowed: torch.Tensor | None, block: tuple[int, int] | None,
             max_new: int, stop_ids: set[int]) -> list[int]:
    """Greedy decode after ids; new tokens cannot attend to columns block[0]:block[1]."""
    base, head = parts(model)
    device = next(model.parameters()).device
    dt = next(model.parameters()).dtype
    x = torch.tensor(ids, device=device)
    kw = {} if allowed is None else {"attention_mask": _mask4d(model, allowed, device)}
    out = base(input_ids=x[None], use_cache=True, **kw)
    pkv = out.past_key_values
    tok = int(head(out.last_hidden_state[0, -1]).argmax())
    row = torch.zeros(1, 1, 1, len(ids), dtype=dt, device=device)
    if block:
        row[..., block[0]:block[1]] = torch.finfo(dt).min
    gen = []
    for _ in range(max_new):
        if tok in stop_ids:
            break
        gen.append(tok)
        row = torch.cat([row, torch.zeros(1, 1, 1, 1, dtype=dt, device=device)], -1)
        out = base(input_ids=torch.tensor([[tok]], device=device), past_key_values=pkv, attention_mask=row, use_cache=True)
        pkv = out.past_key_values
        tok = int(head(out.last_hidden_state[0, -1]).argmax())
    return gen


def generate_all(model, s: dict, max_new: int, stop_ids: set[int]) -> dict[str, list[int]]:
    P, F, M = s["P"], s["F"], s["M"]
    T = (s.get("L") or []) + s["T"]
    f = (len(P), len(P) + len(F))
    out = {}
    for c in ("full", "splice", "mask_all"):
        a = allowed_matrix(c, len(P), len(F), len(M), len(T))
        out[c] = generate(model, P + F + M + T, a, None if c == "full" else f, max_new, stop_ids)
    out["evict_text"] = generate(model, P + s["stub"] + M + T, None, None, max_new, stop_ids)
    return out


@torch.no_grad()
def leak_check(model, vocab: int) -> tuple[float, float]:
    """Does text behind the mask still reach later tokens?

    Changes F and compares what follows under mask_all. Returns the relative
    change in the last hidden state (0 on an attention-only model: a masked
    span is invisible) and, as a scale, the change when F is not masked.
    On a hybrid the linear-attention layers ignore the mask, so the first
    number is above 0: the span stays folded into their state."""
    g = torch.Generator().manual_seed(0)
    device = next(model.parameters()).device
    base, _ = parts(model)
    r = lambda n: torch.randint(10, min(vocab, 5000), (n,), generator=g).tolist()
    P, F1, F2, M, S = r(24), r(40), r(40), r(20), r(12)

    def tail(F, cond):
        x = torch.tensor(P + F + M + S, device=device)
        a = allowed_matrix(cond, 24, 40, 20, 12)
        return base(input_ids=x[None], attention_mask=_mask4d(model, a, device)).last_hidden_state[0, -12:].float()

    masked = float((tail(F1, "mask_all") - tail(F2, "mask_all")).abs().max() / tail(F1, "mask_all").abs().max())
    seen = float((tail(F1, "full") - tail(F2, "full")).abs().max() / tail(F1, "full").abs().max())
    return masked, seen


def find_sample(row: dict, rng: random.Random, lag: int = 0):
    """One stale view plus a model call made after it went stale.

    lag=0 scores the first call after the edit (where the splice happens). lag=k scores the call
    k turns later: what the trace added in between ("later") is encoded after the splice and can
    never attend to the view under splice or mask_all."""
    msgs = row["messages"]
    stale = [v for v in FileLedger("claudecode").stale(msgs, level="window") if msgs[v.index].get("role") == "tool"]
    asst = [j for j, m in enumerate(msgs) if m["role"] == "assistant"]
    cands = []
    for v in stale:
        if v.stale_window + lag >= len(asst):
            continue
        j0 = asst[v.stale_window]             # first assistant turn after the edit result
        j = asst[v.stale_window + lag]        # the turn that is scored
        if j0 > v.index and (msgs[j].get("tool_calls") or []) and len(msgs[v.index].get("content") or "") > 800:
            cands.append((v, j0, j))
    if not cands:
        return None
    v, j0, j = rng.choice(cands)
    text = lambda m: chatml(m["role"], assistant_text(m, False) if m["role"] == "assistant" else (m.get("content") or ""))
    pre = "".join(text(m) for m in msgs[:v.index]) + "<|im_start|>tool\n"
    mid = "<|im_end|>\n" + "".join(text(m) for m in msgs[v.index + 1:j0])
    later = "".join(text(m) for m in msgs[j0:j])
    tc = msgs[j]["tool_calls"][0]["function"]
    ref = json.dumps({"name": tc["name"], "arguments": tc["arguments"]})
    return {"pre": pre, "file": msgs[v.index]["content"], "mid": mid, "later": later, "suf": assistant_text(msgs[j], True),
            "ref": ref, "path": v.path, "calls_between": sum(1 for k in asst if v.index < k < j0)}


def samples(paths: list[str], tok, n: int, seed: int, max_p: int, max_f: int, max_m: int, max_s: int,
            lag: int = 0, max_later: int = 3000):
    import pyarrow.parquet as pq
    enc = lambda t: tok(t, add_special_tokens=False)["input_ids"]
    rng = random.Random(seed)
    paths = paths[:]
    rng.shuffle(paths)
    got = 0
    per_file = max(1, -(-n // len(paths)))
    n_open = len(enc(OPEN))
    for p in paths:
        t = pq.ParquetFile(p)
        took = 0
        for g in range(t.metadata.num_row_groups):
            rows = t.read_row_group(g, columns=["instance_id", "messages"]).to_pylist()
            rng.shuffle(rows)
            for row in rows:
                s = find_sample(row, rng, lag)
                if not s:
                    continue
                F, M = enc(s["file"]), enc(s["mid"])
                if not 200 <= len(F) <= max_f or len(M) > max_m:
                    continue
                L = enc(s["later"]) if s["later"] else []
                if len(L) > max_later:
                    continue
                S = (enc(OPEN) + enc(s["suf"]))[:max_s]
                if len(S) - n_open < 16:
                    continue
                yield {"instance": row["instance_id"], "P": enc(s["pre"])[-max_p:], "F": F, "M": M, "L": L, "S": S,
                       "n_open": n_open, "T": enc(OPEN + CALL), "stub": enc(STUB.format(path=s["path"])),
                       "ref": s["ref"], "calls_between": s["calls_between"]}
                got += 1
                took += 1
                if got >= n:
                    return
                if took >= per_file:
                    break
            if took >= per_file:
                break


def gen_metrics(texts: dict[str, str], ref: str) -> dict:
    ref_tail = ref[len('{"name": "'):].strip()
    norm = lambda t: t.split("<|im_end|>")[0].strip()
    t = {c: norm(x) for c, x in texts.items()}
    out = {}
    for c, x in t.items():
        out[c] = {"name_ref": call_name(x) == call_name(ref_tail), "exact_ref": x == ref_tail}
        for other in ("evict", "full"):
            y = t["evict_text" if other == "evict" else "full"]
            out[c][f"name_{other}"] = call_name(x) == call_name(y)
            out[c][f"exact_{other}"] = x == y
    return out


def aggregate(rows: list[dict]) -> dict:
    agg = {}
    for c in CONDITIONS:
        agg[c] = {k + "_mean": round(statistics.mean(r["m"][c][k] for r in rows), 4)
                  for k in ("nll", "kl_evict", "top1_evict", "kl_full")}
        g = [r["g"][c] for r in rows if r.get("g")]
        if g:
            agg[c]["gen_n"] = len(g)
            for k in g[0]:
                agg[c][k] = round(sum(x[k] for x in g) / len(g), 4)
    return agg


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("parquet_glob")
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-prefix", type=int, default=2500)
    ap.add_argument("--max-file", type=int, default=2000)
    ap.add_argument("--max-between", type=int, default=3000)
    ap.add_argument("--max-suffix", type=int, default=200)
    ap.add_argument("--gen", type=int, default=0, help="also generate the next tool call, up to this many tokens")
    ap.add_argument("--lag", type=int, default=0, help="score the call this many turns after the splice (0 = first call)")
    ap.add_argument("--max-later", type=int, default=3000)
    ap.add_argument("--hybrid", action="store_true",
                    help="model has linear-attention layers: masks only reach its attention layers, so 'splice' "
                         "and 'mask_all' leave the span in the linear state (attention-only splice)")
    ap.add_argument("--out")
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    dt = torch.float32 if dev == "cpu" else torch.float16
    tok = AutoTokenizer.from_pretrained(a.model)
    try:
        model = AutoModelForCausalLM.from_pretrained(a.model, dtype=dt)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=dt)
    model = model.to(dev).eval()
    print(f"model {a.model} on {dev} ({dt})")
    if not a.hybrid:
        self_check(model, len(tok))
    leak, seen = leak_check(model, len(tok))
    print(f"leak-check: masked span changes later states by {leak:.2e} (unmasked: {seen:.2e})")
    if not a.hybrid and leak > 2e-2:
        raise SystemExit("leak-check failed: a masked span still reaches later tokens; use --hybrid if this model has linear-attention layers")
    if a.hybrid and leak < 1e-4:
        print("note: no leak found, this model behaves as attention-only under masks")
    paths = sorted(glob.glob(a.parquet_glob, recursive=True))
    stop_ids = {i for i in (tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id) if isinstance(i, int) and i >= 0}
    rows, t0 = [], time.time()
    for i, s in enumerate(samples(paths, tok, a.n, a.seed, a.max_prefix, a.max_file, a.max_between, a.max_suffix,
                                  a.lag, a.max_later), 1):
        m = score(model, s)
        row = {"instance": s["instance"], "nP": len(s["P"]), "nF": len(s["F"]), "nM": len(s["M"]),
               "nL": len(s.get("L") or []), "nS": len(s["S"]), "calls_between": s["calls_between"], "m": m}
        if a.gen:
            texts = {c: tok.decode(g) for c, g in generate_all(model, s, a.gen, stop_ids).items()}
            row["g"] = gen_metrics(texts, s["ref"])
            row["gen_text"] = {c: t[:200] for c, t in texts.items()}
            row["ref"] = s["ref"][:200]
        rows.append(row)
        if dev == "cuda":
            torch.cuda.empty_cache()
        el = time.time() - t0
        print(f"[{i}/{a.n}] P={row['nP']} F={row['nF']} M={row['nM']} splice kl={m['splice']['kl_evict']:.3f} "
              f"mask_all kl={m['mask_all']['kl_evict']:.3f} full kl={m['full']['kl_evict']:.3f} | "
              f"{el / i:.1f}s each, ~{el / i * (a.n - i) / 60:.0f} min left", flush=True)
        if a.out and i % 10 == 0:
            json.dump({"model": a.model, "n": len(rows), "aggregate": aggregate(rows), "rows": rows}, open(a.out, "w"))
    res = {"model": a.model, "device": dev, "n": len(rows), "seed": a.seed, "hybrid": a.hybrid, "lag": a.lag,
           "leak_check": {"masked": leak, "unmasked": seen},
           "limits": {"prefix": a.max_prefix, "file": a.max_file, "between": a.max_between, "suffix": a.max_suffix},
           "aggregate": aggregate(rows), "rows": rows}
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res["aggregate"], indent=1))


if __name__ == "__main__":
    main()
