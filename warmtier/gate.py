"""Gate experiment: does a file's KV survive being reused out of context?

Run natively (needs a GPU or Apple Silicon):

    uv run --no-project --with torch --with transformers --with accelerate --with pyarrow \\
        python -m warmtier.gate 'data/open-swe-traces/data/claudecode/**/*.parquet' \\
        --model Qwen/Qwen3.5-4B --n 100 --out results/gate_qwen35_4b.json

For each sampled file read in a real agent trace, the prompt is split into
  P  everything before the file content (truncated on the left)
  F  the file content the Read tool returned
  S  the assistant turn that followed
and the model scores S, teacher-forced, under several attention masks:

  full      normal causal attention (what full prefill gives)
  iso       F cannot attend to P. F's keys and values are then exactly what
            encoding the file on its own at that offset would produce, so this
            is position-corrected KV reuse with no repair
  repair_k  iso, but the first k tokens of F attend to P (recompute the head
            of the chunk)
  repair_p  iso, but every (1/p)-th token of F attends to P (recompute a
            fraction p, evenly spaced)
  repair_devN  iso, but the N% of F tokens whose early-layer hidden state
            moves most between iso and full attend to P (CacheBlend-style
            selection on a check layer)
  nofile    S cannot attend to F at all (the floor: file never read)

Because a mask reproduces the reused KV exactly, no engine changes are needed.

Reported per condition, relative to full:
  kl        mean KL(full || condition) per token of S, in nats
  top1      share of S positions where the top prediction matches full
  nll       mean negative log-likelihood of the real S tokens
  retained  (nll_nofile - nll) / (nll_nofile - nll_full): how much of the
            file's benefit survives (1 = all, 0 = as bad as no file)

With --gen N the script also generates the next tool call greedily (up to N
tokens) under each condition, after forcing the turn to open a tool call, and
reports how often the tool name and the whole call match the call the real
agent made (name_ref, exact_ref) and the call generated under full
(name_full, exact_full).

Two self-checks run first on the real model and abort on failure:
  1. a full 4-D mask gives the same logits as no mask
  2. under iso, F's hidden states do not change when P changes
"""
from __future__ import annotations

import argparse
import glob
import json
import random
import statistics
import time

import torch

CONDITIONS = ("full", "iso", "repair_k64", "repair_p15", "repair_p30", "repair_dev15", "repair_dev30", "nofile")
CHECK_LAYER = 2


def allowed_matrix(cond: str, nP: int, nF: int, nS: int, dev: torch.Tensor | None = None) -> torch.Tensor:
    L = nP + nF + nS
    a = torch.tril(torch.ones(L, L, dtype=torch.bool))
    f0, s0 = nP, nP + nF
    if cond == "nofile":
        a[s0:, f0:s0] = False
        return a
    if cond == "full":
        return a
    a[f0:s0, :f0] = False                      # F rows cannot see P
    if cond == "writer":
        return a
    if cond.startswith("repair_dev"):
        k = max(1, round(nF * int(cond[len("repair_dev"):]) / 100))
        rows = f0 + torch.topk(dev, min(k, nF)).indices.cpu()
        a[rows, :f0] = True
    elif cond.startswith("repair_k"):
        k = min(int(cond[len("repair_k"):]), nF)
        a[f0:f0 + k, :f0] = True
    elif cond.startswith("repair_p"):
        step = max(1, round(100 / int(cond[len("repair_p"):])))
        rows = torch.arange(f0, s0, step)
        a[rows, :f0] = True
    return a


def parts(model):
    base = getattr(model, "model", None) or model.base_model
    return base, model.get_output_embeddings()


def _mask4d(model, allowed: torch.Tensor, dev) -> torch.Tensor:
    dt = next(model.parameters()).dtype
    m = torch.zeros(allowed.shape, dtype=dt, device=dev)
    m.masked_fill_(~allowed.to(dev), torch.finfo(dt).min)
    return m[None, None]


@torch.no_grad()
def _set_gate(model, rows: tuple[int, int] | None, length: int, device) -> None:
    g = getattr(model, "_writer_gate", None)
    if g is None:
        return
    if rows is None:
        g.value = None
        g.reader = None
    else:
        v = torch.zeros(1, length, 1, device=device)
        v[:, rows[0]:rows[1]] = 1.0
        g.value = v
        if getattr(g, "has_reader", False):          # reader adapter: every row after the file
            r = torch.zeros(1, length, 1, device=device)
            r[:, rows[1]:] = 1.0
            g.reader = r


def hidden(model, ids: torch.Tensor, allowed: torch.Tensor | None, layer: int | None = None,
           gate_rows: tuple[int, int] | None = None) -> torch.Tensor:
    base, _ = parts(model)
    _set_gate(model, gate_rows, len(ids), ids.device)
    kw = {}
    if allowed is not None:
        kw["attention_mask"] = _mask4d(model, allowed, ids.device)
    if layer is None:
        return base(input_ids=ids[None], **kw).last_hidden_state[0]
    return base(input_ids=ids[None], output_hidden_states=True, **kw).hidden_states[layer][0]


@torch.no_grad()
def deviation(model, ids: torch.Tensor, nP: int, nF: int, nS: int) -> torch.Tensor:
    """Per-token distance between iso and full hidden states of F at the check layer."""
    a = hidden(model, ids, allowed_matrix("full", nP, nF, nS), CHECK_LAYER)[nP:nP + nF].float()
    b = hidden(model, ids, allowed_matrix("iso", nP, nF, nS), CHECK_LAYER)[nP:nP + nF].float()
    return (a - b).norm(dim=-1)


@torch.no_grad()
def generate(model, P: list[int], F: list[int], T: list[int], cond: str, dev, max_new: int, stop_ids: set[int]) -> list[int]:
    """Greedy decode after P+F+T under one condition, with a KV cache built under that mask."""
    base, head = parts(model)
    device = next(model.parameters()).device
    nP, nF, nT = len(P), len(F), len(T)
    ids = torch.tensor(P + F + T, device=device)
    allowed = allowed_matrix(cond, nP, nF, nT, dev)
    _set_gate(model, (nP, nP + nF) if cond == "writer" else None, len(ids), device)
    out = base(input_ids=ids[None], attention_mask=_mask4d(model, allowed, device), use_cache=True)
    _set_gate(model, None, 0, device)
    g = getattr(model, "_writer_gate", None)
    reader_on = cond == "writer" and g is not None and getattr(g, "has_reader", False)
    pkv = out.past_key_values
    tok = int(head(out.last_hidden_state[0, -1]).argmax())
    gen = []
    dt = next(model.parameters()).dtype
    row = torch.zeros(1, 1, 1, len(ids), dtype=dt, device=device)
    if cond == "nofile":
        row[..., nP:nP + nF] = torch.finfo(dt).min
    for _ in range(max_new):
        if tok in stop_ids:
            break
        gen.append(tok)
        row = torch.cat([row, torch.zeros(1, 1, 1, 1, dtype=dt, device=device)], -1)
        if reader_on:
            g.reader = torch.ones(1, 1, 1, device=device)
        out = base(input_ids=torch.tensor([[tok]], device=device), past_key_values=pkv, attention_mask=row, use_cache=True)
        if reader_on:
            g.reader = None
        pkv = out.past_key_values
        tok = int(head(out.last_hidden_state[0, -1]).argmax())
    return gen


@torch.no_grad()
def score(model, P: list[int], F: list[int], S: list[int], conds=None, dev=None) -> dict:
    """Teacher-forced metrics over S for each condition."""
    conds = conds or CONDITIONS
    device = next(model.parameters()).device
    ids = torch.tensor(P + F + S, device=device)
    nP, nF, nS = len(P), len(F), len(S)
    s0 = nP + nF
    _, head = parts(model)
    tgt = ids[s0:]
    logp = {}
    if dev is None and any(c.startswith("repair_dev") for c in conds):
        dev = deviation(model, ids, nP, nF, nS)
    for c in conds:
        h = hidden(model, ids, allowed_matrix(c, nP, nF, nS, dev), gate_rows=(nP, s0) if c == "writer" else None)[s0 - 1:-1]
        _set_gate(model, None, 0, ids.device)
        logp[c] = torch.log_softmax(head(h).float(), dim=-1)
    full = logp["full"]
    out = {}
    for c in conds:
        lp = logp[c]
        out[c] = {
            "kl": float((full.exp() * (full - lp)).sum(-1).mean()),
            "top1": float((lp.argmax(-1) == full.argmax(-1)).float().mean()),
            "nll": float(-lp.gather(1, tgt[:, None]).mean()),
        }
    return out


@torch.no_grad()
def self_check(model, vocab: int) -> None:
    g = torch.Generator().manual_seed(0)
    dev = next(model.parameters()).device
    r = lambda n: torch.randint(10, min(vocab, 5000), (n,), generator=g).tolist()
    P1, P2, F, S = r(24), r(24), r(40), r(12)
    ids = torch.tensor(P1 + F + S, device=dev)
    a = hidden(model, ids, None).float()
    b = hidden(model, ids, allowed_matrix("full", 24, 40, 12)).float()
    d1 = float((a - b).abs().max() / a.abs().max())
    iso = allowed_matrix("iso", 24, 40, 12)
    h1 = hidden(model, ids, iso)[24:64].float()
    h2 = hidden(model, torch.tensor(P2 + F + S, device=dev), iso)[24:64].float()
    d2 = float((h1 - h2).abs().max() / h1.abs().max())
    f1 = hidden(model, ids, allowed_matrix("full", 24, 40, 12))[24:64].float()
    d3 = float((h1 - f1).abs().max() / f1.abs().max())
    print(f"self-check: 4-D mask vs none {d1:.2e} | iso F vs other prefix {d2:.2e} | iso vs full on F {d3:.2e}")
    if d1 > 2e-2:
        raise SystemExit("self-check 1 failed: this model does not honour a 4-D attention mask the same as no mask")
    if d2 > 2e-2:
        raise SystemExit("self-check 2 failed: F still depends on the prefix under the iso mask")
    if d3 < 1e-4:
        raise SystemExit("self-check failed: iso mask had no effect, the mask is being ignored")


def chatml(role: str, text: str) -> str:
    return f"<|im_start|>{role}\n{text}<|im_end|>\n"


def assistant_text(m: dict, with_reasoning: bool) -> str:
    t = (m.get("reasoning_content") or "") if with_reasoning else ""
    t += m.get("content") or ""
    for tc in m.get("tool_calls") or []:
        t += "\n" + json.dumps({"name": tc["function"]["name"], "arguments": tc["function"]["arguments"]})
    return t


def find_sample(row: dict, rng: random.Random):
    """Pick one Read result that is directly followed by an assistant turn."""
    msgs = row["messages"]
    cands = []
    pending = []
    for j, m in enumerate(msgs):
        if m["role"] == "assistant":
            pending = [tc["function"]["name"] for tc in (m.get("tool_calls") or [])]
        elif m["role"] == "tool" and pending:
            name = pending.pop(0)
            if name == "Read" and j + 1 < len(msgs) and msgs[j + 1]["role"] == "assistant" and len(m.get("content") or "") > 800:
                cands.append(j)
    if not cands:
        return None
    j = rng.choice(cands)
    pre = ""
    for m in msgs[:j]:
        role = m["role"]
        pre += chatml(role, assistant_text(m, False) if role == "assistant" else (m.get("content") or ""))
    nxt = msgs[j + 1]
    calls = nxt.get("tool_calls") or []
    ref = json.dumps({"name": calls[0]["function"]["name"], "arguments": calls[0]["function"]["arguments"]}) if calls else None
    return pre + "<|im_start|>tool\n", msgs[j]["content"], "<|im_end|>\n<|im_start|>assistant\n", assistant_text(nxt, True), ref


def samples(paths: list[str], tok, n: int, seed: int, max_p: int, max_f: int, max_s: int):
    import pyarrow.parquet as pq
    rng = random.Random(seed)
    paths = paths[:]
    rng.shuffle(paths)
    got = 0
    per_file = max(1, -(-n // len(paths)))
    for p in paths:
        t = pq.ParquetFile(p)
        rows = t.read_row_group(0, columns=["instance_id", "messages"]).to_pylist() if t.metadata.num_row_groups else []
        rng.shuffle(rows)
        took = 0
        for row in rows:
            s = find_sample(row, rng)
            if not s:
                continue
            pre, f, mid, suf, ref = s
            F = tok(f, add_special_tokens=False)["input_ids"]
            if not 200 <= len(F) <= max_f:
                continue
            S = (tok(mid, add_special_tokens=False)["input_ids"] + tok(suf, add_special_tokens=False)["input_ids"])[:max_s]
            n_mid = len(tok(mid, add_special_tokens=False)["input_ids"])
            if len(S) - n_mid < 16:
                continue
            P = tok(pre, add_special_tokens=False)["input_ids"][-max_p:]
            T = tok(mid + '\n{"name": "', add_special_tokens=False)["input_ids"]
            yield {"instance": row["instance_id"], "P": P, "F": F, "S": S, "T": T, "ref": ref}
            got += 1
            took += 1
            if got >= n:
                return
            if took >= per_file:
                break


def call_name(text: str) -> str:
    return text.split('"', 1)[0]


def gen_metrics(texts: dict[str, str], ref: str) -> dict:
    """texts: condition -> generated continuation after '{"name": "'."""
    ref_tail = ref[len('{"name": "'):]
    norm = lambda t: t.split("<|im_end|>")[0].strip()
    full = norm(texts["full"])
    out = {}
    for c, t in texts.items():
        t = norm(t)
        out[c] = {"name_ref": call_name(t) == call_name(ref_tail), "exact_ref": t == ref_tail.strip(),
                  "name_full": call_name(t) == call_name(full), "exact_full": t == full}
    return out


def aggregate(rows: list[dict]) -> dict:
    agg = {}
    for c in CONDITIONS:
        kl = [r["m"][c]["kl"] for r in rows]
        top1 = [r["m"][c]["top1"] for r in rows]
        nll = [r["m"][c]["nll"] for r in rows]
        num = sum(r["m"]["nofile"]["nll"] - r["m"][c]["nll"] for r in rows)
        den = sum(r["m"]["nofile"]["nll"] - r["m"]["full"]["nll"] for r in rows)
        agg[c] = {"kl_mean": round(statistics.mean(kl), 4), "kl_median": round(statistics.median(kl), 4),
                  "top1_mean": round(statistics.mean(top1), 4), "nll_mean": round(statistics.mean(nll), 4),
                  "retained": round(num / den, 4) if den else None}
        g = [r["g"][c] for r in rows if r.get("g")]
        if g:
            agg[c]["gen_n"] = len(g)
            for k in ("name_ref", "exact_ref", "name_full", "exact_full"):
                agg[c][k] = round(sum(x[k] for x in g) / len(g), 4)
    return agg


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("parquet_glob")
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-prefix", type=int, default=3000)
    ap.add_argument("--max-file", type=int, default=2000)
    ap.add_argument("--max-suffix", type=int, default=200)
    ap.add_argument("--gen", type=int, default=0, help="also generate the next tool call, up to this many tokens")
    ap.add_argument("--adapter", help="writer adapter from warmtier.writer; adds the 'writer' condition")
    ap.add_argument("--only", help="comma-separated conditions to run (full is always included), e.g. iso,writer,nofile")
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
    self_check(model, len(tok))
    if a.adapter:
        global CONDITIONS
        from .writer import load_adapter
        load_adapter(model, a.adapter)
        CONDITIONS = CONDITIONS[:-1] + ("writer", "nofile")
        print("adapter loaded:", a.adapter)
    if a.only:
        globals()["CONDITIONS"] = ("full",) + tuple(c for c in CONDITIONS if c in set(a.only.split(",")) and c != "full")
        print("conditions:", CONDITIONS)
    paths = sorted(glob.glob(a.parquet_glob, recursive=True))
    rows, t0 = [], time.time()
    stop_ids = {i for i in (tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id) if isinstance(i, int) and i >= 0}
    for i, s in enumerate(samples(paths, tok, a.n, a.seed, a.max_prefix, a.max_file, a.max_suffix), 1):
        ids = torch.tensor(s["P"] + s["F"] + s["S"], device=next(model.parameters()).device)
        d = deviation(model, ids, len(s["P"]), len(s["F"]), len(s["S"])) if any(c.startswith("repair_dev") for c in CONDITIONS) else None
        m = score(model, s["P"], s["F"], s["S"], dev=d)
        row = {"instance": s["instance"], "nP": len(s["P"]), "nF": len(s["F"]), "nS": len(s["S"]), "m": m}
        if a.gen and s["ref"]:
            texts = {c: tok.decode(generate(model, s["P"], s["F"], s["T"], c, d, a.gen, stop_ids)) for c in CONDITIONS}
            row["g"] = gen_metrics(texts, s["ref"])
            row["gen_text"] = {c: texts[c][:200] for c in ("full", "iso", "nofile")}
            row["ref"] = s["ref"][:200]
        rows.append(row)
        if dev == "mps":
            torch.mps.empty_cache()
        el = time.time() - t0
        print(f"[{i}/{a.n}] P={len(s['P'])} F={len(s['F'])} S={len(s['S'])} iso kl={m['iso']['kl']:.3f} "
              f"top1={m['iso']['top1']:.2f} | {el / i:.1f}s each, ~{el / i * (a.n - i) / 60:.0f} min left", flush=True)
        if a.out and i % 10 == 0:
            json.dump({"model": a.model, "n": len(rows), "aggregate": aggregate(rows), "rows": rows}, open(a.out, "w"))
    res = {"model": a.model, "device": dev, "n": len(rows), "seed": a.seed,
           "limits": {"prefix": a.max_prefix, "file": a.max_file, "suffix": a.max_suffix},
           "aggregate": aggregate(rows), "rows": rows}
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res["aggregate"], indent=1))


if __name__ == "__main__":
    main()
