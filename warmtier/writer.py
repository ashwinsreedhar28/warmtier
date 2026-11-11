"""Writer adapter: train a LoRA so a file encoded alone behaves as if in context.

    python -m warmtier.writer '<train parquet glob>' --model Qwen/Qwen3-4B \\
        --n 2000 --epochs 2 --out out/writer_qwen3_4b.pt

Setup (after SemPIC, arXiv 2607.28069, adapted to agent file reads):
  P  prompt before the file, F  the file content, S  the assistant's next turn.
  Teacher  the frozen model with normal attention over P+F+S.
  Student  the same frozen model, except that F cannot attend to P (so F's
           keys and values are what encoding the file alone would give) and a
           LoRA is switched on at F's positions only. P and S positions run
           the untouched base model, so serving is unchanged: the adapter is
           only used when a file's KV is written.
  Loss     KL(teacher || student) on the next-token distributions over S.

The LoRA is "gated": each wrapped Linear adds its low-rank update only where a
per-position gate is 1. warmtier.gate reads the same gate to score a trained
adapter as the "writer" condition.
"""
from __future__ import annotations

import argparse
import glob
import math
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as Fn

from .gate import allowed_matrix, parts, samples, _mask4d

TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")


class Gate:
    """Shared per-forward gate: None, or a [1, seq, 1] tensor of 0/1."""
    value: torch.Tensor | None = None


class GatedLoRA(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float, gate: Gate):
        super().__init__()
        self.base, self.gate, self.scale = base, gate, alpha / rank
        self.A = nn.Parameter(torch.empty(rank, base.in_features, dtype=torch.float32, device=base.weight.device))
        self.B = nn.Parameter(torch.zeros(base.out_features, rank, dtype=torch.float32, device=base.weight.device))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x):
        y = self.base(x)
        g = self.gate.value
        if g is None:
            return y
        d = Fn.linear(Fn.linear(x.float(), self.A), self.B) * self.scale
        return y + (d * g).to(y.dtype)


def install(model, rank: int = 8, alpha: float = 16.0) -> Gate:
    """Wrap attention projections with gated LoRA. Returns the shared gate."""
    gate = Gate()
    base, _ = parts(model)
    for p in model.parameters():
        p.requires_grad_(False)
    for layer in base.layers:
        attn = layer.self_attn
        for name in TARGETS:
            lin = getattr(attn, name)
            if isinstance(lin, nn.Linear):
                setattr(attn, name, GatedLoRA(lin, rank, alpha, gate))
    n = sum(isinstance(m, GatedLoRA) for m in model.modules())
    if n == 0:
        raise SystemExit("no attention projections named q_proj/k_proj/v_proj/o_proj found to adapt")
    model._writer_gate = gate
    return gate


def lora_state(model) -> dict:
    return {k: v.detach().cpu() for k, v in model.state_dict().items() if k.endswith(".A") or k.endswith(".B")}


def load_adapter(model, path: str) -> Gate:
    blob = torch.load(path, map_location="cpu")
    gate = install(model, blob["rank"], blob["alpha"])
    missing = model.load_state_dict(blob["state"], strict=False)
    assert not missing.unexpected_keys, missing.unexpected_keys[:3]
    return gate


def f_gate(n_before: int, nF: int, n_after: int, device) -> torch.Tensor:
    g = torch.zeros(1, n_before + nF + n_after, 1, device=device)
    g[:, n_before:n_before + nF] = 1.0
    return g


@torch.no_grad()
def teacher(model, P, F, S):
    """Full-attention logits over S, plus the KV cache of P for the student."""
    base, head = parts(model)
    dev = next(model.parameters()).device
    ids = torch.tensor(P + F + S, device=dev)
    model._writer_gate.value = None
    out = base(input_ids=ids[None], use_cache=True)
    s0 = len(P) + len(F)
    logp = torch.log_softmax(head(out.last_hidden_state[0, s0 - 1:-1]).float(), -1)
    cache = out.past_key_values
    drop = len(F) + len(S)
    try:
        cache.crop(-drop)                       # newer API: negative = tokens to remove
    except (ValueError, TypeError):
        cache.crop(len(P))                      # older API: positive = final length
    assert cache.get_seq_length() == len(P), (cache.get_seq_length(), len(P))
    return logp, cache


def student_logp(model, P, F, S, p_cache):
    """Student log-probs over S: F is isolated from P and written with the adapter on."""
    base, head = parts(model)
    dev = next(model.parameters()).device
    nP, nF, nS = len(P), len(F), len(S)
    allowed = allowed_matrix("iso", nP, nF, nS)[nP:]          # rows F+S, cols P+F+S
    model._writer_gate.value = f_gate(0, nF, nS, dev)
    try:
        out = base(input_ids=torch.tensor(F + S, device=dev)[None], past_key_values=p_cache,
                   attention_mask=_mask4d(model, allowed, dev), use_cache=True)
    finally:
        model._writer_gate.value = None
    return torch.log_softmax(head(out.last_hidden_state[0, nF - 1:-1]).float(), -1)


def kd_loss(model, s: dict) -> torch.Tensor:
    t_logp, cache = teacher(model, s["P"], s["F"], s["S"])
    s_logp = student_logp(model, s["P"], s["F"], s["S"], cache)
    return (t_logp.exp() * (t_logp - s_logp)).sum(-1).mean()


@torch.no_grad()
def teacher_gen(model, P, F, T, max_new: int, stop_ids: set[int]):
    """Greedy continuation G after P+F+T under full prefill, its log-probs, and the P cache.

    Row i of the returned log-probs is the distribution that produced G[i]; one extra row holds
    the distribution after the last token (usually the stop token)."""
    base, head = parts(model)
    dev = next(model.parameters()).device
    model._writer_gate.value = None
    out = base(input_ids=torch.tensor(P + F + T, device=dev)[None], use_cache=True)
    cache = out.past_key_values
    rows = [torch.log_softmax(head(out.last_hidden_state[0, -1]).float(), -1)]
    G = []
    while len(G) < max_new:
        tok = int(rows[-1].argmax())
        if tok in stop_ids:
            break
        G.append(tok)
        out = base(input_ids=torch.tensor([[tok]], device=dev), past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        rows.append(torch.log_softmax(head(out.last_hidden_state[0, -1]).float(), -1))
    drop = len(F) + len(T) + len(G)
    try:
        cache.crop(-drop)
    except (ValueError, TypeError):
        cache.crop(len(P))
    assert cache.get_seq_length() == len(P), (cache.get_seq_length(), len(P))
    return G, torch.stack(rows), cache


def gen_kd_loss(model, s: dict, max_new: int, stop_ids: set[int]) -> torch.Tensor | None:
    """Distil on the call the full-prefill model itself writes, the thing the gate scores.

    None when the teacher stops at once (nothing to match)."""
    G, t_logp, cache = teacher_gen(model, s["P"], s["F"], s["T"], max_new, stop_ids)
    if not G:
        return None
    nT = len(s["T"])
    # student_logp row j predicts (T+G)[j]; rows nT.. predict G[0..]. Teacher row i predicts G[i].
    s_logp = student_logp(model, s["P"], s["F"], s["T"] + G, cache)[nT:nT + len(G)]
    t_logp = t_logp[:len(G)]
    return (t_logp.exp() * (t_logp - s_logp)).sum(-1).mean()


@torch.no_grad()
def evaluate(model, val: list[dict]) -> float:
    return sum(float(kd_loss(model, s)) for s in val) / len(val)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("parquet_glob")
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--val", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=16.0)
    ap.add_argument("--max-prefix", type=int, default=2500)
    ap.add_argument("--max-file", type=int, default=1500)
    ap.add_argument("--max-suffix", type=int, default=160)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--gen-mix", type=float, default=0.0,
                    help="share of steps that distil on the call the full-prefill model generates, instead of the real next turn")
    ap.add_argument("--gen-max", type=int, default=64)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(a.seed)
    tok = AutoTokenizer.from_pretrained(a.model)
    try:
        model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16)
    model = model.to("cuda").eval()
    install(model, a.rank, a.alpha)
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"model {a.model}; trainable {sum(p.numel() for p in params) / 1e6:.2f}M params", flush=True)

    t0 = time.time()
    data = list(samples(sorted(glob.glob(a.parquet_glob, recursive=True)), tok, a.n + a.val, a.seed,
                        a.max_prefix, a.max_file, a.max_suffix))
    val, train = data[:a.val], data[a.val:]
    print(f"{len(train)} train / {len(val)} val samples tokenised in {time.time() - t0:.0f}s", flush=True)
    base_kl = evaluate(model, val)
    print(f"step 0 val KL (plain reuse, no repair) {base_kl:.4f}", flush=True)

    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    total = a.epochs * len(train) // a.accum
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda u: min(1.0, (u + 1) / 50) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(u, total) / total))))
    rng = random.Random(a.seed)
    stop_ids = {i for i in (tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id) if isinstance(i, int) and i >= 0}
    n_gen = 0
    step, ema, skipped, t0 = 0, None, 0, time.time()
    history = []
    for ep in range(a.epochs):
        rng.shuffle(train)
        for s in train:
            try:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = None
                    if a.gen_mix and rng.random() < a.gen_mix:
                        loss = gen_kd_loss(model, s, a.gen_max, stop_ids)
                        n_gen += loss is not None
                    if loss is None:
                        loss = kd_loss(model, s)
                (loss / a.accum).backward()
            except torch.cuda.OutOfMemoryError:
                skipped += 1
                opt.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                continue
            step += 1
            ema = float(loss) if ema is None else 0.98 * ema + 0.02 * float(loss)
            if step % a.accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            if step % 50 == 0:
                el = time.time() - t0
                print(f"step {step}/{a.epochs * len(train)} loss_ema {ema:.4f} {el / step:.2f}s/step "
                      f"~{el / step * (a.epochs * len(train) - step) / 60:.0f} min left mem {torch.cuda.max_memory_allocated() / 2**30:.1f}G skipped {skipped} gen-steps {n_gen}", flush=True)
            if step % 500 == 0:
                v = evaluate(model, val)
                history.append((step, v))
                print(f"== step {step} val KL {v:.4f} (start {base_kl:.4f})", flush=True)
                torch.save({"rank": a.rank, "alpha": a.alpha, "model": a.model, "state": lora_state(model),
                            "val_kl_start": base_kl, "history": history, "gen_mix": a.gen_mix, "n_train": len(train)}, a.out)
    v = evaluate(model, val)
    history.append((step, v))
    torch.save({"rank": a.rank, "alpha": a.alpha, "model": a.model, "state": lora_state(model),
                "val_kl_start": base_kl, "history": history, "gen_mix": a.gen_mix, "n_train": len(train)}, a.out)
    print(f"done: val KL {base_kl:.4f} -> {v:.4f}; saved {a.out}", flush=True)


if __name__ == "__main__":
    main()
