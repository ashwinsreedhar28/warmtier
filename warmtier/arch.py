"""What per-token KV costs on attention-only versus hybrid models.

    python -m warmtier.arch --out results/arch_kv.json

Only attention layers keep keys and values per context token. A linear
attention layer (Gated DeltaNet in Qwen3.5) keeps a fixed-size state instead,
however long the context is. That changes what reuse and eviction can do:

                          attention layer        linear-attention layer
  per-token state         K and V per token      none (fixed-size state)
  prefix reuse            yes, any block edge    only at a saved state snapshot
  reuse out of position   KV can be moved, but   state after the chunk depends on
                          is wrong unrepaired    the state before it
                          (gate.py)
  cut a span out          delete its KV          not possible without replaying
                          (splice.py)            everything after the span

So on a hybrid, evicting a stale view frees KV memory in the attention layers
only, and the linear layers either keep the stale text folded into their state
or force a replay.

Config values were read from each model's config.json on Hugging Face on
2026-11-10. Bytes assume fp16 KV. The linear-state size is an estimate from
config dimensions (heads x key_dim x value_dim per layer) and has not been
checked against the implementation.
"""
from __future__ import annotations

import argparse
import json

MODELS = {
    "Qwen/Qwen3-4B": {"layers": 36, "attn_layers": 36, "kv_heads": 8, "head_dim": 128},
    "Qwen/Qwen3-8B": {"layers": 36, "attn_layers": 36, "kv_heads": 8, "head_dim": 128},
    "Qwen/Qwen3.5-4B": {"layers": 32, "attn_layers": 8, "kv_heads": 4, "head_dim": 256,
                        "linear_layers": 24, "linear_value_heads": 32, "linear_key_dim": 128, "linear_value_dim": 128},
}
STALE_SHARE = {"sweagent": 0.151, "claudecode_style": 0.127}   # context saved by eviction, results/evict_*.json
BYTES = 2                                                      # fp16


def kv_bytes_per_token(m: dict) -> int:
    return 2 * m["attn_layers"] * m["kv_heads"] * m["head_dim"] * BYTES


def linear_state_bytes(m: dict) -> int:
    if "linear_layers" not in m:
        return 0
    return m["linear_layers"] * m["linear_value_heads"] * m["linear_key_dim"] * m["linear_value_dim"] * BYTES


def compute(context_tokens: int = 100_000) -> dict:
    base = kv_bytes_per_token(MODELS["Qwen/Qwen3-4B"])
    out = {"context_tokens": context_tokens, "stale_share_of_context": STALE_SHARE, "models": {}}
    for name, m in MODELS.items():
        per = kv_bytes_per_token(m)
        lin = linear_state_bytes(m)
        row = {"attention_layers": m["attn_layers"], "layers": m["layers"],
               "kv_bytes_per_token": per, "kv_per_token_vs_qwen3_4b": round(per / base, 3),
               "kv_gb_at_context": round(per * context_tokens / 1e9, 2),
               "linear_state_mb_estimate": round(lin / 1e6, 1) if lin else 0,
               "linear_state_in_tokens_of_kv_estimate": round(lin / per) if lin else 0}
        for k, s in STALE_SHARE.items():
            row[f"gb_freed_by_eviction_{k}"] = round(per * context_tokens * s / 1e9, 2)
        out["models"][name] = row
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--context", type=int, default=100_000)
    ap.add_argument("--out")
    a = ap.parse_args()
    res = compute(a.context)
    text = json.dumps(res, indent=2)
    if a.out:
        open(a.out, "w").write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
