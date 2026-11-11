from warmtier.arch import MODELS, compute, kv_bytes_per_token


def test_kv_bytes():
    assert kv_bytes_per_token(MODELS["Qwen/Qwen3-4B"]) == 2 * 36 * 8 * 128 * 2 == 147_456
    assert kv_bytes_per_token(MODELS["Qwen/Qwen3.5-4B"]) == 2 * 8 * 4 * 256 * 2 == 32_768


def test_compute():
    r = compute(100_000)["models"]
    assert r["Qwen/Qwen3-4B"]["kv_gb_at_context"] == 14.75
    assert r["Qwen/Qwen3.5-4B"]["kv_per_token_vs_qwen3_4b"] == 0.222
    assert r["Qwen/Qwen3-8B"]["linear_state_mb_estimate"] == 0
