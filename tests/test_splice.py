import json

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from warmtier.splice import allowed_matrix, find_sample, gen_metrics, generate, leak_check, logp_tail


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    cfg = transformers.LlamaConfig(vocab_size=300, hidden_size=64, intermediate_size=128, num_hidden_layers=3,
                                   num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=512)
    return transformers.LlamaForCausalLM(cfg).eval()


def test_masks():
    s = allowed_matrix("splice", 3, 4, 5, 2)
    assert s[7:12, 3:7].all()                 # M still sees F
    assert not s[12:, 3:7].any()              # the new call does not
    assert s[12:, :3].all() and s[12:, 7:12].all()
    m = allowed_matrix("mask_all", 3, 4, 5, 2)
    assert not m[7:, 3:7].any() and m[3:7, :3].all()
    assert not torch.triu(s, 1).any()


def test_splice_mask_equals_cutting_kv(model):
    """The splice mask must match building KV for P+F+M, deleting F's entries, and decoding on."""
    g = torch.Generator().manual_seed(1)
    P, F, M, S = (torch.randint(5, 300, (n,), generator=g).tolist() for n in (9, 14, 11, 6))
    masked = logp_tail(model, P + F + M + S, allowed_matrix("splice", 9, 14, 11, 6), 5)
    with torch.no_grad():
        kv = model(input_ids=torch.tensor([P + F + M]), use_cache=True).past_key_values
        cut = transformers.DynamicCache()
        keep = list(range(9)) + list(range(23, 34))
        for layer in range(model.config.num_hidden_layers):
            cut.update(kv.layers[layer].keys[:, :, keep], kv.layers[layer].values[:, :, keep], layer)
        logits = model(input_ids=torch.tensor([S]), past_key_values=cut,
                       position_ids=torch.arange(34, 40)[None]).logits[0]
    spliced = torch.log_softmax(logits[:-1].float(), -1)
    assert torch.allclose(masked, spliced, atol=1e-4), float((masked - spliced).abs().max())


def test_full_mask_changes_nothing_and_splice_does(model):
    g = torch.Generator().manual_seed(2)
    ids = torch.randint(5, 300, (40,), generator=g).tolist()
    a = logp_tail(model, ids, None, 6)
    b = logp_tail(model, ids, allowed_matrix("full", 9, 14, 11, 6), 6)
    c = logp_tail(model, ids, allowed_matrix("splice", 9, 14, 11, 6), 6)
    assert torch.allclose(a, b, atol=1e-5)
    assert (a - c).abs().max() > 1e-3


def test_cached_generation_matches_recompute(model):
    g = torch.Generator().manual_seed(3)
    P, F, M, T = (torch.randint(5, 300, (n,), generator=g).tolist() for n in (8, 12, 10, 3))
    ids = P + F + M + T
    gen = generate(model, ids, allowed_matrix("splice", 8, 12, 10, 3), (8, 20), 5, set())
    cur = list(ids)
    for t in gen:
        lp = logp_tail(model, cur + [0], allowed_matrix("splice", 8, 12, 10, len(cur) + 1 - 30), 1)
        assert int(lp[0].argmax()) == t
        cur.append(t)


def _call(name, **args):
    return {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": name, "arguments": json.dumps(args)}}]}


def test_find_sample_cuts_at_first_call_after_edit():
    import random
    body = "\n".join(f"{i}\tline {i} of the file" for i in range(1, 80))
    msgs = [
        {"role": "system", "content": "sys"}, {"role": "user", "content": "fix it"},
        _call("Read", file_path="/r/a.py"), {"role": "tool", "content": body},
        _call("Bash", command="ls"), {"role": "tool", "content": "a.py"},
        _call("Edit", file_path="/r/a.py", old_string="line 5 of the file", new_string="x"),
        {"role": "tool", "content": "The file /r/a.py has been updated."},
        _call("Bash", command="pytest"), {"role": "tool", "content": "ok"},
    ]
    s = find_sample({"messages": msgs}, random.Random(0))
    assert s["file"] == body and s["path"] == "/r/a.py"
    assert json.loads(s["ref"])["name"] == "Bash" and "pytest" in s["ref"]
    assert "has been updated" in s["mid"] and "pytest" not in s["mid"]
    assert s["calls_between"] == 2
    assert find_sample({"messages": msgs[:6]}, random.Random(0)) is None


def test_gen_metrics():
    ref = '{"name": "Bash", "arguments": "{}"}'
    m = gen_metrics({"full": 'Bash", "arguments": "{}"}<|im_end|>', "evict_text": 'Read", "arguments": "{}"}',
                     "splice": 'Read", "arguments": "{}"}', "mask_all": 'Bash", "x"'}, ref)
    assert m["full"]["exact_ref"] and not m["splice"]["name_ref"]
    assert m["splice"]["exact_evict"] and not m["splice"]["name_full"]
    assert m["mask_all"]["name_ref"] and not m["mask_all"]["exact_ref"]


def test_leak_check_is_zero_on_attention_only(model):
    masked, seen = leak_check(model, 300)
    assert masked < 1e-5 and seen > 1e-3


def test_find_sample_lag_scores_a_later_call():
    import random
    body = "\n".join(f"{i}\tline {i} of the file" for i in range(1, 80))
    msgs = [
        {"role": "system", "content": "sys"}, {"role": "user", "content": "fix it"},
        _call("Read", file_path="/r/a.py"), {"role": "tool", "content": body},
        _call("Edit", file_path="/r/a.py", old_string="line 5 of the file", new_string="x"),
        {"role": "tool", "content": "The file /r/a.py has been updated."},
        _call("Bash", command="pytest"), {"role": "tool", "content": "2 failed"},
        _call("Bash", command="git diff"), {"role": "tool", "content": "diff"},
    ]
    s0 = find_sample({"messages": msgs}, random.Random(0), 0)
    s1 = find_sample({"messages": msgs}, random.Random(0), 1)
    assert s0["later"] == "" and "pytest" in s0["ref"]
    assert s1["mid"] == s0["mid"] and "pytest" in s1["later"] and "2 failed" in s1["later"] and "git diff" in s1["ref"]
    assert find_sample({"messages": msgs}, random.Random(0), 2) is None
