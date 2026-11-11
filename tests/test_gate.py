import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from warmtier.gate import allowed_matrix, deviation, gen_metrics, generate, hidden, parts, score, self_check


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    cfg = transformers.LlamaConfig(vocab_size=300, hidden_size=64, intermediate_size=128, num_hidden_layers=3,
                                   num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=512)
    return transformers.LlamaForCausalLM(cfg).eval()


def test_masks():
    a = allowed_matrix("iso", 3, 4, 2)
    assert not a[3:7, :3].any() and a[7:, :7].all() and a[3, 3]
    assert allowed_matrix("repair_k2", 3, 4, 2)[3:5, :3].all()
    d = allowed_matrix("repair_dev50", 3, 4, 2, torch.tensor([0.1, 9.0, 0.2, 8.0]))
    assert d[[4, 6], :3].all() and not d[[3, 5], :3].any()
    assert not allowed_matrix("repair_k2", 3, 4, 2)[5:7, :3].any()
    n = allowed_matrix("nofile", 3, 4, 2)
    assert not n[7:, 3:7].any() and n[7:, :3].all()
    assert not torch.triu(allowed_matrix("repair_p15", 5, 40, 3), 1).any()


def test_self_check_passes(model):
    self_check(model, 300)


def test_iso_equals_explicit_kv_splice(model):
    """The iso mask must match encoding F alone at its offset and splicing its KV in."""
    g = torch.Generator().manual_seed(1)
    P, F, S = (torch.randint(5, 300, (n,), generator=g) for n in (10, 17, 6))
    ids = torch.cat([P, F, S])
    masked = model.get_output_embeddings()(hidden(model, ids, allowed_matrix("iso", 10, 17, 6)))[27:]
    with torch.no_grad():
        kp = model(input_ids=P[None], use_cache=True).past_key_values
        kf = model(input_ids=F[None], position_ids=torch.arange(10, 27)[None], use_cache=True).past_key_values
        merged = transformers.DynamicCache()
        for layer in range(model.config.num_hidden_layers):
            pk, pv, fk, fv = kp.layers[layer].keys, kp.layers[layer].values, kf.layers[layer].keys, kf.layers[layer].values
            merged.update(torch.cat([pk, fk], 2), torch.cat([pv, fv], 2), layer)
        spliced = model(input_ids=S[None], past_key_values=merged, position_ids=torch.arange(27, 33)[None]).logits[0]
    assert torch.allclose(masked, spliced, atol=1e-4), float((masked - spliced).abs().max())


def test_score_shapes(model):
    g = torch.Generator().manual_seed(2)
    r = lambda n: torch.randint(5, 300, (n,), generator=g).tolist()
    m = score(model, r(30), r(40), r(12))
    assert m["full"]["kl"] == pytest.approx(0, abs=1e-6) and m["full"]["top1"] == 1.0
    assert m["iso"]["kl"] > 0 and m["nofile"]["kl"] > 0


def test_cached_generation_matches_recompute(model):
    """Greedy decoding from the masked KV cache must equal recomputing the whole masked forward each step."""
    g = torch.Generator().manual_seed(3)
    r = lambda n: torch.randint(5, 300, (n,), generator=g).tolist()
    P, F, T = r(12), r(20), r(5)
    ids = torch.tensor(P + F + T)
    dev = deviation(model, ids, 12, 20, 5)
    _, head = parts(model)
    for cond in ("full", "iso", "repair_dev30", "nofile"):
        fast = generate(model, P, F, T, cond, dev, 8, set())
        slow, cur = [], list(T)
        for _ in range(8):
            a = allowed_matrix(cond, 12, 20, len(cur), dev)
            h = hidden(model, torch.tensor(P + F + cur), a)
            nxt = int(head(h[-1]).argmax())
            slow.append(nxt); cur.append(nxt)
        assert fast == slow, cond


def test_gen_metrics():
    ref = '{"name": "Read", "arguments": "{}"}'
    m = gen_metrics({"full": 'Read", "arguments": "{}"}<|im_end|>', "iso": 'Bash", "arguments": "{}"}'}, ref)
    assert m["full"] == {"name_ref": True, "exact_ref": True, "name_full": True, "exact_full": True}
    assert m["iso"]["name_ref"] is False and m["iso"]["exact_full"] is False
