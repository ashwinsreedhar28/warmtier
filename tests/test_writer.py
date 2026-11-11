import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from warmtier import gate as G
from warmtier.writer import install, kd_loss, lora_state, student_logp, teacher


def make():
    torch.manual_seed(0)
    cfg = transformers.LlamaConfig(vocab_size=300, hidden_size=64, intermediate_size=128, num_hidden_layers=3,
                                   num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=512)
    m = transformers.LlamaForCausalLM(cfg).eval()
    install(m, rank=4, alpha=8.0)
    return m


def ids(seed):
    g = torch.Generator().manual_seed(seed)
    r = lambda n: torch.randint(5, 300, (n,), generator=g).tolist()
    return r(14), r(21), r(9)


def single_pass_logp(model, P, F, S, cond):
    x = torch.tensor(P + F + S)
    s0 = len(P) + len(F)
    h = G.hidden(model, x, G.allowed_matrix(cond, len(P), len(F), len(S)),
                 gate_rows=(len(P), s0) if cond == "writer" else None)[s0 - 1:-1]
    G._set_gate(model, None, 0, x.device)
    return torch.log_softmax(model.get_output_embeddings()(h).float(), -1)


def test_zero_adapter_student_equals_iso():
    m = make()
    P, F, S = ids(1)
    _, cache = teacher(m, P, F, S)
    with torch.no_grad():
        got = student_logp(m, P, F, S, cache)
    assert torch.allclose(got, single_pass_logp(m, P, F, S, "iso"), atol=1e-4)


def test_teacher_equals_full():
    m = make()
    P, F, S = ids(2)
    t, _ = teacher(m, P, F, S)
    assert torch.allclose(t, single_pass_logp(m, P, F, S, "full"), atol=1e-4)


def test_trained_adapter_cached_path_equals_gate_writer_condition():
    m = make()
    for n, p in m.named_parameters():
        if n.endswith(".B"):
            torch.nn.init.normal_(p, std=0.05)
    P, F, S = ids(3)
    _, cache = teacher(m, P, F, S)
    with torch.no_grad():
        got = student_logp(m, P, F, S, cache)
    want = single_pass_logp(m, P, F, S, "writer")
    assert torch.allclose(got, want, atol=1e-4), float((got - want).abs().max())
    assert not torch.allclose(want, single_pass_logp(m, P, F, S, "iso"), atol=1e-3)   # adapter does something
    # the adapter must not touch the prefix: P's hidden states equal the plain model's
    x = torch.tensor(P + F + S)
    a = G.hidden(m, x, G.allowed_matrix("writer", 14, 21, 9), gate_rows=(14, 35))[:14]
    b = G.hidden(m, x, G.allowed_matrix("full", 14, 21, 9))[:14]
    assert torch.allclose(a, b, atol=1e-5)


def test_training_reduces_kd_loss():
    m = make()
    s = dict(zip("PFS", ids(4)))
    params = [p for p in m.parameters() if p.requires_grad]
    assert params and all(n.endswith((".A", ".B")) for n, p in m.named_parameters() if p.requires_grad)
    opt = torch.optim.AdamW(params, lr=1e-2)
    first = float(kd_loss(m, s))
    for _ in range(30):
        loss = kd_loss(m, s)
        opt.zero_grad(); loss.backward(); opt.step()
    assert float(kd_loss(m, s)) < 0.5 * first
    assert len(lora_state(m)) == 3 * 4 * 2


def test_teacher_gen_matches_gate_generate():
    from warmtier.writer import teacher_gen
    m = make()
    P, F, T = ids(11)
    Gt, rows, cache = teacher_gen(m, P, F, T, 6, set())
    assert Gt == G.generate(m, P, F, T, "full", None, 6, set())
    assert rows.shape[0] == len(Gt) + 1 and cache.get_seq_length() == len(P)


def test_gen_kd_rows_are_aligned():
    """With one prefix token and an untrained adapter the student differs from the teacher only by
    isolation from that token; shifting the student rows by one must be clearly worse."""
    from warmtier.writer import student_logp, teacher_gen
    m = make()
    _, F, T = ids(12)
    P = [7]
    Gt, t, cache = teacher_gen(m, P, F, T, 8, set())
    s = student_logp(m, P, F, T + Gt, cache)
    nT, n = len(T), len(Gt)
    kl = lambda a, b: float((a.exp() * (a - b)).sum(-1).mean())
    aligned = kl(t[:n], s[nT:nT + n])
    assert aligned < kl(t[:n - 1], s[nT + 1:nT + n]) and aligned < kl(t[1:n], s[nT:nT + n - 1])
    assert all(int(s[nT + i].argmax()) == Gt[i] for i in range(n)) or aligned < 0.5


def test_gen_kd_loss_trains_the_adapter():
    from warmtier.writer import gen_kd_loss
    m = make()
    P, F, T = ids(13)
    loss = gen_kd_loss(m, {"P": P, "F": F, "T": T}, 6, set())
    assert loss is not None and float(loss) >= 0
    loss.backward()
    grads = [p.grad for n, p in m.named_parameters() if n.endswith(".B")]
    assert grads and any(g is not None and float(g.abs().sum()) > 0 for g in grads)


def test_reader_adapter_trains_and_decodes_consistently():
    """With a reader adapter, cached greedy decoding must equal one full masked pass with the
    writer gate on the file and the reader gate on everything after it."""
    from warmtier.writer import gen_kd_loss, install as inst
    torch.manual_seed(0)
    cfg = transformers.LlamaConfig(vocab_size=300, hidden_size=64, intermediate_size=128, num_hidden_layers=3,
                                   num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=512)
    m = transformers.LlamaForCausalLM(cfg).eval()
    gate = inst(m, rank=4, alpha=8.0, reader_rank=4)
    assert gate.has_reader
    for n, p in m.named_parameters():                     # make both adapters non-trivial
        if n.endswith((".B", ".B2")):
            torch.nn.init.normal_(p, std=0.05)
    P, F, T = ids(21)
    gen = G.generate(m, P, F, T, "writer", None, 5, set())
    cur = P + F + T
    for t in gen:
        x = torch.tensor(cur)
        h = G.hidden(m, x, G.allowed_matrix("writer", len(P), len(F), len(cur) - len(P) - len(F)),
                     gate_rows=(len(P), len(P) + len(F)))[-1]
        G._set_gate(m, None, 0, x.device)
        _, head = G.parts(m)
        assert int(head(h).argmax()) == t
        cur.append(t)
    loss = gen_kd_loss(m, {"P": P, "F": F, "T": T}, 5, set())
    loss.backward()
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0 for n, p in m.named_parameters() if n.endswith(".B2"))
    assert gate.value is None and gate.reader is None
