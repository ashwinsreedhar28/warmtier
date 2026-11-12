import random

from warmtier import evict, policy


def _random_run(rng):
    msgs, call = [], 0
    for _ in range(rng.randint(3, 25)):
        for _ in range(rng.randint(0, 2)):
            stale = rng.choice([None, None, call + rng.randint(1, 6)])
            msgs.append((call, rng.randint(50, 5000), False, stale))
        msgs.append((call, rng.randint(20, 800), True, None))
        call += 1
    return msgs


def test_matches_the_audited_evict_replay_when_stubs_are_empty():
    rng = random.Random(0)
    for _ in range(300):
        old = _random_run(rng)
        new = [(c, ch, a, s, 0) for c, ch, a, s in old]
        assert policy.simulate(new, "prefix") == evict.simulate(old, "evict_now")
        assert policy.simulate(new, "prefix_lazy") == evict.simulate(old, "lazy_25")
        assert policy.simulate(new, "splice") == evict.simulate(old, "splice")
        assert policy.append_only(new) == evict.simulate(old, "append")[:2]


def test_splice_prefills_only_the_stub_and_context_shrinks_by_the_rest():
    msgs = [(0, 1000, False, 1, 80), (0, 10, True, None, 0), (1, 500, False, None, 0), (1, 10, True, None, 0),
            (2, 10, True, None, 0)]
    base = policy.append_only(msgs)
    p, ctx, ev = policy.simulate(msgs, "splice")
    assert ev == 1 and p == base[0] + 80 and ctx == base[1] - 2 * (1000 - 80)
    p2, ctx2, _ = policy.simulate(msgs, "prefix")
    assert ctx2 == ctx and p2 == base[0] + 80 + 20  # stub, then both assistant turns after it are prefilled again


def test_view_signals_find_duplicates_and_keep_edit_calls():
    v = lambda call, sha, lo, hi, sf=None, sw=None: dict(file="a.py", call=call, sha=sha, line_lo=lo, line_hi=hi,
                                                         stale_file_call=sf, stale_window_call=sw)
    sig = policy.view_signals([v(1, "x", 1, 50, sf=6), v(3, "y", 1, 100), v(5, "x", 200, 250), v(7, "x", 300, 310)])
    assert sig[(1, "x")] == {"edit": None, "edit_file": 6, "dup": 3}       # covered by the 1-100 view
    assert sig[(3, "y")]["dup"] is None and sig[(5, "x")]["dup"] == 7       # same bytes later


def test_age_policy_drops_live_text_and_counts_blind_first_edits():
    rows = [(0, "observation", 3000, "x"), (0, "assistant", 10, ""), (1, "assistant", 10, ""), (2, "assistant", 10, ""),
            (3, "assistant", 10, ""), (4, "assistant", 10, "")]
    sig = {(0, "x"): {"edit": 4, "edit_file": 4, "dup": None}}
    age = policy.plan(rows, sig, "age2")
    assert age[0][3] == 2 and age[0][4] == policy.AGE_STUB
    assert policy.plan(rows, sig, "age5")[0][3] == 5 and policy.plan(rows, {}, "semantic2")[0][3] == 2
    assert policy.plan(rows, sig, "semantic2")[0][3] == 4  # a view waits for its edit
    d = policy.classify(rows, sig, age, 5)
    assert d["live"] == 2800 and d["blind_first_edits"] == 1 and d["stale"] == 0 and d["live_other"] == 0
    ed = policy.plan(rows, sig, "edit")
    d = policy.classify(rows, sig, ed, 5)
    assert ed[0][3] == 4 and d["stale"] == 3000 - policy.VIEW_STUB and d["blind_first_edits"] == 0
