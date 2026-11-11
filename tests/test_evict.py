from warmtier.evict import simulate

# system, task, a1 (reads file), view V, a2 (edits), edit result E, a3, tool out, a4
MSGS = [
    (0, 100, False, None),   # system
    (0, 50, False, None),    # task
    (1, 10, True, None),     # assistant 1
    (1, 400, False, 2),      # view, stale from call 2
    (2, 10, True, None),     # assistant 2
    (2, 30, False, None),    # edit result
    (3, 10, True, None),     # assistant 3
    (3, 20, False, None),    # tool output
]
# 4 assistant messages needed for 4 calls
MSGS.append((4, 10, True, None))


def test_append():
    prefill, ctx, ev = simulate(MSGS, "append")
    assert prefill == 150 + 400 + 30 + 20 and ev == 0
    assert ctx == 150 + 560 + 600 + 630


def test_evict_now_reprefills_suffix():
    prefill, ctx, ev = simulate(MSGS, "evict_now")
    # at call 2 the view goes; assistant 2 (10) sits after it and must be redone; the edit result is new anyway
    assert ev == 1 and prefill == 150 + 400 + (30 + 10) + 20
    assert ctx == 150 + 560 + 200 + 230


def test_splice_is_free():
    prefill, ctx, ev = simulate(MSGS, "splice")
    assert prefill == 150 + 400 + 30 + 20 and ctx == 150 + 560 + 200 + 230


def test_lazy_waits():
    big = [(0, 5000, False, None)] + MSGS
    p_now, c_now, e_now = simulate(big, "evict_now")
    p_lazy, c_lazy, e_lazy = simulate(big, "lazy_25")
    assert e_now == 1 and e_lazy == 0 and c_lazy > c_now
