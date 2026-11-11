from warmtier.team import team_stats


def test_team_stats_counts_only_other_runs():
    H, A, B, C = (True, 100, 1), (False, 10, 2), (False, 20, 3), (False, 30, 4)
    run1 = [H, A, B, B]            # B twice in one run: never reuse on its own
    run2 = [H, B, A, C]            # same header; body in a different order
    s = team_stats([run1, run2], [[("a.py", 50, 7)], [("a.py", 50, 7), ("a.py", 40, 8), ("b.py", 5, 9)]])
    assert s["tot"] == {"header": 200, "body": 110}
    assert s["pc"] == {"header": 100, "body": 0}        # chain breaks at the first differing message
    assert s["ca"] == {"header": 100, "body": 30}       # B and A seen in run1, C is new
    assert (s["v_tot"], s["v_content"], s["v_file"]) == (145, 50, 90)


def test_own_repeats_count_once():
    A = (False, 10, 2)
    s = team_stats([[A], [A, A, A]], [[("a.py", 50, 7)], [("a.py", 50, 7), ("a.py", 50, 7)]])
    assert s["tot"]["body"] == 40 and s["ca"]["body"] == 10
    assert (s["v_tot"], s["v_content"], s["v_file"]) == (150, 50, 50)
