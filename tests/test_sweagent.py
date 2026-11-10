from warmtier.sweagent import process_run

VIEW = "[File: /r/a.py (50 lines total)]\n1:import x\n2:y = 1\n3:z = 2\n(Open file: /r/a.py)\nbash-$"
EDITED = "[File: /r/a.py (50 lines total)]\n1:import x\n2:y = 5\n3:z = 2\nFile updated. Please review.\nbash-$"


def run(edit_cmd):
    return {"instance_id": "o__r-7", "model_name": "m", "target": True, "exit_status": "submitted", "trajectory": [
        {"role": "system", "system_prompt": "S" * 100, "text": None},
        {"role": "user", "text": "task"},
        {"role": "ai", "text": "look\n```\nopen a.py\n```"},
        {"role": "user", "text": VIEW},
        {"role": "ai", "text": "fix\n```\n" + edit_cmd + "\ny = 5\nend_of_edit\n```"},
        {"role": "user", "text": EDITED},
        {"role": "ai", "text": "done\n```\nsubmit\n```"},
        {"role": "user", "text": "ok"},
    ]}


def test_views_and_staleness():
    chunks, views, r = process_run(run("edit 2:2"), 1)
    assert [c[5] for c in chunks] == ["system", "task", "assistant", "observation", "assistant", "observation", "assistant", "observation"]
    assert [c[9] for c in chunks] == [3, 3, 2, 2, 1, 1, 0, 0]          # calls carrying each message
    assert len(views) == 2 and views[0][3] == "/r/a.py"
    assert views[0][7:9] == (1, 3)
    assert views[0][10] == 2 and views[0][11] == 2                      # stale from call 2 on
    assert views[1][10] is None and views[0][12] is False
    d = dict(zip(["n_calls", "n_views", "n_files", "n_edits"], [r[6], r[12], r[13], r[14]]))
    assert d == {"n_calls": 3, "n_views": 2, "n_files": 1, "n_edits": 1}
    assert r[17] == len(VIEW) * (3 - 2)                                 # stale char-calls, file level


def test_window_miss():
    _, views, _ = process_run(run("edit 40:41"), 1)
    assert views[0][10] == 2 and views[0][11] is None                   # file stale, window not
