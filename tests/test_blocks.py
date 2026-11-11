import pickle

from warmtier.blocks import reduce, run_units, view_units


def _view(path, total, lo, hi, text=lambda n: f"line {n}"):
    body = "\n".join(f"{n}:{text(n)}" for n in range(lo, hi + 1))
    return {"role": "user", "text": f"[File: {path} ({total} lines total)]\n{body}\n(Open file: {path})"}


def _open(path):
    return {"role": "ai", "text": f"```\nopen {path}\n```"}


def test_view_units_only_whole_blocks():
    lines = [(n, f"l{n}") for n in range(6, 26)]          # lines 6..25 of a 30-line file
    u = view_units("a.py", 30, lines)
    assert len(u["view"]) == 1 and len(u["1"]) == 20
    assert len(u["10"]) == 1                              # only 11..20 is whole
    assert u["25"] == [] and u["100"] == []


def test_last_block_may_be_short_at_end_of_file():
    lines = [(n, "x") for n in range(21, 30)]             # 21..29 of a 30-line file: line 30 missing
    assert len(view_units("a.py", 30, lines)["25"]) == 0
    lines = [(n, "x") for n in range(21, 31)]             # 26..30 is whole, 1..25 is not
    assert len(view_units("a.py", 30, lines)["25"]) == 1


def test_run_units_dedupes_and_skips_created_files():
    traj = [_open("a.py"), _view("a.py", 200, 1, 100), _open("a.py"), _view("a.py", 200, 1, 100),
            {"role": "ai", "text": "```\ncreate b.py\n```"}, _view("b.py", 1, 1, 1)]
    units, total = run_units(traj)
    assert len(units["view"]) == 1 and len(units["100"]) == 1 and len(units["10"]) == 10
    assert total == 2 * sum(len(f"line {n}") for n in range(1, 101))


def test_reduce_overlapping_windows(tmp_path):
    # run 1 sees lines 1..100, run 2 sees 51..150: no shared view, half the lines shared
    r1 = run_units([_open("a.py"), _view("a.py", 400, 1, 100)])
    r2 = run_units([_open("a.py"), _view("a.py", 400, 51, 150)])
    parts = []
    for i in range(2):                                    # two instances so k=2 has teams
        p = tmp_path / f"{i}.pkl"
        pickle.dump([(f"inst{i}", *r1), (f"inst{i}", *r2)], open(p, "wb"))
        parts.append(str(p))
    res = reduce(parts)["sizes"]["2"]
    assert res["teams"] == 2
    assert res["reuse_pct"]["view"] == 0 and res["reuse_pct"]["100"] == 0
    assert 20 < res["reuse_pct"]["50"] < 30 and 20 < res["reuse_pct"]["10"] < 30 and 20 < res["reuse_pct"]["1"] < 30
