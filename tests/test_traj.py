import json
from warmtier.traj import chunks, messages_of, summarize

SYS = "S" * 1000
FILE = "F" * 500


def nebius(iid, obs):
    return {"instance_id": iid, "trajectory": json.dumps([
        {"role": "system", "system_prompt": SYS},
        {"role": "user", "text": "fix " + iid + "x" * 300},
        {"role": "ai", "text": "open a.py"},
        {"role": "user", "text": obs},
        {"role": "ai", "text": "submit"},
    ])}


def test_shapes():
    assert messages_of(nebius("o__r-1", FILE))[0] == ("system", SYS)
    assert messages_of({"messages": [{"role": "assistant", "content": [{"type": "text", "text": "hi"}]}]}) == [("assistant", "hi")]


def test_calls_carrying():
    cs = list(chunks(nebius("o__r-1", FILE), "1"))
    assert [c["kind"] for c in cs] == ["system", "task", "assistant", "observation", "assistant"]
    assert [c["calls_carrying"] for c in cs] == [2, 2, 1, 1, 0]
    assert cs[0]["repo"] == "o__r"


def test_scopes():
    rows = [nebius("o__r-1", FILE), nebius("o__r-1", FILE), nebius("o__r-2", FILE), nebius("p__q-9", "G" * 500)]
    g = summarize(rows)
    assert g["trajectories"] == 4
    # system prompt shared by 4 runs across 2 repos: 3 extra copies
    assert g["cross_run_reusable_chars"]["cross_repo"] == 3 * 1000
    # file read shared by 3 runs in one repo, two instances
    assert g["cross_run_reusable_chars"]["cross_instance_same_repo"] == 2 * 500
    # task text shared by the two runs of instance 1
    assert g["cross_run_reusable_chars"]["same_instance"] == len("fix o__r-1" + "x" * 300)
