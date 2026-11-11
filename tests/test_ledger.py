import json
from warmtier.ledger import FileLedger


def tc(name, **args):
    return {"function": {"name": name, "arguments": json.dumps(args)}, "id": "x", "type": "function"}


def a(*calls):
    return {"role": "assistant", "content": "", "tool_calls": list(calls)}


def t(text):
    return {"role": "tool", "content": text}


READ_A = "1\timport x\n2\ty = 1\n3\tz = 2\n" + "4\tpad = 0\n" * 40
READ_B = "1\tother\n" + "2\tpad = 0\n" * 40
OK = "The file /r/a.py has been updated successfully."
MSGS = [
    {"role": "system", "content": "S" * 200}, {"role": "user", "content": "task"},
    a(tc("Read", file_path="/r/a.py"), tc("Read", file_path="/r/b.py")), t(READ_A), t(READ_B),
    a(tc("Edit", file_path="/r/a.py", old_string="y = 1", new_string="y = 5")), t(OK),
    a(tc("Read", file_path="/r/a.py")), t(READ_A.replace("y = 1", "y = 5")),
    a(tc("Edit", file_path="/r/a.py", old_string="nope", new_string="q")), t("<tool_use_error>String to replace not found</tool_use_error>"),
]


def test_scan_versions_and_staleness():
    led = FileLedger("claudecode")
    v = led.scan(MSGS)
    assert [(x.index, x.path, x.version) for x in v] == [(3, "/r/a.py", 0), (4, "/r/b.py", 0), (8, "/r/a.py", 1)]
    assert v[0].stale_file == 2 and v[0].stale_window == 2      # edited after the first model call that saw it
    assert v[1].stale_file is None and v[2].stale_file is None  # other file; fresh re-read; failed edit ignored


def test_window_level_ignores_edits_elsewhere():
    msgs = MSGS[:5] + [a(tc("Edit", file_path="/r/a.py", old_string="not shown here", new_string="q")), t(OK)]
    v = FileLedger().scan(msgs)[0]
    assert v.stale_file == 2 and v.stale_window is None


def test_plan_and_apply():
    led = FileLedger()
    assert led.plan(MSGS, mode="splice") == [3]
    assert led.plan(MSGS, mode="batch", threshold=0.9) == []
    assert led.plan(MSGS, mode="batch", threshold=0.2) == [3]
    out = led.apply(MSGS, [3])
    assert out[3]["content"].startswith("[stale view of /r/a.py") and MSGS[3]["content"] == READ_A
    assert led.plan(out, mode="splice") == []                   # already stubbed


def test_note_validity_tags():
    led = FileLedger()
    tag = led.tag("y is set in a.py", ["/r/a.py", "/r/b.py"], MSGS[:5])
    assert led.valid(tag, MSGS[:5])
    assert not led.valid(tag, MSGS)                             # a.py was edited since
    assert not led.valid(led.tag("b", ["/r/b.py"], MSGS), MSGS, changed_paths={"/r/b.py"})


def test_shwin_text_protocol():
    msgs = [
        {"role": "system", "content": "s"}, {"role": "user", "content": "task"},
        {"role": "assistant", "content": '{"type":"tool_call","name":"read_file","arguments":{"path":"a.py"}}'},
        {"role": "user", "content": "<tool_result name='read_file'>\n" + "line\n" * 60 + "\n</tool_result>"},
        {"role": "assistant", "content": '{"type":"tool_call","name":"write_file","arguments":{"path":"a.py","content":"x"}}'},
        {"role": "user", "content": "<tool_result name='write_file'>\nwrote a.py\n</tool_result>"},
    ]
    led = FileLedger("shwin")
    assert [v.index for v in led.stale(msgs)] == [3]
    assert "<tool_result name='read_file'>\n[stale view of a.py" in led.apply(msgs, [3])[3]["content"]
