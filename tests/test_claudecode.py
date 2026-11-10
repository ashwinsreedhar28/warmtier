import json
from warmtier.claudecode import _bash_edit, _bash_view, process_run


def tc(name, **args):
    return {"function": {"name": name, "arguments": json.dumps(args)}, "id": "x", "type": "function"}


def a(*calls):
    return {"role": "assistant", "content": "", "reasoning_content": "think", "tool_calls": list(calls)}


def t(text):
    return {"role": "tool", "content": text}


READ = "1\timport x\n2\ty = 1\n3\tz = 2"
CAT = "import x\n" + "pad = 0\n" * 40


def test_bash_parsers():
    assert _bash_view("cat /testbed/a.py") == ("/testbed/a.py", 0, 0)
    assert _bash_view("sed -n '10,20p' /testbed/a.py") == ("/testbed/a.py", 10, 20)
    assert _bash_view("cat /testbed/a.py | head") is None
    assert _bash_view("ls /testbed") is None
    assert _bash_edit("sed -i 's/a/b/' /testbed/a.py") == "/testbed/a.py"
    assert _bash_edit("cat > /testbed/new.py <<'EOF'\nx\nEOF") == "/testbed/new.py"
    assert _bash_edit("pytest -q") is None


def test_run():
    row = {"instance_id": "o__r-1", "repo": "o/r", "resolved": 1, "tools": ["{}"], "metadata": None, "messages": [
        {"role": "system", "content": "S" * 50},
        {"role": "user", "content": "task"},
        a(tc("Read", file_path="/testbed/a.py"), tc("Bash", command="cat /testbed/b.py")),
        t(READ), t(CAT),
        a(tc("Edit", file_path="/testbed/a.py", old_string="y = 1", new_string="y = 5", replace_all=False)),
        t("The file /testbed/a.py has been updated successfully."),
        a(tc("Edit", file_path="/testbed/a.py", old_string="nope", new_string="q", replace_all=False)),
        t("<tool_use_error>String to replace not found</tool_use_error>"),
        a(),
    ]}
    chunks, views, run = process_run(row, 1)
    assert run[6] == 4 and run[12] == 2 and run[14] == 1          # calls, views, edits
    va, vb = views
    assert va[3] == "/testbed/a.py" and (va[7], va[8]) == (1, 3)
    assert va[10] == 2 and va[11] == 2                             # stale from call 2, window too
    assert vb[3] == "/testbed/b.py" and vb[10] is None
    assert run[17] == len(READ) * (4 - 2)
    assert chunks[0][5] == "tools" and chunks[0][9] == 4
