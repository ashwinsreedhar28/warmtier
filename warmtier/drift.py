"""Valid life of file content, from git history.

Question: if an agent reads a file at a random moment, how long does that
content stay valid (unchanged on the default branch)?

Method: for every tracked file matching the pathspec, build the list of
change events inside a window [t0, t1] on the first-parent history. A file's
life is cut into gaps between consecutive events. For a horizon D, a random
(file, time) point "survives" if the next change is more than D away:

    S(D) = sum(max(0, gap - D)) / sum(eligible time)

The last gap of a file still alive at t1 is right-censored, so points closer
than D to t1 are dropped from both sums. Deletion counts as a change.

Two weightings are reported:
  uniform  every file-second counts the same
  hot      each file's S_f(D) weighted by its change count in the window,
           a proxy for "files agents get sent to fix". This is a proxy; the
           real weighting (files agents actually read) comes from trajectories.
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import subprocess
from collections import defaultdict

HORIZONS = {"1h": 3600, "1d": 86400, "7d": 7 * 86400, "30d": 30 * 86400,
            "90d": 90 * 86400, "365d": 365 * 86400}


def _git(repo: str, *args: str) -> str:
    return subprocess.run(["git", "-C", repo, *args], check=True,
                          capture_output=True, text=True, errors="replace").stdout


def file_events(repo: str, years: float, pathspec: str):
    """Return (t0, t1, {path: (birth, [change times], death_or_None)})."""
    t1 = int(_git(repo, "log", "-1", "--format=%ct").strip())
    t0 = t1 - int(years * 365 * 86400)
    base = _git(repo, "rev-list", "-1", "--first-parent", f"--before={t0}", "HEAD").strip()
    files: dict[str, list] = {}
    if base:
        # ls-tree does not glob across directories, so filter here.
        for p in _git(repo, "ls-tree", "-r", "--name-only", base).splitlines():
            if fnmatch.fnmatch(p, pathspec):
                files[p] = [t0, [], None]
    log = _git(repo, "log", "--first-parent", "--reverse", f"--since={t0}",
               "--no-renames", "--name-status", "--format=%x00%ct", "--", pathspec)
    ts = None
    for line in log.splitlines():
        if line.startswith("\x00"):
            ts = int(line[1:])
            continue
        if not line.strip() or ts is None:
            continue
        status, _, path = line.partition("\t")
        rec = files.get(path)
        if status.startswith("A") or rec is None or rec[2] is not None:
            files[path] = [ts, [], None]          # born (or re-born) here
        elif status.startswith("D"):
            rec[2] = ts
        else:
            rec[1].append(ts)
    return t0, t1, files


def survival(t1: int, files: dict, horizon: int):
    """Uniform and hot-weighted S(D)."""
    num = den = 0.0
    hot_num = hot_den = 0.0
    for birth, changes, death in files.values():
        pts = [birth, *changes]
        f_num = f_den = 0.0
        for a, b in zip(pts, pts[1:]):            # closed gaps
            f_num += max(0, b - a - horizon)
            f_den += b - a
        last = pts[-1]
        if death is not None:                     # deletion is a change
            f_num += max(0, death - last - horizon)
            f_den += death - last
        else:                                     # censored at t1
            elig = max(0, t1 - last - horizon)
            f_num += elig
            f_den += elig
        num += f_num
        den += f_den
        w = len(changes) + (1 if death is not None else 0)
        if w and f_den > 0:
            hot_num += w * f_num / f_den
            hot_den += w
    return (num / den if den else float("nan"),
            hot_num / hot_den if hot_den else float("nan"))


def half_life_days(t1: int, files: dict, weighting: int) -> float:
    """Smallest horizon (days) where S drops to 0.5, by bisection."""
    lo, hi = 0.0, 3650.0
    if survival(t1, files, int(hi * 86400))[weighting] > 0.5:
        return float("inf")
    for _ in range(40):
        mid = (lo + hi) / 2
        if survival(t1, files, int(mid * 86400))[weighting] > 0.5:
            lo = mid
        else:
            hi = mid
    return hi


def analyze(repo: str, years: float = 3.0, pathspec: str = "*.py") -> dict:
    t0, t1, files = file_events(repo, years, pathspec)
    n_changes = sum(len(c) for _, c, _ in files.values())
    changed = sum(1 for _, c, d in files.values() if c or d is not None)
    by_count = defaultdict(int)
    for _, c, _ in files.values():
        by_count[len(c)] += 1
    top = sorted((len(c) for _, c, _ in files.values()), reverse=True)
    top10pct = top[: max(1, len(top) // 10)]
    out = {
        "repo": repo.rstrip("/").split("/")[-1], "window_years": years, "pathspec": pathspec,
        "files": len(files), "files_changed_in_window": changed,
        "change_events": n_changes,
        "share_of_changes_in_top10pct_files": round(sum(top10pct) / n_changes, 4) if n_changes else None,
        "survival": {}, "half_life_days": {},
    }
    for name, h in HORIZONS.items():
        u, hot = survival(t1, files, h)
        out["survival"][name] = {"uniform": round(u, 4), "hot": round(hot, 4)}
    out["half_life_days"] = {"uniform": round(half_life_days(t1, files, 0), 1),
                             "hot": round(half_life_days(t1, files, 1), 1)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("repos", nargs="+")
    ap.add_argument("--years", type=float, default=3.0)
    ap.add_argument("--pathspec", default="*.py")
    ap.add_argument("--out")
    a = ap.parse_args()
    res = [analyze(r, a.years, a.pathspec) for r in a.repos]
    text = json.dumps(res, indent=2)
    if a.out:
        open(a.out, "w").write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
