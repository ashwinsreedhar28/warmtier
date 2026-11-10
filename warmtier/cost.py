"""Cost model v0: when does promoting read-only repo knowledge pay?

    python -m warmtier.cost data/silver/nebius --out results/cost_v0.json

Everything is in GPU-seconds on one GPU class, so price cancels out.

Measured inputs (from the silver tables): per run, how much read-only
repo-file content the agent pulled into context, how many LLM calls carried
it, and how many calls it spent opening those files.

What a promoted tier (cartridge or LoRA built per repo) could save per run:
  no_cache      prefill of that content on every call that carries it
  prefix_cache  prefill of that content once (an ideal in-session prefix cache
                already removes the repeats)
  +calls        additionally the decode for the calls spent opening the files,
                if the agent no longer needs to open them

Break-even runs per build:      N* = C_build / saved_per_run
Runs per day needed:            N* / rebuild_days
Tolerable extra failure rate:   q* = saved_per_run / full_run_cost
  (if promotion makes more than q* of runs fail and be redone, it loses even
  with a free build)

ASSUMED constants (not measured here; Phase 1 calibrates them on vLLM):
  chars_per_token   4.0. The dataset card's token averages imply 4.25-4.55 if
                    they count the whole trajectory.
  r_prefill         prefill tokens/s per GPU, swept.
  r_decode          batched decode tokens/s per GPU.
  rebuild_days      30. Git drift (results/drift_git_4repos.json) puts 59-81%
                    of frequently changed files still unchanged at 30 days.
SOURCED constant:
  C_build           Cartridges (arXiv 2506.06266): "~30 minutes on a single
                    8xH100 node (for LLAMA-8B)", described as unoptimized, for
                    100k-484k-token corpora = 4 GPU-hours = 14,400 GPU-s.
                    The cheaper rows are hypothetical.
LoRA is not priced: no sourced per-repo training cost yet.
"""
from __future__ import annotations

import argparse
import json

import duckdb

C_BUILD = {"cartridge_as_published": 14_400.0, "10x_cheaper_hypothetical": 1_440.0,
           "hypernetwork_scale_hypothetical": 60.0}
R_PREFILL = (5_000.0, 10_000.0, 20_000.0)
R_DECODE = 2_000.0
CHARS_PER_TOKEN = 4.0
REBUILD_DAYS = 30.0


def measured(silver: str) -> dict:
    R = f"read_parquet('{silver}/runs/*.parquet')"
    V = f"read_parquet('{silver}/views/*.parquet')"
    rel = duckdb.sql(f"""
        with ed as (select distinct run, file from {V} where stale_file_call is not null),
        ro as (select v.run, sum(v.chars) chars, sum(v.chars*v.carried) char_calls, count(*) as "views"
               from {V} v left join ed using (run,file)
               where not v.scratch and ed.file is null group by 1)
        select count(*) runs,
               avg(coalesce(ro.chars,0)) ro_chars, avg(coalesce(ro.char_calls,0)) ro_char_calls,
               avg(coalesce(ro.views,0)) ro_views,
               avg((ro.run is not null)::int) share_runs_with_ro_views,
               avg(r.prefill_chars_no_cache) run_prefill_char_calls,
               avg(r.system_chars+r.task_chars+r.assistant_chars+r.observation_chars) run_unique_chars,
               avg(r.assistant_chars) run_assistant_chars, avg(r.n_calls) run_calls
        from {R} r left join ro using (run)""")
    return dict(zip(rel.columns, rel.fetchall()[0]))


def model(m: dict) -> dict:
    cpt = CHARS_PER_TOKEN
    asst_tok_per_call = m["run_assistant_chars"] / m["run_calls"] / cpt
    rows = []
    for r in R_PREFILL:
        decode_run = m["run_assistant_chars"] / cpt / R_DECODE
        base = {
            "no_cache": (m["ro_char_calls"] / cpt / r, m["run_prefill_char_calls"] / cpt / r + decode_run),
            "prefix_cache": (m["ro_chars"] / cpt / r, m["run_unique_chars"] / cpt / r + decode_run),
        }
        for name, (saved, full) in base.items():
            for with_calls in (False, True):
                s = saved + (m["ro_views"] * asst_tok_per_call / R_DECODE if with_calls else 0.0)
                row = {"r_prefill": r, "baseline": name, "avoided_calls_counted": with_calls,
                       "saved_gpu_s_per_run": round(s, 4), "full_run_gpu_s": round(full, 3),
                       "tolerable_extra_failure_rate": round(s / full, 4)}
                for k, c in C_BUILD.items():
                    row[f"breakeven_runs__{k}"] = round(c / s)
                    row[f"runs_per_day__{k}"] = round(c / s / REBUILD_DAYS, 1)
                rows.append(row)
    return {"assumed": {"chars_per_token": cpt, "r_decode": R_DECODE, "rebuild_days": REBUILD_DAYS},
            "c_build_gpu_s": C_BUILD,
            "measured_per_run": {k: round(v, 3) for k, v in m.items()},
            "rows": rows}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("silver")
    ap.add_argument("--out")
    a = ap.parse_args()
    res = model(measured(a.silver))
    if a.out:
        open(a.out, "w").write(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res["measured_per_run"], indent=1))
    hdr = ("r_prefill", "baseline", "calls", "saved_s", "q*", "N* published", "runs/day", "N* 10x", "N* hyper")
    print(("{:>9} {:>13} {:>5} {:>8} {:>7} {:>13} {:>9} {:>8} {:>9}").format(*hdr))
    for r in res["rows"]:
        print("{:>9.0f} {:>13} {:>5} {:>8.3f} {:>7.4f} {:>13} {:>9} {:>8} {:>9}".format(
            r["r_prefill"], r["baseline"], "yes" if r["avoided_calls_counted"] else "no",
            r["saved_gpu_s_per_run"], r["tolerable_extra_failure_rate"],
            r["breakeven_runs__cartridge_as_published"], r["runs_per_day__cartridge_as_published"],
            r["breakeven_runs__10x_cheaper_hypothetical"], r["breakeven_runs__hypernetwork_scale_hypothetical"]))


if __name__ == "__main__":
    main()
