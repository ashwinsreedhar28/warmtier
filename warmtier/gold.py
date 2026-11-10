"""Gold metrics from the SWE-agent silver tables (see sweagent.py).

    python -m warmtier.gold data/silver/nebius --out results/reuse_nebius.json

All shares are of characters. "prefill" is the append-only, no-cache prefill
volume: each message counted once per LLM call that carries it.
"""
from __future__ import annotations

import argparse
import json

import duckdb


def one(sql: str) -> dict:
    rel = duckdb.sql(sql)
    return dict(zip(rel.columns, rel.fetchall()[0]))


def many(sql: str) -> list[dict]:
    rel = duckdb.sql(sql)
    return [dict(zip(rel.columns, r)) for r in rel.fetchall()]


def compute(silver: str) -> dict:
    R = f"read_parquet('{silver}/runs/*.parquet')"
    C = f"read_parquet('{silver}/chunks/*.parquet')"
    V = f"read_parquet('{silver}/views/*.parquet')"
    duckdb.sql(f"""create or replace temp table vv as
        with firstv as (select run, file, min(call) c0 from {V} group by 1,2),
        cr as (select distinct c.run, f.file from firstv f join {C} c
               on c.run=f.run and c.call=f.c0 and c.kind='observation' and c.cmd='create'),
        ed as (select distinct run, file from {V} where stale_file_call is not null)
        select v.*, (cr.file is not null) scratch, (ed.file is not null) edited, r.n_calls
        from {V} v left join cr using (run,file) left join ed using (run,file)
        join {R} r using (run)""")
    out = {}
    out["dataset"] = one(f"""select count(*) runs, count(distinct instance) instances, count(distinct repo) repos,
        sum(resolved::int) resolved, median(n_calls) median_calls, round(avg(n_calls),2) mean_calls from {R}""")
    out["prefill"] = one(f"""select sum(system_chars+task_chars+assistant_chars+observation_chars) unique_chars,
        sum(prefill_chars_no_cache) prefill_chars_no_cache,
        round(sum(prefill_chars_no_cache)/sum(system_chars+task_chars+assistant_chars+observation_chars),2) amplification,
        round(median(prefill_chars_no_cache/(system_chars+task_chars+assistant_chars+observation_chars)),2) amplification_median_run
        from {R}""")
    out["by_kind"] = many(f"""select kind, round(100.0*sum(chars)/(select sum(chars) from {C}),1) pct_unique_chars,
        round(100.0*sum(chars*carried)/(select sum(chars*carried) from {C}),1) pct_prefill
        from {C} group by 1 order by 3 desc""")
    out["file_views"] = one(f"""select count(*) views,
        round(100.0*sum(chars)/(select sum(chars) from {C} where kind='observation'),1) pct_of_observation_chars,
        round(100.0*sum(chars*carried)/(select sum(chars*carried) from {C}),1) pct_prefill,
        round(100.0*avg((stale_file_call is not null)::int),1) pct_views_made_stale_file_level,
        round(100.0*avg((stale_window_call is not null)::int),1) pct_views_made_stale_window_level,
        median(stale_file_call-call) filter (stale_file_call is not null) median_valid_life_calls,
        quantile_cont(stale_file_call-call,0.9) filter (stale_file_call is not null) p90_valid_life_calls,
        median(carried) median_calls_remaining_after_view from {V}""")
    out["stale_share_of_prefill"] = one(f"""select
        round(100.0*sum(stale_file_char_calls)/sum(prefill_chars_no_cache),2) file_level_pct,
        round(100.0*sum(stale_window_char_calls)/sum(prefill_chars_no_cache),2) window_level_pct,
        round(100.0*avg((n_edits>0)::int),1) pct_runs_with_successful_edit from {R}""")
    out["view_classes"] = many(f"""select case when scratch then 'scratch file (agent created it)'
          when edited then 'repo file, edited in run' else 'repo file, read only' end class,
        count(*) views,
        round(100.0*sum(chars*carried)/(select sum(prefill_chars_no_cache) from {R}),1) pct_prefill,
        round(100.0*coalesce(sum(chars*(n_calls-stale_file_call)) filter (stale_file_call is not null),0)/sum(chars*carried),1) stale_pct_file_level,
        round(100.0*coalesce(sum(chars*(n_calls-stale_window_call)) filter (stale_window_call is not null),0)/sum(chars*carried),1) stale_pct_window_level
        from vv group by 1 order by 3 desc""")
    out["cross_run_identical_text"] = many(f"""
        with g as (select sha, any_value(kind) kind, any_value(chars) chars, count(distinct run) runs,
                   count(distinct instance) insts, count(distinct repo) repos from {C} where chars>=200 group by sha)
        select case when repos>1 then 'cross_repo' when insts>1 then 'cross_instance_same_repo' else 'same_instance' end scope,
               kind, round(100.0*sum(chars*(runs-1))/(select sum(chars) from {C}),2) pct_of_all_unique_chars
        from g where runs>1 group by 1,2 having sum(chars*(runs-1))*1000 > (select sum(chars) from {C}) order by 3 desc""")
    out["cross_instance_file_identity"] = one("""
        with fi as (select repo,file,count(distinct instance) insts from vv where not scratch group by 1,2)
        select round(100.0*sum(chars) filter (fi.insts>1)/sum(chars),1) pct_repo_view_chars_on_files_seen_by_2plus_instances,
               round(100.0*sum(chars) filter (fi.insts>=3)/sum(chars),1) pct_3plus_instances
        from vv join fi using (repo,file) where not scratch""")
    out["cross_instance_file_identity_repos_with_5plus_instances"] = one("""
        with fi as (select repo,file,count(distinct instance) insts from vv where not scratch group by 1,2),
        big as (select repo from vv group by 1 having count(distinct instance)>=5)
        select count(distinct repo) repos,
               round(100.0*sum(chars) filter (fi.insts>1)/sum(chars),1) pct_2plus_instances,
               round(100.0*sum(chars) filter (fi.insts>=3)/sum(chars),1) pct_3plus_instances
        from vv join fi using (repo,file) where not scratch and repo in (select repo from big)""")
    out["cross_instance_identical_view_text"] = one(f"""
        with g as (select sha, any_value(chars) chars, count(*) n, count(distinct run) runs, count(distinct instance) insts from {V} group by 1)
        select round(100.0*sum(chars*(runs-1))/sum(chars*n),1) pct_view_chars_duplicated_across_runs,
               round(100.0*sum(chars*(insts-1))/sum(chars*n),1) pct_view_chars_duplicated_across_instances from g""")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("silver")
    ap.add_argument("--out")
    a = ap.parse_args()
    text = json.dumps(compute(a.silver), indent=2, default=float)
    if a.out:
        open(a.out, "w").write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
