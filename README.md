# warmtier

Where should an agent's knowledge live: as text in context, as cached KV, as a
trained KV cache (cartridge), or in weights (LoRA)? Published work says the
answer depends on how often a piece of context is reused and how long it stays
valid, and does not measure either on real agent workloads. warmtier measures
both, then prices the tiers.

Status: Phase 0 (measurement, CPU only). Nothing here is a result about tier
selection yet.

## What exists

- `warmtier/drift.py`: valid life of file content from git history.
- `warmtier/sweagent.py`: bronze to silver for SWE-agent trajectories
  (per-message chunks, per-file views with staleness, per-run totals).
- `warmtier/gold.py`: gold metrics from the silver tables (DuckDB).
- `warmtier/cost.py`: cost model v0 (break-even runs per build).
- `warmtier/traj.py`: small in-memory version for other dataset shapes.

## First measurement: upstream drift is slow (Nov 10, 2026)

`python -m warmtier.drift <repo>...`, 3-year window, `*.py`, first-parent
history of the default branch. S(D) is the probability that a file read at a
random moment is unchanged D later. "hot" weights each file by how often it
changed, a proxy for the files agents get sent to fix.

| repo | files | S(1d) hot | S(7d) hot | S(30d) hot | half-life hot | half-life uniform |
|---|---|---|---|---|---|---|
| django | 2,967 | 0.987 | 0.920 | 0.735 | 81 d | 651 d |
| sympy | 1,647 | 0.991 | 0.946 | 0.812 | 120 d | 254 d |
| scikit-learn | 1,109 | 0.977 | 0.862 | 0.592 | 43 d | 156 d |
| pytest | 283 | 0.976 | 0.863 | 0.601 | 44 d | 150 d |

Full numbers: `results/drift_git_4repos.json`.

What this does and does not show:
- On these four repos, a file an agent reads has at least a 97.6% chance of
  being unchanged upstream a day later, even among frequently changed files.
- It covers upstream changes on the default branch of mature open-source
  projects only. It says nothing about the agent's own edits inside a session,
  about feature branches, or about private monorepos, all of which may drift
  faster.
- "hot" is a proxy. The real weighting is the files agents actually read,
  which needs the trajectory data.

## Second measurement: reuse and in-session staleness (Nov 10, 2026)

80,036 SWE-agent runs (nebius/SWE-agent-trajectories; 3,591 instances, 1,276
repos, Llama 3.1 8B/70B/405B). Reproduce:

    python -m warmtier.sweagent 'data/nebius/data/*.parquet' --out data/silver/nebius
    python -m warmtier.gold data/silver/nebius --out results/reuse_nebius.json

Shares are of characters. "Prefill" is the append-only, no-cache prefill
volume: each message counted once per LLM call that carries it.

1. File views dominate. Observations that show file content are 81.0% of
   observation characters and 41.9% of prefill.
2. Agents invalidate their own context fast. 55.5% of file views are followed
   by a successful edit to the same file (47.6% if the edit must touch the
   lines shown). Among those, the view stays valid for a median of 2 LLM
   calls (p90: 5), with a median of 13 calls still to go in the run.
3. Stale file content is 15.1% (window level) to 18.3% (file level) of all
   prefill in an append-only replay.
4. Views split into three classes with very different behaviour:

   | class | share of prefill | stale share (window to file level) |
   |---|---|---|
   | repo file, edited in the run | 20.7% | 47.8% to 62.6% |
   | repo file, read only | 13.4% | 0% |
   | scratch file the agent created | 7.8% | 66.2% to 67.8% |

5. Across tasks, reuse is by file identity, not by bytes. 0.6% of file-view
   characters are byte-identical across different instances, yet 64.6% of
   repo-file view characters are on files that at least one other instance in
   the same repo also viewed (88.7% in the 138 repos with 5+ instances).
   Exact-prefix KV caching cannot reuse what is not byte-identical.
6. Byte-identical reuse across runs is the system prompt (8.1% of unique
   characters, one prompt shared by all runs), the task text within an
   instance (6.8%), and observations within an instance (13.1%).

Limits:
- Append-only replay. SWE-agent elides old observations when it renders
  prompts, and the dataset stores trajectories, not rendered prompts.
- Characters, not tokens.
- Instances in one repo sit at different base commits, often far apart in
  time, which lowers byte-identical overlap relative to a fleet working on
  one commit. The 0.6% is a floor for that reason.
- Edits made through shell commands are not detected. Failed-edit previews
  count as views.
- One harness, one model family, 16.7% of runs resolved.

## Third: cost model v0 (Nov 10, 2026)

`python -m warmtier.cost data/silver/nebius --out results/cost_v0.json`

Question: if a repo's read-only file knowledge were served from a trained
tier built once per repo, how many agent runs does one build need to pay for
itself on prefill? Units are GPU-seconds, so price cancels.

Measured per run (mean over 80,036 runs): 8,945 characters of read-only
repo-file views, carried for 160,378 character-calls, from 2.7 views; only
33.8% of runs have any.

Sourced: one cartridge build = 4 H100-hours (Cartridges, arXiv 2506.06266:
about 30 minutes on 8 H100s for Llama-8B, described as unoptimized).
Assumed, to be calibrated in Phase 1: 4.0 characters per token, prefill
5,000 to 20,000 tokens/s per GPU, batched decode 2,000 tokens/s, rebuild
every 30 days.

| baseline the tier competes with | break-even runs per build | runs/day on one repo | extra failure rate it can afford |
|---|---|---|---|
| no cache | 1,796 to 7,183 | 60 to 239 | 12.2% to 13.1% |
| ideal in-session prefix cache | 32,196 to 128,785 | 1,073 to 4,293 | 5.0% to 9.9% |
| prefix cache, also counting the file-open calls avoided | 23,970 to 54,275 | 799 to 1,809 | 11.9% to 13.4% |

Reading:
- Against a working prefix cache, a cartridge at its published build cost
  needs on the order of a thousand or more runs a day on one repo. That is a
  large fleet on a monorepo, not a typical workload.
- Build cost is the lever. At a tenth of the published cost the bar drops to
  roughly 80 to 430 runs a day; at hypernetwork scale (hypothetical) to under
  20.
- Quality is the other constraint: the tier loses if it makes more than
  about 5% to 13% of runs fail, even with a free build.
- Prefill is not the only possible benefit (time to first token, context
  headroom, fewer exploration calls). Those are not priced here.

This rests on one harness in which agents read few files. A harness that
reads more would move every number.

## Next

1. A second harness with rendered prompts and heavier file reading
   (nvidia/Open-SWE-Traces, one split).
2. Weight git drift by the files agents actually read.
3. Phase 1: calibrate the assumed throughputs on vLLM and measure quality.

## Related work this builds on

MemOS (arXiv 2507.03724), Cartridges (arXiv 2506.06266), What It Costs to
Compose, Rebuild, and Correct Precomputed Memory (arXiv 2608.30647), The
Missing Memory Hierarchy (arXiv 2603.09023), ForkKV (arXiv 2604.06370).
