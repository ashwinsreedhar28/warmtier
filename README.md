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
- `warmtier/claudecode.py`: the same silver tables for Claude Code style
  traces (nvidia/Open-SWE-Traces).
- `warmtier/gold.py`: gold metrics from the silver tables (DuckDB).
- `warmtier/ladder.py`: cross-run reuse seen by prefix-chained hashing versus
  content addressing.
- `warmtier/gate.py`: gate experiment, file KV reused out of context (needs a GPU).
- `warmtier/ledger.py`: edit-aware ledger a harness can call: finds stale file
  views in a message list, plans what to drop, tags notes with file versions.
- `warmtier/writer.py`: trains a writer adapter for file KV (needs a GPU).
- `warmtier/evict.py`: stale-view eviction policies replayed under a prefix cache.
- `warmtier/writer.py`, `warmtier/writer_diag.py`: writer adapter for file KV, and its audit.
- `warmtier/splice.py`: can a stale view be cut out of cached KV in place? (needs a GPU)
- `warmtier/team.py`: reuse among runs of the same task, a stand-in for an agent team.
- `warmtier/blocks.py`: reuse when the unit is a line range instead of a whole view.
- `warmtier/floor.py`: noise floor for the gate metrics.
- `warmtier/arch.py`: per-token KV on attention-only versus hybrid models.
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
   | scratch file the agent created | 7.8% | 66.1% to 67.8% |

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

## Fourth: a second harness (Nov 10, 2026)

133,376 runs of a Claude Code style harness (Read, Edit, Bash) driven by
Qwen3.8 models, from the claudecode split of nvidia/Open-SWE-Traces (39,611
instances, 3,715 repos, 47.8% resolved).

    python -m warmtier.claudecode 'data/open-swe-traces/data/claudecode/**/*.parquet' --out data/silver/claudecode
    python -m warmtier.gold data/silver/claudecode --out results/reuse_claudecode.json
    python -m warmtier.cost data/silver/claudecode --out results/cost_v0_claudecode.json

| | SWE-agent, Llama 3.1 | Claude Code style, Qwen3.8 |
|---|---|---|
| runs | 80,036 | 133,376 |
| LLM calls per run (median) | 17 | 32 |
| prefill / unique content | 19.9x | 38.2x |
| file views, share of prefill | 41.9% | 48.1% |
| views later invalidated by an edit (window to file level) | 47.6% to 55.5% | 29.7% to 37.0% |
| valid life of those views (median calls) | 2 | 7 |
| stale file content, share of prefill | 15.1% to 18.3% | 12.7% to 14.5% |
| read-only repo files, share of prefill | 13.4% | 25.3% |
| runs with any read-only repo view | 33.8% | 88.7% |
| view text byte-identical across instances | 0.6% | 5.4% |
| repo-file view text on files another instance also viewed | 64.6% | 83.8% |
| cartridge break-even vs prefix cache (runs/day on one repo) | 1,073 to 4,293 | 412 to 1,647 |
| same, counting avoided file-open calls | 799 to 1,809 | 309 to 708 |
| extra failure rate the tier can afford (vs prefix cache) | 5.0% to 9.9% | 6.3% to 13.2% |

What holds across both harnesses:
- File views are 42% to 48% of prefill, and 13% to 18% of all prefill is
  file content the agent has already edited.
- Cross-task reuse is by file identity (65% to 84%), not by bytes (0.6% to
  5.4%).
- At the published cartridge build cost, promotion needs hundreds to
  thousands of runs a day on one repo to pay on prefill against a prefix
  cache.

What changes: the Claude Code style harness reads about 2.6 times as much
read-only repo content per run, which lowers the break-even by about the
same factor.

Extra limits for this dataset: file views read through Bash are counted only
for single-file `cat`, `head`, `tail` and `sed -n` commands, and Bash edits
only for `sed -i` and redirects, so views and edits are both undercounted.
`reasoning_content` is not counted as carried context. The traces are
teacher-model data generated for training, not production traffic.

## Fifth: what prefix-chained caching misses, and reasoning (Nov 10, 2026)

`python -m warmtier.ladder 'data/silver/<set>/chunks/*.parquet'`

Share of all unique characters that a sharing scheme would not have to
prefill again across runs, at whole-message granularity:

| scheme | SWE-agent | Claude Code style |
|---|---|---|
| prefix chain (a message counts only if everything before it matches; what hash-chained block caches reuse) | 15.0% | 3.4% |
| content addressed (same text in another run at any position) | 30.4% | 18.4% |

Content addressing sees 2.0x and 5.4x what the prefix chain sees. It is an
upper bound for position-free KV reuse, which is not free: KV reused out of
position has to be repaired. In the Claude Code style traces the system
prompt carries a date and git status, so the chain breaks almost at once.

Reasoning (Claude Code style traces only): `reasoning_content` is 72.7% of
decoded characters. Keeping it in history would add 59.5% to unique context;
stripping it means re-prefilling each turn's visible output once, 22.4% of
unique context.

## Sixth: gate experiment, does a file's KV survive reuse? (Nov 10, 2026)

`python -m warmtier.gate` on 100 real file reads from the Claude Code style
traces, on two standard-attention models (one A40 GPU). For each read the
model scores the assistant's next turn with the file's KV computed normally
("full"), computed as if the file were encoded alone at the same offset
("iso", exact via an attention mask), with some file tokens recomputed in
context, and with the file hidden ("nofile"). "Retained" is the share of the
file's benefit (drop in negative log-likelihood from nofile to full) that
survives; brackets are bootstrap 95% intervals.

| condition | Qwen3-4B retained | Qwen3-8B retained |
|---|---|---|
| iso, no repair | 0.15 [-0.21, 0.38] | 0.37 [0.12, 0.53] |
| recompute first 16 file tokens | 0.75 [0.62, 0.93] | 0.36 [0.20, 0.49] |
| recompute first 64 file tokens | 0.90 [0.77, 1.08] | 0.40 [0.23, 0.53] |
| recompute an evenly spaced 15% | 0.84 [0.70, 0.98] | 0.53 [0.39, 0.64] |

Reading:
- Reusing a code file's KV with no repair loses most of the file's benefit
  on both models.
- Cheap repair does not transfer: recomputing the head of the file recovers
  about 90% on the 4B model and about 40% on the 8B model.
- So training-free, lightly repaired reuse does not pass the gate. The next
  candidates are deviation-based token selection (CacheBlend-style) and a
  trained writer adapter (SemPIC-style).

Limits: the metric is teacher-forced on text another model wrote, and the
file's benefit is small on it (0.14 nats per token on average, negative in
17% to 23% of samples), so "retained" is a ratio of small numbers. KL to the
full distribution falls only 27% to 43% versus nofile under any repair tried.
Repair tokens were chosen naively (head or evenly spaced), not by deviation.
Qwen3.5-4B could not be tested this way: it is a hybrid (24 of 32 layers are
linear attention), and the script's self-check correctly refused it.

## Seventh: gate rerun with deviation-based repair and generated tool calls (Nov 10, 2026)

`python -m warmtier.gate ... --n 60 --gen 64`, same two models, one A40.
Results: `results/gate2_qwen3_4b.json`, `results/gate2_qwen3_8b.json`.
"KL" is the mean divergence of the next-token distribution from full prefill
over the real next turn (0 = identical). "Identical call" is how often the
greedily generated next tool call is character-identical to the one generated
under full prefill (56 samples with a reference call). Brackets are bootstrap
95% intervals.

| condition | 4B KL | 4B identical call | 8B KL | 8B identical call |
|---|---|---|---|---|
| file hidden | 0.232 [0.172, 0.300] | 25% [14, 38] | 0.190 [0.143, 0.246] | 34% [21, 46] |
| plain reuse, no repair | 0.149 [0.125, 0.176] | 23% [12, 36] | 0.114 [0.098, 0.133] | 34% [21, 46] |
| recompute first 64 tokens | 0.164 [0.133, 0.199] | 34% [21, 46] | 0.121 [0.099, 0.147] | 46% [34, 59] |
| recompute evenly spaced 15% | 0.124 [0.104, 0.146] | 36% [23, 48] | 0.101 [0.082, 0.122] | 46% [34, 59] |
| recompute evenly spaced 30% | 0.101 [0.082, 0.122] | 45% [32, 57] | 0.074 [0.062, 0.089] | 54% [41, 66] |
| recompute top 15% by deviation | 0.142 [0.116, 0.171] | 39% [27, 54] | 0.108 [0.089, 0.132] | 54% [41, 66] |
| recompute top 30% by deviation | 0.118 [0.098, 0.142] | 45% [32, 57] | 0.086 [0.072, 0.103] | 48% [36, 61] |

Reading:
- Plain reuse generates the same call as full prefill no more often than
  hiding the file does.
- Recomputing up to 30% of the file's tokens still yields the same call as
  full prefill only about half the time, and leaves two thirds of plain
  reuse's KL in place.
- Choosing tokens by deviation is no better than even spacing on either
  model. Recomputing only the first 64 tokens does not reduce KL at all.

An earlier version of this table led with "retained" (share of the file's
likelihood benefit kept). That metric was withdrawn after the adapter audit
in the Twelfth section: it rewards any change that flattens predictions, and
it had made deviation-based repair look near-perfect (0.99) when it is not.

The comparison against the call the real agent made did not work as a
metric: under full prefill these models pick the same tool as the real agent
39% to 43% of the time and never the identical call, and hiding the file
does not lower that. They are weak stand-ins for the agent that produced the
traces, and the forced no-reasoning format departs from how it ran.

## Eighth: what evicting stale views costs under a prefix cache (Nov 10, 2026)

`python -m warmtier.evict <chunks glob> <views glob>`, CPU only, all runs of
both datasets. Each run is replayed call by call. "Prefill" is what the
engine must actually prefill with a prefix cache (new tool output, plus
anything redone after an eviction). "Context" is the context length summed
over calls: what decode attends over and what KV memory holds. Both are
relative to an append-only harness; staleness at window level.

| policy | SWE-agent prefill | SWE-agent context | Claude Code style prefill | Claude Code style context |
|---|---|---|---|---|
| evict a stale view at once | +24.7% | -15.1% | +100.4% | -12.7% |
| evict when stale content reaches 10% of context | +24.2% | -13.8% | +49.4% | -10.9% |
| evict when it reaches 25% | +13.8% | -9.7% | +16.4% | -7.1% |
| splice it out of the cached KV (no re-prefill) | 0% | -15.1% | 0% | -12.7% |

Reading:
- Dropping stale file views shrinks context by 13% to 15%, but with an
  ordinary prefix cache every drop invalidates the cache after that point.
  Evicting eagerly costs 25% to 100% more prefill.
- Batching evictions helps but gives back part of the context saving.
- The saving is only free if the engine can remove a span from cached KV in
  place. That is the capability an edit-aware layer needs from the engine.

Limits: characters, not tokens; no stub is left in place of an evicted view;
the trade between prefill and carried context is not converted to GPU time
here; whether dropping stale views changes task success is untested.

## Ninth: agents on the same task, a stand-in for a team (Nov 10, 2026)

`python -m warmtier.team <chunks glob> <views glob>`, CPU only. A "team" is k
runs of the same task instance (same repo, base commit and issue), taken in a
fixed random order. Reuse counts only text an earlier teammate already
prefilled; a run's own repeats are not counted. "Body" is everything after the
system prompt, tool list and task text, which every teammate shares by
construction.

| | SWE-agent k=2 | k=4 | k=8 | k=16 | Claude Code style k=2 | k=4 |
|---|---|---|---|---|---|---|
| teams | 3,581 | 3,387 | 2,929 | 1,811 | 38,251 | 21,875 |
| body reusable by prefix chain | 0.01% | 0.03% | 0.05% | 0.07% | 0.01% | 0.04% |
| body reusable by content hash | 3.65% | 7.42% | 11.14% | 15.34% | 7.16% | 14.17% |
| file views: same bytes as a teammate saw | 6.33% | 12.63% | 18.60% | 24.27% | 16.43% | 31.41% |
| file views: same file as a teammate saw | 24.80% | 42.17% | 54.08% | 62.03% | n/a | n/a |

Reading:
- After the shared header, prefix-chained caching reuses nothing across
  teammates (under 0.1% of body text). Runs diverge at the first model turn
  and never line up again.
- Content addressing finds 4% to 15% of body text, growing with team size.
  For file views alone it is 6% to 31%.
- On SWE-agent the gap between "same file" and "same bytes" is large (62% vs
  24% at k=16): teammates open the same files but mostly not the same bytes.
  SWE-agent shows files in 100-line windows, so two agents rarely land on the
  same window; how much of the gap that explains is not measured. A store
  keyed on whole-view bytes misses this overlap; one keyed on file version
  plus line range might not.

Limits: these runs never saw each other, so this is overlap among independent
attempts, not a coordinated team that splits work (which would overlap less).
The Claude Code style data has at most 4 runs per task, from different models.
Message-level matching; the "same file" row is not available for that dataset
in the tables used here. Reuse found by content hash is only useful if KV can
be reused out of position, which the gate experiment says needs training.
Results: `results/team_nebius.json`, `results/team_claudecode.json`. The k=2
SWE-agent rows were recomputed independently in DuckDB and match.

## Tenth: what hybrid models change (Nov 10, 2026)

`python -m warmtier.arch`. Only attention layers keep keys and values per
context token. Qwen3.5-4B has 8 attention layers out of 32; the other 24 are
linear attention with a fixed-size state.

| model | attention layers | KV per token (fp16) | KV at 100k tokens | freed by stale-view eviction |
|---|---|---|---|---|
| Qwen3-4B | 36 of 36 | 147,456 B | 14.75 GB | 1.87 to 2.23 GB |
| Qwen3-8B | 36 of 36 | 147,456 B | 14.75 GB | 1.87 to 2.23 GB |
| Qwen3.5-4B | 8 of 32 | 32,768 B | 3.28 GB | 0.42 to 0.49 GB |

Reading:
- Per-token KV on the hybrid is 22% of the attention-only model of the same
  width, so the memory side of eviction is worth about a fifth as much.
- A linear-attention layer cannot give a span back: its state has absorbed the
  text. Cutting a stale view out of a hybrid's cache removes it from the 8
  attention layers only, unless everything after the view is replayed.
- The linear state is an estimated 25 MB per sequence (from config
  dimensions, not checked against the implementation), about 770 tokens'
  worth of attention KV, and it has to be snapshotted for any prefix reuse.

Limits: memory only, no speed measurements. The "freed" column applies the
character-level context saving from the eviction replay (12.7% to 15.1%) to
tokens. Config values read from Hugging Face on 2026-11-10.

## Eleventh: cutting a stale view out of cached KV (Nov 10, 2026)

`python -m warmtier.splice <traces glob> --model ... --n 120 --gen 64`, one
GPU. The eviction replay says dropping stale views is only free if the engine
can cut a span out of cached KV without prefilling what follows again. This
tests what that does to the model. 120 real Claude Code style traces per
model, each cut at the first model call after an edit made an earlier file
view stale (median about 560 tokens of stale view and 550 tokens encoded
after it).

- full: nothing dropped.
- text eviction: the view replaced by a one-line stub and everything after it
  prefilled again. The reference for "the view is gone".
- splice: the view cut from the cache; what was encoded after it keeps the KV
  it was given while the view was visible. No re-prefill.
- recompute, no stub: everything after the view re-encoded without it, same
  positions. Separates "stale KV" from "no stub, gap in positions".

Mean over samples, 95% bootstrap interval in brackets:

| | Qwen3-4B | Qwen3-8B | Qwen3.5-4B (hybrid) |
|---|---|---|---|
| NLL of the real next turn, full | 2.378 | 2.051 | 1.375 |
| text eviction minus full | +0.027 [0.004, 0.053] | +0.036 [0.016, 0.060] | +0.042 [0.024, 0.065] |
| splice minus text eviction | +0.027 [0.016, 0.038] | +0.010 [0.002, 0.018] | -0.001 [-0.004, 0.002] |
| KL from text eviction: full | 0.138 | 0.117 | 0.113 |
| KL from text eviction: splice | 0.037 | 0.027 | 0.017 |
| KL from text eviction: recompute, no stub | 0.015 | 0.009 | 0.007 |
| next call, same tool as text eviction: full / splice / recompute | 88% / 87% / 88% | 88% / 88% / 93% | 84% / 85% / 95% |
| next call identical to the full run's: text eviction | 43% | 39% | 37% |
| next call identical to the full run's: splice | 68% | 64% | 62% |

Reading:
- Splicing does not break the model. The real next turn is at most 0.027 nats
  less likely than under text eviction (about 1%), and the next tool matches
  text eviction as often as the un-evicted run does.
- It is a partial eviction. In next-token terms the spliced model sits close
  to text eviction (KL 0.02 to 0.04, against 0.11 to 0.14 for no eviction).
  But the call it actually generates is identical to the un-evicted run's 62%
  to 68% of the time, against 37% to 43% for text eviction. The stale view
  keeps steering the model through the KV of what was encoded after it.
- So splicing frees the memory and attention cost of a stale view at once, but
  does not reliably remove its influence. Whether that influence hurts or
  helps task success is not measured here.
- Hybrid: the masks reach only the 8 attention layers of Qwen3.5-4B; the 24
  linear layers keep the span in their state (on random tokens, masking a span
  changes later hidden states as much as leaving it visible). Even so, on real
  traces "recompute, no stub" lands as close to text eviction as on the
  attention-only models (KL 0.007). In this test, what the model takes from a
  file view reaches the next turn through the attention layers.

Limits: one decision point per trace, the first call after the edit; later
calls are not followed. Teacher-forced scores and one greedy 64-token call per
condition, no sampling, no task outcomes. The models did not produce these
traces (a Qwen3.8-27B agent did). Prefix truncated to 2,500 tokens. The real
next turn was written by an agent that still had the stale view, which favours
"full" on the NLL row. For 1,350 candidate samples from one trace file, an
independent walk of the raw messages finds an edit to the shown text in 80%,
a shell rewrite of the file in 16%, an edit elsewhere in the file in 3%, and
no edit in 1 case.
Results: `results/splice_qwen3_4b.json`, `results/splice_qwen3_8b.json`,
`results/splice_qwen35_4b_hybrid.json`.

## Twelfth: a trained writer adapter for file KV (Nov 10, 2026)

`python -m warmtier.writer` then `python -m warmtier.gate --adapter`. A rank-8
LoRA on the attention projections (5.9M parameters), active only on the file's
tokens, trained so that a file encoded without its prefix gives the same
next-token distribution as full prefill. 2,000 training samples, 2 epochs,
97 minutes on one A40; validation KL 0.155 to 0.043. Evaluated on the same 60
held-out samples as the gate rerun (Qwen3-4B; 56 with a generated call).

| condition | KL from full | same top token | next call: same tool as full | identical to full |
|---|---|---|---|---|
| plain reuse | 0.149 [0.125, 0.175] | 90.9% | 77% | 23% |
| recompute 30%, evenly spaced | 0.101 [0.082, 0.122] | 94.4% | 89% | 45% |
| writer adapter, no recompute | 0.034 [0.029, 0.039] | 95.0% | 93% | 46% |

Reading:
- The adapter cuts the distribution gap of plain reuse by 77% with nothing
  recomputed, and is 3.0x closer to full prefill than the best training-free
  repair.
- On the generated next call it ties that repair (46% vs 45% identical to
  full, intervals about +/-13 points) and does not beat it.

Audit note on NLL (resolved Nov 10): under the adapter the real next turn is
more likely than under full prefill (NLL 2.573 vs 2.716, lower in 59 of 60
samples). `python -m warmtier.writer_diag` shows this is a calibration effect,
not recovered or leaked information:
- The plain model is overconfident on these traces: re-tempering its own
  predictions at T=1.1 lowers NLL from 2.734 to 2.561 per token.
- The adapter gives the same NLL gain with full attention and no isolation
  (2.610 per token), so the gain does not come from repairing reuse.
- All of the gain is on tokens where the real token is not the plain model's
  top choice (6.13 to 5.78); where it is the top choice the adapter is slightly
  worse (0.143 to 0.166). The largest part sits in the first 8 tokens of the
  turn (9.32 to 7.48 per token).
- The distillation loss KL(full || adapter) is mass-covering, so the adapted
  model spreads probability slightly wider (entropy 0.763 to 0.790).
So NLL of the real turn, and the "retained" figure built on it, are not valid
measures of reuse quality: anything that flattens predictions scores well on
them. The Seventh section's table was redone with KL for that reason; the
"retained" column in the Sixth section is kept as first recorded and should
not be quoted.

On 60 samples from a different agent's traces (qwen38_flash_next), not seen in
training, the adapter cuts KL from full from 0.201 to 0.049.

Limits: one model, one adapter, one seed, 60 samples; training and evaluation
traces come from the same dataset and agent (different files of it).
Results: `results/gate3_qwen3_4b_writer.json`, `results/writer_qwen3_4b_val_kl.txt`,
`results/writer_diag_same.json`, `results/writer_diag_other.json`.

## Thirteenth: would keying on line ranges catch more? (Nov 10, 2026)

`python -m warmtier.blocks <raw SWE-agent parquet glob>`, CPU only. The Ninth
section left a gap on SWE-agent: teammates open the same files far more often
than they see the same bytes. One guess was window misalignment (the viewer
shows 100 lines at a time), which a store keyed on line ranges would fix.
This tests that on a third of the tasks (1,195 tasks, 23,811 runs): a unit is
reusable if an earlier teammate saw the same path, same line numbers and same
text; blocks are aligned to line numbers and count only when shown whole.

Share of file-view line text an earlier teammate already supplied:

| unit | k=2 | k=4 | k=8 | k=16 |
|---|---|---|---|---|
| whole view | 7.56% | 14.82% | 21.09% | 28.81% |
| 100-line blocks | 4.95% | 9.09% | 12.32% | 15.45% |
| 25-line blocks | 8.01% | 15.26% | 21.05% | 27.62% |
| 10-line blocks | 8.70% | 16.65% | 22.85% | 30.00% |
| single lines (upper bound) | 9.08% | 17.39% | 23.86% | 31.35% |

Reading:
- The guess was wrong. Even at single-line granularity, reuse across
  teammates is only 1.5 to 2.5 points above whole-view hashing. Teammates
  open the same files but read different parts of them.
- Fixed 100-line blocks are worse than whole views, because a window rarely
  covers an aligned block.
- Inside one run the picture is different: 54% of the file-view line text a
  run reads is lines it has already read itself at the same position with the
  same text, against 28% when the unit is the whole view. Scrolling, reopening
  and the window echoed after each edit repeat most lines. So line-range
  keying matters for reuse within an agent's own context, not across agents.

Limits: SWE-agent only (the line numbers come from its viewer). Keys include
the line number, so text that shifts after an insertion counts as new. A third
of tasks, chosen by a hash of the task id. Whether KV can be reused for a line
range seen at another position is the same open question as for whole files.
Results: `results/blocks_nebius.json`.

## Next

2. Task outcomes: run a harness with and without stale-view eviction and
   compare pass rate (shwin-harness, `SHWIN_COMPACTOR=edit_aware:0.15`).
3. Follow a splice past the first call: does the stale view's influence fade
   or persist over later turns?
4. Audit of the measurement code and every number above.

## Related work this builds on

MemOS (arXiv 2507.03724), Cartridges (arXiv 2506.06266), What It Costs to
Compose, Rebuild, and Correct Precomputed Memory (arXiv 2608.30647), The
Missing Memory Hierarchy (arXiv 2603.09023), ForkKV (arXiv 2604.06370).
