# Measurement records

The files behind every number in the READMEs, from 2026-10-05 on four DGX Sparks (GB10). They were produced by the
stack `tensorfold-four-spark-tp4/` packages (same TensorFold tree, same `lib/` wrappers, pinned tiles, b12x, drafter,
BF16 lm_head and chat template), before this repo existed; see the root README's
[Measurement status](../../README.md#measurement-status). The int4 record is one step further removed (a copy of
`lib/` with its own launcher), so its figures are TODO(confirm) everywhere they are quoted.

Copied as produced, with three edits: line endings are LF; operator paths are rewritten to `/workspace/...`
(`/workspace/view` is the serve view, `/workspace/TensorFold` the checkout, `/workspace/b12x` the b12x stage); and host
names and fabric addresses are replaced by `spark-0` … `spark-3` and `10.0.0.10` … `10.0.0.13` in rank order.
Files marked *derived* are smaller extracts of a larger raw file; their header says what was dropped.

| Directory | What | Produced by |
|---|---|---|
| [`2026-10-05-sweep/`](2026-10-05-sweep/README.md) | the speed sweep: RoCE vs NCCL, DFlash2 depth / confidence, pinned tiles, exactness | in-process harness calling `Glm53Engine.generate` on all four ranks (no HTTP), context 32768, greedy, 512 tokens, median of 3 |
| [`2026-10-05-serve-32k/`](2026-10-05-serve-32k/) | the `/v1` server at `--context 32768`: SixCat v0.7.0 speed (decode/balanced in the console log; prefill JSON), the raw SSE trace, the smoke result | `tensorfold serve` + `lib/`, SixCat from a client through an ssh tunnel |
| [`2026-10-05-serve-262k-dcp1-failed/`](2026-10-05-serve-262k-dcp1-failed/) | 262,144 tokens with `TF_GLM53_DCP=1`: cache guard refusals on ranks 0/1, swap growth and watchdog trip on ranks 2/3 | the same server; rank logs and watchdog flags |
| [`2026-10-05-serve-262k-dcp4/`](2026-10-05-serve-262k-dcp4/) | 262,144 tokens with `TF_GLM53_DCP=4` (`PROFILE=dcp4-262k`): the 6 prompts (`sweep6.json`), `bench_v1.py` against the 32K reference (fails on ids by design: another numeric path), SixCat, smoke, memory | the same server |
| [`2026-10-05-serve-160k/`](2026-10-05-serve-160k/) | 163,840 tokens with `TF_GLM53_DCP=1` (`PROFILE=fast-160k`, the default): the 6 prompts (`sweep6.json`), the 128K and 162.5K book prompts, SixCat, rank logs, memory | the same server; `bench/longctx_run.py` |
| [`2026-10-05-serve-262k-int4/`](2026-10-05-serve-262k-int4/README.md) | 262,144 tokens with an int4 latent KV cache (`PROFILE=int4-262k`): the 6 prompts (1 serial + 3 drafted each), smoke, SixCat, rank logs, memory, per-host kernel parity | **TODO(confirm)**: the same TensorFold tree (`glm53-kv-int4` @ `0c858e3`) through a *copy* of `lib/` and the operator's own launcher, not this repo's entry scripts; see its README |
| [`2026-10-05-quality-g1/`](2026-10-05-quality-g1/REPORT_RESULTS.md) | teacher-forced quality gate G1: TensorFold vs exllamav3 TP=4 on 43,903 prompt and 2,424 decode positions | gate harness; NCCL, context 4096, no drafting, that boot's tile picks (`tile_picks_tf.txt`), not the pinned table |

Notes per record:

- `sweep6.json` (160K and 262K) comes from `bench/longctx_run.py` with `THINKING=1` on `bench/sweep6_prompts.json`:
  one serial and one drafted run per prompt, the server's `tensorfold` stats, and whether the ids match.
- `longctx_128k.json` / `.log`: Moby-Dick (Project Gutenberg #2701) trimmed to 127,544 served prompt tokens.
  `longctx_162k.log`: War and Peace (#2600), 162,544 tokens; the run was stopped during the drafted arm, so only the
  serial arm exists and no JSON was written.
- `sixcat_speed_all.log`: `--profile all`. The suite's 600 s budget expired before its prefill profile on this
  one-request-at-a-time engine, so it wrote no JSON; decode and balanced figures exist only in this log.
  `sixcat_speed_prefill.json` / `.log` is the second invocation with the same flags and `--profile prefill`. A third
  160K invocation was contaminated by another client's requests on the same tunnel port and is not included.
- `memwatch_events.jsonl` (*derived*): the watchdogs' start / stop / violation events, plus the minimum MemAvailable
  over each node's 1 s samples (the sample logs are ~1 MB a node).
- `rank*.log`: TensorFold's and the wrappers' start-up lines: RoCE (`equals the NCCL rank-order sum: True`), tiles
  pinned and in use, startup consensus, the cache guard, `serving ... /v1`, and one `done req-...` line per request
  with the server's own prefill time.
