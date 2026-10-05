# Speed sweep, 2026-10-05

In-process harness on all four Sparks: each rank built `Glm53Engine(view, rank, master, port, context=32768,
mtp_drafts=2)` with the same wrappers as `tensorfold-four-spark-tp4/lib/` (tile pinning, the RoCE rank-order hard stop,
the per-round health guard and after-request consensus) and rank 0 called `engine.generate(ids, 512, None, ...)`
directly: no HTTP. Greedy, 512 new tokens, thinking on (the official template, effort Max), `TF_GLM53_PROMPT_ROWS=2048`,
verify rows 8. Drafted cells are the median of 3 runs; serial cells the median of the serial run and its bookend.
Four engine loads (smoke, A, B, C), every one exited 0 on all four ranks.

| Arm | Reductions | Tiles | Serial, mean of 6 | Serial, code / prose / math | Best config | DFlash2, mean of 6 | DFlash2, code / prose / math |
|---|---|---|---:|---:|---|---:|---:|
| A | RoCE (b12x one-shot) | tuned at boot, saved as `tiles.json` | 26.78 | 27.13 | d7 c0.60 | **41.42** | 44.53 |
| B | NCCL | loaded (pinned) | 23.99 | 24.15 | d7 c0.60 | 37.07 | 39.74 |
| C | RoCE (b12x one-shot) | loaded (pinned) | 26.66 | 26.95 | d7 c0.60 | 41.24 | 44.29 |

Per-prompt cells, every configuration tried (confidence 0.30-0.75, depth 5 and 7, drafter edge 0.4 / 0.9), verify-window
timings, kernel profiles and memory: [`results_sections.md`](results_sections.md) and `table_*.md`. Every run's stats
without its token ids: `runs_*.json` (*derived* from the harness's `results.json`; each run keeps the sha of its ids).

What it established:

- **Exactness.** 283 drafted runs (smoke 8, A 111, B 73, C 91) equal their serial reply ids, 0 differences; all 48
  teacher-forced 8-row probe windows equal the serial reply.
- **RoCE changes speed, not outputs.** The serial ids are identical across A (RoCE), B (NCCL) and C (RoCE re-boot) on
  all 6 prompts. On the same pinned tiles, serial decode is 27.13 vs 24.15 tok/s (+12.3%) for code / prose / math.
- **Confidence is the lever, and a small one.** From 0.30 to 0.60: +1.9 tok/s mean on RoCE. From 0.50 to 0.75 the
  means are flat within 0.5. Depth 5 and drafter edge 0.4 / 0.9 gave nothing.
- **Pinned tiles.** A boot tunes different tile splits on each rank; the pinned table (`tiles.json`, sha256
  `db409731…`, 567 linears) makes the bits the same across boots (`rank0_C_roce.log`: `tiles pinned ... 352 differ
  from plan()`, then `took rank 0's tiles (0 of 567 linears differed)`).
- **Health guard cost.** The per-round all-rank b12x check costs 0.61 ms a round (median, RoCE).
- **Memory.** Lowest MemAvailable 16.45 GiB (rank 1, arm B); swap growth 0 on every node in every arm.
