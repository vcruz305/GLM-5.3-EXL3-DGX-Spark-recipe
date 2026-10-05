# Four DGX Sparks: TensorFold TP=4

**Model:** [`vcruz305/GLM-5.3-EXL3-3.38bpw`](https://huggingface.co/vcruz305/GLM-5.3-EXL3-3.38bpw) (revision
`cc64e77`, 58 shards, 319 GB) + the original BF16 `lm_head.weight` from
[zai-org/GLM-5.3](https://huggingface.co/zai-org/GLM-5.3) (revision `aca966e`)
**Drafter:** [`incoai/GLM-5.3-DFlash2`](https://huggingface.co/incoai/GLM-5.3-DFlash2) (revision `425aa61`,
CC BY-NC-ND 4.0: non-commercial use only)
**Engine:** TensorFold PR #159 (`glm_moe_dsa`, TP=4) + two loader fixes =
[`vcruz305/TensorFold@757a851`](https://github.com/vcruz305/TensorFold/tree/glm53-tp4-spark), b12x `b58f34e` RoCE
reductions
**Status:** measured 2026-10-05 on the stack these scripts package; the scripts themselves are statically checked
only (see the root README's [Measurement status](../README.md#measurement-status))

One rank per Spark over the ConnectX-7 fabric. Rank 0 serves the OpenAI-compatible API with TensorFold's own
`tensorfold serve`; ranks 1-3 follow it. Every reply is token-identical to serial decoding, and RoCE reductions give
the same bits as NCCL.

## Contents

- [Performance (default profile)](#performance-default-profile)
- [Profiles](#profiles)
- [Runtime identity](#runtime-identity)
- [Run it](#run-it)
- [Requests](#requests)
- [What runs where](#what-runs-where)
- [What the scripts add over PR #159](#what-the-scripts-add-over-pr-159)
- [Guards](#guards)
- [Context sizing](#context-sizing)
- [Changing the configuration](#changing-the-configuration)
- [Not measured yet](#not-measured-yet)
- [Troubleshooting](#troubleshooting)

## Performance (default profile)

`PROFILE=fast-160k`: `--context 163840`, `TF_GLM53_DCP=1` (whole bf16 KV cache on every rank),
`TF_GLM53_PROMPT_ROWS=2048`, verify rows 8, DFlash2 depth 7 / confidence 0.60, b12x RoCE, pinned tiles, thinking on
by default. Records: [`bench/records/2026-10-05-serve-160k/`](../bench/records/2026-10-05-serve-160k/).

### Decode per prompt

Greedy, 512 new tokens, thinking on (the official template, effort Max), through `/v1`
([`bench/longctx_run.py`](../bench/longctx_run.py) with `THINKING=1` on
[`bench/sweep6_prompts.json`](../bench/sweep6_prompts.json)), one run per arm. Engine-reported decode tok/s.

| Prompt (prompt tokens) | Decode with DFlash2 | Decode without drafting | Tokens per round | Sweep, 32K (median of 3) |
|---|---:|---:|---:|---:|
| code (52) | 40.64 | 27.07 | 3.08 | 40.62 |
| prose (45) | 35.71 | 27.17 | 2.36 | 35.58 |
| math (59) | 56.50 | 26.84 | 5.38 | 57.39 |
| chat_explain (73) | 37.67 | 26.16 | 2.73 | 38.69 |
| chat_multiturn (174) | 32.57 | 26.68 | 2.16 | 32.83 |
| long_doc (2,842) | 43.54 | 24.83 | 3.60 | 43.44 |
| **mean of 6** | **41.11** | **26.46** | | 41.42 |

On every prompt the drafted ids equal the serial ids, and the serial ids equal the 32K sweep's serial ids
(`bench/sweep_reference.json`): 163840 with `TF_GLM53_DCP=1` is the same numeric path as 32768.

### Long prompts

[`bench/make_prompts.py`](../bench/make_prompts.py) + [`bench/longctx_run.py`](../bench/longctx_run.py), thinking off,
greedy, 512 new tokens. TTFT is client-observed (the server's own prefill time was within 1 s of it).

| Measurement | Result |
|---|---:|
| 127,544-token prompt (Moby-Dick), decode with DFlash2 | **33.30 tok/s** |
| 127,544-token prompt, decode without drafting | 21.76 tok/s |
| 127,544-token prompt, DFlash2 tokens per round | 2.57 |
| 127,544-token prompt, drafted == serial | 512 / 512 tokens |
| 127,544-token prompt, TTFT (serial arm) | 514.98 s |
| 127,544-token prompt, TTFT (DFlash2 arm) | 511.23 s |
| 127,544-token prompt, prefill rate (DFlash2 arm) | 249.5 tok/s |
| 162,544-token prompt (War and Peace), decode without drafting | 20.82 tok/s |
| 162,544-token prompt, TTFT | 639.04 s |
| 162,544-token prompt, prefill rate | 254.4 tok/s |
| 162,544-token prompt, decode with DFlash2 | TODO(confirm): stopped before the drafted arm ran |

Serial decode falls from ~27 tok/s on short prompts to 21.8 at 128K and 20.8 at 162.5K; DFlash2 keeps a +53% margin
at 128K. A 162.5K prompt takes about 10.6 minutes to its first token.

### SixCat v0.7.0 speed suite

`sixcat speed --policy strict --thinking off --profile all|prefill --candidates 1,2,4,8 --samples 32`, client-side,
through an ssh tunnel. The decode profile is a synthetic word-list prompt that drafts far better than real text: a
ceiling, not what a user sees. The server sends no timing fields, so TTFT and prefill are client-observed.

| Measurement | 160K server (default) | 32K server |
|---|---:|---:|
| Decode, C=1, per-stream p50 | 69.19 tok/s | 69.41 tok/s |
| Prefill, ~2,475-token prompt, p50 | 313.4 tok/s | 313.5 tok/s |
| TTFT on that prompt, p50 | 7.90 s | 7.89 s |

The suite's 600 s budget ran out before its prefill profile on this one-request-at-a-time server, so each context
ran two invocations (`--profile all` for decode, `--profile prefill`); the decode figures exist only in the console
logs. C>1 levels only queue on this engine. A third invocation at 160K that another client's requests contaminated
was discarded.

### Memory (MemAvailable, GiB; swap growth vs the watchdog's baseline)

| Rank | Predicted minimum | Session minimum | While serving | Swap growth |
|---|---:|---:|---:|---:|
| 0 | 9.2 | 7.18 | 8.0 | 0 |
| 1 | 5.0 | **4.86** | 5.95 | 0 |
| 2 | 6.9 | 6.96 | 7.9 | 0 |
| 3 | 7.1 | 6.96 | 8.0 | 0 |

The cache and prompt buffers are allocated at load, so long prompts did not lower MemAvailable further. The
watchdog floor was 2 GiB with any swap growth fatal; it never tripped.

## Profiles

`PROFILE=` selects `profiles/<name>.env` on every Spark (`serve.sh` forwards it).

| Profile | Context | KV cache | TensorFold | Status |
|---|---:|---|---|---|
| `fast-160k` (default) | 163,840 | bf16, whole on every rank (`TF_GLM53_DCP=1`) | `glm53-tp4-spark` `757a851` | **measured**, numbers above |
| `dcp4-262k` | 262,144 | bf16, split across ranks (`TF_GLM53_DCP=4`) | `glm53-tp4-spark` `757a851` | measured, below |
| `int4-262k` | 262,144 | int4 latent, whole on every rank | `glm53-kv-int4` `47c7aa0` | **pending validation**, opt-in (`ALLOW_UNVALIDATED=1`) |

### `dcp4-262k`: 262,144 tokens with decode context parallelism

Loads and serves; drafting stays exact against this server's own serial replies. It is another numeric path: its
serial ids differ from the 32K / 160K ids (first differences at tokens 104 / 84 / 24 / 119 / 7 / 20 on the six
prompts), and it is slower (prompt chunks are capped at 1,024 rows under DCP). Records:
[`bench/records/2026-10-05-serve-262k-dcp4/`](../bench/records/2026-10-05-serve-262k-dcp4/).

| Measurement | Result |
|---|---:|
| Decode with DFlash2, mean of 6 prompts | 35.91 tok/s |
| Decode without drafting, mean of 6 prompts | 23.67 tok/s |
| Drafted == this server's serial | 6 / 6 prompts |
| SixCat decode, C=1, p50 | 57.08 tok/s |
| SixCat prefill, ~2,474-token prompt, p50 | 232.2 tok/s |
| SixCat TTFT on that prompt, p50 | 10.66 s |
| Lowest MemAvailable on any rank | 14.79 GiB |

`serve.sh bench` on this profile compares drafted replies with the server's own serial replies (`BENCH_IDS=self`),
not with the 32K reference.

**bf16 at 262,144 with `TF_GLM53_DCP=1` does not fit.** The cache guard refused on ranks 0 and 1 (21.9 GiB of
caches needed, 27.6 / 27.8 GiB free with the 6 GiB reserve); on ranks 2 and 3 it passed, MemAvailable fell to
4.60 / 4.69 GiB, swap grew 4 kB, and the watchdog stopped all four
([`bench/records/2026-10-05-serve-262k-dcp1-failed/`](../bench/records/2026-10-05-serve-262k-dcp1-failed/)).

### `int4-262k`: pending validation

An int4 MLA latent cache in exllamav3's Q4 cache format (the 512-wide latent in 32-value groups with fp16 scales;
RoPE dims and indexer keys stay 16-bit), from the
[`glm53-kv-int4`](https://github.com/vcruz305/TensorFold/tree/glm53-kv-int4) branch. Validated on CPU only (the
writer is bit-identical to exllamav3's quantizer; the readers are row-invariant). Predicted, not measured: 9.23 GiB
of cache per rank at 262,144 tokens. No speed, memory or quality figure exists on a Spark; int4 changes the
numerics, so its replies are not the bf16 ids and the pack's quality figures do not carry over.

```bash
PROFILE=int4-262k bash tensorfold-four-spark-tp4/setup.sh                       # on each Spark: second checkout
PROFILE=int4-262k ALLOW_UNVALIDATED=1 bash tensorfold-four-spark-tp4/serve.sh up
```

## Runtime identity

| | |
|---|---|
| engine | [ashhart/TensorFold PR #159](https://github.com/ashhart/TensorFold/pull/159) (`drowzeys/TensorFold` `glm-moe-dsa-tp4` @ `689596d`) + 2 commits = [`vcruz305/TensorFold` `glm53-tp4-spark` @ `757a851`](https://github.com/vcruz305/TensorFold/commits/glm53-tp4-spark): per-layer `safe_open` release + `malloc_trim`, per-shard page-cache drop before the cache guard, fail-closed fp16 → bf16 cast. `git diff 689596d 757a851` has sha256 `1555d8ad…`, byte-identical to the measured tree's diff |
| serving wrappers | `lib/` (applied at import by the launcher; the TensorFold tree is not edited) |
| reductions | b12x `b58f34e` RoCE one-shot, one rail; `equals the NCCL rank-order sum: True` on all four ranks is enforced |
| tiles | `tiles/tiles.json`, sha256 `db409731…`, 567 EXL3 linears, loaded instead of tuned at boot |
| drafter | GLM-5.3-DFlash2 `425aa61` (block 8, taps 5/19/33/47/61/75), depth 7, confidence 0.60 |
| lm_head | `lm_head.weight` BF16 [154880, 6144] from zai-org/GLM-5.3 `aca966e` |
| chat template | the official template + the thinking-off line, sha256 `2059ad4b…` |
| knobs | `--context 163840`, `TF_GLM53_DCP=1`, `TF_GLM53_PROMPT_ROWS=2048`, `TF_GLM53_VERIFY_ROWS=8`, per-round RoCE health check, one request at a time |
| torch | 2.14.1+cu130 |

The measured runs used these files' predecessors: `lib/tf_speed_patches.py`, `lib/tf_speed_common.py` and
`lib/tf_serve_patches.py` are byte for byte the measured files apart from line endings; the launcher
(`lib/tf_serve_rank.py`) and watchdog (`lib/tf_serve_memwatch.py`) differ only as their headers say (paths and opt-in
knobs; with the default profile the environment and `tensorfold serve` argv are the measured ones).

## Run it

**Requirements, on every Spark:** DGX OS / Ubuntu 24.04 (aarch64), CUDA 13 with `nvcc`, `python3` with its headers
(`libpython3.12-dev`: Triton JIT-compiles against `Python.h`), `gcc` and `libibverbs-dev` (b12x's RDMA proxy),
~330 GB free disk for the pack, drafter and head, and at least 100 GiB of MemAvailable before a start. The four
Sparks reach each other over the ConnectX-7 fabric (one RoCE rail), and rank 0 can ssh to ranks 1-3 without a
password (its watchdog stops them on a breach). Every rank reads every shard, so **each Spark holds the full pack**.

**1. Each Spark** (same clone path on all four):

```bash
git clone https://github.com/vcruz305/GLM-5.3-EXL3-DGX-Spark-recipe.git && cd GLM-5.3-EXL3-DGX-Spark-recipe
hf auth login                                   # the pack is gated: request access first
bash tensorfold-four-spark-tp4/setup.sh         # --check later verifies without changing anything
```

Downloading on one Spark and copying `~/models/` to the others over the fabric (`rsync`) also works; then run
`SKIP_DOWNLOADS=1 bash tensorfold-four-spark-tp4/setup.sh` there.

**2. The hosts file**, where you drive the cluster from:

```bash
cp tensorfold-four-spark-tp4/hosts.example tensorfold-four-spark-tp4/hosts && $EDITOR tensorfold-four-spark-tp4/hosts
```

Columns: rank, ssh target (or `local`), fabric IP, and optionally how rank 0 reaches that rank over ssh. `serve.sh`
hands the table to every Spark it calls, so the Sparks need no copy (running `rank.sh` by hand on a Spark does).

**3. Serve:**

```bash
bash tensorfold-four-spark-tp4/serve.sh preflight   # "preflight OK on all four"
bash tensorfold-four-spark-tp4/serve.sh up          # page cache, watchdogs, ranks 1-3, rank 0, waits for READY
bash tensorfold-four-spark-tp4/serve.sh smoke       # /v1 checks on rank 0
bash tensorfold-four-spark-tp4/serve.sh bench       # the 6 reference prompts: ids and tok/s vs the tables above
bash tensorfold-four-spark-tp4/chat.sh "Explain RoCE in two sentences."   # on rank 0, or through serve.sh tunnel
bash tensorfold-four-spark-tp4/serve.sh down
```

`DRY_RUN=1` before any `serve.sh` step prints every ssh command and runs nothing. `serve.sh logs` copies the rank
logs, watchdog logs and bench JSON into `tensorfold-four-spark-tp4/runs/<time>/`.

Start order and timing: ranks 1-3 start first, then rank 0. Each rank loads 78 layers (about 400 s), sets up RoCE,
loads the pinned tiles and the drafter and captures 96 decode graphs; rank 0 then runs a 16-token serial and a
16-token drafted warm-up and starts HTTP. `up` returns when rank 0 logs `[tensorfold] serving ... /v1`. The first
start on a Spark also JIT-builds TensorFold's CUDA kernels.

## Requests

- **Drafting:** no `tf_mtp` means DFlash2 (`"tf_mtp": "dflash"` is the same). `"draft": false` decodes serially. Any
  other `tf_mtp` is refused.
- **Thinking:** on by default; the prompt ends `<|assistant|><think>` exactly like the official template. With
  `"chat_template_kwargs": {"enable_thinking": false}` (or `"reasoning_effort": "none"`) it ends
  `<|assistant|></think>`. The reasoning arrives in `reasoning_content`, the answer in `content`; one SSE chunk per
  token, and the chunk that closes `</think>` can carry both fields: read both.
- **Stop:** `<|endoftext|>` 154820, `<|user|>` 154827, `<|observation|>` 154829. Stop strings and a client disconnect
  do not stop the engine: it decodes on to an end token or `max_tokens` (default 4096). Send a sensible `max_tokens`.
- **Concurrency:** one request at a time; others queue.
- **Context:** the profile's `--context`, prompt plus reply. Longer requests get HTTP 400.
- **Stats:** every reply carries a `tensorfold` block (decode `tok_s`, `tokens_per_round`, `mtp_mode`, `depth`,
  `confidence`); `"return_token_ids": true` adds the ids.

## What runs where

| File | Role |
|---|---|
| `env.sh` | every pin (TensorFold, b12x, pack, drafter, head, shas), path and knob; `verify_runtime`, `verify_view`, the hosts-file reader |
| `profiles/*.env` | `fast-160k` (default), `dcp4-262k`, `int4-262k` (pending) |
| `setup.sh` | per Spark: venv + torch cu130, the pinned TensorFold checkout, b12x stage, downloads, serve view, checks |
| `serve.sh` | driver: preflight / up / status / smoke / bench / down / logs / tunnel over ssh, `DRY_RUN=1` |
| `rank.sh` | one Spark, one rank: preflight, fadvise, watch, start, state, stop, unwatch, status, smoke, bench |
| `chat.sh` | one streamed chat request and the engine's stats for it |
| `drop-model-cache.sh` | `posix_fadvise(DONTNEED)` of the model files (GB10 counts page cache as used) |
| `hosts.example` | the four Sparks: rank, ssh target, fabric IP |
| `lib/tf_serve_rank.py` | the launcher: environment before imports, refusals, the wrappers, then `tensorfold.cli.main(["serve", ...])` |
| `lib/tf_serve_patches.py` | serving wrappers: DFlash2 default, startup + per-request rank consensus, fatal exit on a failed round, warm-up |
| `lib/tf_speed_patches.py`, `lib/tf_speed_common.py` | the speed sweep's wrappers: tile pinning, the RoCE rank-order hard stop, the per-round health guard and consensus |
| `lib/tf_serve_memwatch.py` | per-Spark memory / liveness watchdog; rank 0's also polls the peers and stops all four |
| `tiles/tiles.json` | the pinned tile table |
| `tools/make_view.sh`, `tools/fix_chat_template.py`, `tools/fetch_lm_head.py` | the serve view: pack symlinks + BF16 lm_head + fixed template |
| `tools/b12x_env.sh` | the RoCE environment (stage path, HCA, spin limit, per-host GID) |

## What the scripts add over PR #159

PR #159's `tensorfold serve --tp 4` runs the same `Glm53Engine.generate` path the speed sweep drove, but as shipped
it does not reproduce the measured configuration:

1. **Default draft mode.** A request without `tf_mtp` runs MTP, and this pack has no MTP layer, so it decodes serially
   at ~27 tok/s. `TF_GLM53_MTP=dflash` cannot be used (it raises before the drafter is attached); the wrapper makes
   DFlash2 the default.
2. **DFlash2 confidence.** The engine default is 0.3; the measured setting is 0.60 (0.50-0.75 measured flat).
3. **Context.** The CLI default is the model's 1,048,576 tokens, which turns on decode context parallelism (other
   bits, slower). The profiles always pass `--context` and an explicit `TF_GLM53_DCP`.
4. **Tiles.** Picks are re-tuned at every boot, so the bits vary per boot; the pinned table fixes them.
5. **Silent RoCE fallback.** A wrong HCA name, a missing stage or a failed order check falls back to NCCL without
   stopping; the launcher refuses instead.
6. **No b12x health check.** A timed-out RoCE runtime is poisoned silently under graph replay; the per-round guard
   (~0.6 ms a round) stops every rank instead.
7. **Thinking off is ignored** by the official template; the view serves the fixed one.
8. **No warm-up**; rank 0 warms up before HTTP starts.
9. **Half-alive server.** A failed round leaves rank 0 answering against dead followers; every rank exits instead.
10. **No consensus.** Ranks check policy and reply hashes after every request.

And two loader fixes in the TensorFold tree itself (the `glm53-tp4-spark` commits): the PR's loader keeps every
`safe_open` handle for the whole load (host RSS grew ~2.7 GiB per MoE layer and the first launch swapped), and GB10's
`mem_get_info` excludes page cache, so the cache guard refused 32K with ~1.3 GiB "free". The pack also stores
`kv_b_proj`, the router gate and the indexer's small tensors as fp16 while the kernels take bf16; the loader casts
them only after checking each round-trips exactly (all 237 do).

## Guards

| Event | Who acts | Result |
|---|---|---|
| MemAvailable < `TFS_MIN_AVAIL_GIB` (2) or swap grows more than `TFS_MAX_SWAP_GROWTH_KB` (0) on any Spark | that Spark's watchdog (1 s); rank 0's polls the peers every 5 s | `kill -9` of every rank, `/tmp/tf_serve_memwatch.VIOLATION` |
| a rank dies while others run > 45 s | rank 0's watchdog | all ranks killed (the others would block in a collective forever) |
| RoCE env wrong (`NCCL_IB_HCA` with `=`, no RoCE v2 GID, spin limit) | launcher | exit 3 before loading |
| RoCE sum ≠ NCCL rank-order sum, or RoCE did not come up | wrapper + post-load check | exit 3 on every rank |
| tile table or chat template bytes not the pinned ones; tile table in use ≠ pinned | launcher / post-load | exit 3 |
| ranks disagree on DFlash2 policy, verify rows, RoCE or tiles at startup | startup consensus | exit 3 on every rank |
| a rank's reply or policy differs from rank 0's after a request | per-request consensus | exit 4 on every rank |
| b12x runtime poisoned in a round, or any exception inside a request's collectives | health guard + fatal-round wrapper | exit 5 on every rank |
| another job on a GPU, < 100 GiB MemAvailable, no or flagged watchdog | `rank.sh start` | refuses |

After any exit: `serve.sh logs`, read `rank*.log` (`[tf_serve] FATAL ...`) and the watchdog logs, then `serve.sh down`
and `serve.sh preflight` before the next `up`.

## Context sizing

With `TF_GLM53_DCP=1` every rank holds the whole cache: per cached token and rank, the latent cache
78 × 576 × 2 = 89,856 B, the indexer keys (21 layers carry an indexer in this pack) 5,376 B and the score scratch
2,112 B: **97,344 B**. The engine's guard (`runner._check_cache_fits`) counts only the latent part. Predicted minimum
MemAvailable = the 32K run's per-rank minimum − 97,344 × (context − 32,768); it held within ~0.3 GiB at 163,840
(table above). 163,840 is the largest multiple of 8,192 that keeps every rank ≥ 5 GiB by that model; 196,608 would
leave ranks 1-3 at 2.0-4.2 GiB, and swap already grew at 4.6 GiB in the 262K attempt.

## Changing the configuration

Export the variable before `serve.sh`; it forwards every `env.sh` variable you set to all four Sparks and prints the
resolved profile on start. Values marked `(measured)` in `env.sh` are the measured configuration: change one only to
measure a new configuration, and label the numbers as new.

- `TFS_ROCE=0`: NCCL reductions (sweep: 37.07 vs 41.42 tok/s).
- `TFS_DRAFT_DEFAULT=0`: a serial-only server.
- `TFS_THINKING=0`: thinking off unless a request turns it on.
- `TFS_DFLASH_CONFIDENCE`: 0.50-0.75 measured flat within 0.5 tok/s.
- `TFS_CONTEXT=32768`: the sweep's context (same ids and speed as the default).
- `TFS_HTTP_HOST=0.0.0.0` needs `TFS_API_KEY_FILE`; the launcher refuses a non-loopback bind without a key.
- `TFS_MIN_AVAIL_GIB`, `TFS_MAX_SWAP_GROWTH_KB`: the watchdog rules (2 GiB, 0 kB).
- Per-request policy changes are not supported on purpose: the engine re-reads a `DFLASH_CFG` file per request,
  separately on each rank, and different windows on different ranks would trip the consensus check.

## Not measured yet

- **This repo's scripts end to end** on fresh Sparks: `setup.sh` from scratch (venv, TensorFold checkout, b12x stage,
  downloads, view), then `serve.sh up`, `smoke`, `bench`. Every number above came from the predecessor stack.
- **Sampled requests.** Everything was greedy; the speed-up at temperature > 0 is unmeasured.
- **The thinking-on stress sequence** (long prompts with tool calls, streamed channels) has not been run.
- **HTTP overhead, directly.** Streaming does per-token work on rank 0 inside the decode round. The 6-prompt mean
  through `/v1` (41.11, one run per prompt) is 0.75% below the in-process sweep's (41.42, median of 3).
- **DFlash2 at 162.5K**: the run was stopped before the drafted arm.
- **Quality on the pinned tile table** (the G2 gate was started and stopped), and anything about `int4-262k`.
- `vm.compaction_proactiveness` is 20 on the measured Sparks; PR #159's own recipe uses 0. No stalls were seen.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `no hosts file at ...` | `cp hosts.example hosts` and fill in the four Sparks |
| `recipe clone at <sha>, here <sha>` | `git pull` on every Spark: the four ranks must run the same `lib/` |
| `TensorFold at ... is <sha>, the recipe pins ...` | `setup.sh` on that Spark (with the same `PROFILE` for int4) |
| `lacks the measured loader fixes` | a TensorFold tree without the `glm53-tp4-spark` commits; `setup.sh` |
| `no Python.h under ...` | `apt install libpython3.12-dev` (or the matching version), then `setup.sh` |
| `b12x tarball sha256 ... !=` | GitHub served a different archive for `b58f34e`; inspect it, then `B12X_TARBALL_SHA=<sha>` to accept |
| `REFUSE: NCCL_IB_HCA='=rocep1s0f0'` | start through `serve.sh` / `rank.sh`, never by hand: `tools/b12x_env.sh` strips the `=` |
| `REFUSE: no RoCE v2 GID for <ip>` | the hosts file's `fabric_ip` is not on `FABRIC_IFNAME`, or the RDMA device is not `ROCE_HCA` |
| `decode-window reductions: NCCL` then `FATAL ... RoCE requested` | b12x stage or GID problem; `rank.sh preflight --rank R` prints the resolved GID |
| `context ... x 1 streams needs ... is free` | page cache or another job's memory: `serve.sh fadvise`, check `preflight`; never lower `TF_GLM53_CACHE_RESERVE_GB` |
| `REFUSE: no watchdog` / `watchdog flagged` | `serve.sh up` starts the watchdogs; after a trip read `/tmp/tf_serve_memwatch.VIOLATION`, then `serve.sh down` |
| `PROFILE=int4-262k is pending validation` | intended: set `ALLOW_UNVALIDATED=1` only to validate it |
| first request slow | the warm-up runs before HTTP starts; the first long prompt can still build prompt kernels once |
| client gets `reasoning_content` only, `finish_reason: length` | the reply spent `max_tokens` thinking: raise it or send thinking off |
| decode far below the tables | the reply's `tensorfold` block must show `mtp_mode` `dflash`, `depth` 7, `confidence` 0.6; rank 0's log must show `decode-window reductions: RoCE one-shot`; nothing else on the GPUs |
| `bench` fails on ids with `PROFILE=dcp4-262k` | expected against the 32K reference; the profile sets `BENCH_IDS=self` (drafted vs this server's serial) |
