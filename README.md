# GLM-5.3 EXL3 on NVIDIA DGX Spark

Serves [zai-org/GLM-5.3](https://huggingface.co/zai-org/GLM-5.3) (78 layers, 256 routed experts per MoE layer, DSA
sparse attention) from the SAGE MixedK EXL3 pack
[vcruz305/GLM-5.3-EXL3-3.38bpw](https://huggingface.co/vcruz305/GLM-5.3-EXL3-3.38bpw) (3.38 bpw, 8-bit head,
319 GB) on **four NVIDIA DGX Sparks**, as an OpenAI-compatible `/v1` API.

The engine is [TensorFold](https://github.com/ashhart/TensorFold)'s full-GLM-5.3 tensor-parallel CUDA engine
([PR #159](https://github.com/ashhart/TensorFold/pull/159) by [@drowzeys](https://github.com/drowzeys), still open),
plus two loader fixes this pack needs. It runs on the
[`glm53-tp4-spark`](https://github.com/vcruz305/TensorFold/tree/glm53-tp4-spark) branch of my TensorFold fork (see
[Upstream status](#upstream-status)). Decode
reductions go over the ConnectX-7 fabric with [b12x](https://github.com/local-inference-lab/b12x)'s RoCE one-shot
all-reduce. Drafting uses the [DFlash2](https://huggingface.co/incoai/GLM-5.3-DFlash2) drafter. **Drafted replies are
token-identical to serial decoding:** every measured drafted reply equals the `"draft": false` reply, token for token.

> **Using an AI agent to set this up?** It should read [`AGENTS.md`](AGENTS.md) (every step, the expected output and
> the fix for each failure), [Do not](#do-not) and [Troubleshooting](#troubleshooting) before running anything. Nobody
> needs to know the Sparks' names: `./glm53 init`, run on any one of them, finds the other three.

| Sparks | Folder | Model | Engine | Status |
|---|---|---|---|---|
| **4** | [`tensorfold-four-spark-tp4/`](tensorfold-four-spark-tp4/README.md), driven by [`./glm53`](glm53) | [`GLM-5.3-EXL3-3.38bpw`](https://huggingface.co/vcruz305/GLM-5.3-EXL3-3.38bpw) + [DFlash2](https://huggingface.co/incoai/GLM-5.3-DFlash2) drafter | TensorFold PR #159 (`glm_moe_dsa`, TP=4) + b12x RoCE | **Measured** (2026-10-05) on the stack this folder packages; this repo's own scripts are statically checked only, see [Measurement status](#measurement-status) |

## Performance

Four DGX Sparks (GB10, 128 GB unified memory each), one rank per Spark, one ConnectX-7 RoCE rail (MTU 1500). Default
profile `fast-160k`: `--context 163840`, bf16 KV cache whole on every rank (`TF_GLM53_DCP=1`), DFlash2 depth 7 /
confidence 0.60, b12x RoCE reductions, pinned tile table. One request at a time.

**Decode on real prompts.** The 6 reference prompts (code, prose, math, two chat, a 2,842-token document), greedy,
512 new tokens, thinking on. Measured through the `/v1` server with
[`bench/longctx_run.py`](bench/longctx_run.py), one serial and one drafted run per prompt. Engine-reported decode
rate.

| Measurement | Result |
|---|---:|
| **Decode with DFlash2, mean of 6 prompts** | **41.11 tok/s** |
| Decode with DFlash2, best prompt (math) | 56.50 tok/s |
| Decode with DFlash2, worst prompt (multi-turn chat) | 32.57 tok/s |
| Decode without drafting, mean of 6 prompts | 26.46 tok/s |
| Drafted reply == serial reply | 6 / 6 prompts |
| Serial reply == the 32K speed sweep's serial reply | 6 / 6 prompts |

Per-prompt rows: [`tensorfold-four-spark-tp4/README.md`](tensorfold-four-spark-tp4/README.md#decode-per-prompt).

**Long prompts.** Two Project Gutenberg books ([`bench/make_prompts.py`](bench/make_prompts.py)) as one user
message, thinking off, greedy, 512 new tokens, same server. TTFT is client-observed; prefill rate is prompt tokens
over TTFT.

| Measurement | Result |
|---|---:|
| 127,544-token prompt: decode with DFlash2 | **33.30 tok/s** |
| 127,544-token prompt: decode without drafting | 21.76 tok/s |
| 127,544-token prompt: drafted reply == serial reply | 512 / 512 tokens |
| 127,544-token prompt: time to first token (DFlash2 arm) | 511 s |
| 127,544-token prompt: prefill rate (DFlash2 arm) | 249.5 tok/s |
| 162,544-token prompt: decode without drafting | 20.82 tok/s |
| 162,544-token prompt: time to first token | 639 s |
| 162,544-token prompt: prefill rate | 254.4 tok/s |
| 162,544-token prompt: decode with DFlash2 | TODO(confirm): the run was stopped before the drafted arm |

**SixCat v0.7.0 speed suite** (`--policy strict --thinking off`, client-side, same server). Its decode profile is a
synthetic word-list prompt that drafts far better than real text: **a ceiling, not the rate a user sees** (compare
41.11 above on real prompts).

| Measurement | Result |
|---|---:|
| SixCat decode, C=1, per-stream p50 | 69.19 tok/s |
| SixCat prefill, ~2,476-token prompt, p50 | 313.4 tok/s |
| SixCat TTFT on that prompt, p50 | 7.90 s |

**Memory.** Lowest MemAvailable on any rank over the whole 160K session (load, SixCat, both long prompts):
**4.86 GiB**; swap growth **0** on all four. bf16 at 262,144 tokens unsplit does **not** fit (it swapped).

**Where the speed comes from** (2026-10-05 speed sweep: in-process harness driving the same engine, no HTTP,
context 32768, median of 3; [`bench/records/2026-10-05-sweep/`](bench/records/2026-10-05-sweep/README.md)):

| Measurement | Result |
|---|---:|
| Decode with DFlash2 d7/c0.60, RoCE, mean of 6 prompts | 41.42 tok/s |
| Same, after a re-boot on the pinned tile table | 41.24 tok/s |
| Same with NCCL reductions instead of RoCE | 37.07 tok/s |
| Decode without drafting, RoCE, code / prose / math mean | 27.13 tok/s |
| Decode without drafting, NCCL, code / prose / math mean | 24.15 tok/s |
| Drafted runs checked against serial ids | 283, 0 differences |
| Earlier: exllamav3 TP=4 on the same pack, no drafter, short prompts | 12.2–15.7 tok/s |
| Earlier: exllamav3 4-stage pipeline, SixCat decode | 8.05 tok/s |

The two exllamav3 rows are earlier runs on the same pack and Sparks, kept for scale; their records are not in this
repo.

**Other profiles** (`./glm53 up --profile NAME`; details in the
[folder README](tensorfold-four-spark-tp4/README.md#profiles)):

| Profile | Context | KV cache | Decode with DFlash2, mean of 6 | SixCat decode / prefill p50 | Status |
|---|---:|---|---:|---:|---|
| `fast-160k` (default) | 163,840 | bf16, whole on every rank | 41.11 tok/s | 69.19 / 313.4 tok/s | measured |
| `int4-262k` | 262,144 | int4 latent, whole on every rank | 40.62 tok/s TODO(confirm) | 68.18 / 307.9 tok/s TODO(confirm) | opt-in: quality gate not finished |
| `dcp4-262k` | 262,144 | bf16, split across ranks | 35.91 tok/s | 57.08 / 232.2 tok/s | measured; another numeric path |

The `int4-262k` figures were measured with the same TensorFold tree through a copy of `lib/` and the operator's own
launcher, not through this repo's entry scripts, so they stay TODO(confirm)
([record](bench/records/2026-10-05-serve-262k-int4/README.md)). Its teacher-forced quality gate against exllamav3's
Q4 cache passed on the prompt path (160 rows); the decode path is still being finished, so the profile needs
`--allow-unvalidated` until then.

### Measurement status

Every number above except the `int4-262k` row was measured on 2026-10-05 with the operator stack this folder
packages: the same TensorFold tree (byte-identical diff, checked by hash), the same `lib/` wrappers, pinned tile
table, b12x commit, drafter, BF16 lm_head and chat template. The raw records are in
[`bench/records/`](bench/records/README.md). This repo's `./glm53 init` → `setup` → `up` → `bench` sequence (and the
`setup.sh` / `serve.sh` scripts under it) has been checked statically (`bash -n`, `py_compile`, `--dry-run` of every
command and profile, the launcher with ssh and rsync replaced by stand-ins, and `init`'s fabric discovery against
simulated Sparks) but has **not** yet been run end to end on fresh Sparks: TODO(confirm) by one session that follows [Quick start](#quick-start) literally.

## Quick start

**What you need.** Four DGX Sparks cabled to one QSFP switch on the same ConnectX-7 port, with an IPv4 address on that
port (NVIDIA Sync's [Cluster Assistant](https://docs.nvidia.com/sync/latest/cluster-assistant.html) or NVIDIA's
[Multi Sparks Through a Switch](https://build.nvidia.com/spark/multi-sparks-through-switch) playbook sets this up; four
Sparks need a switch), the same user account on all four, and access to the gated
[pack](https://huggingface.co/vcruz305/GLM-5.3-EXL3-3.38bpw) on Hugging Face (request it on the model page). You do
**not** need to know the Sparks' names or addresses.

**0. Get a shell on any one of the four Sparks.** That Spark becomes rank 0 and serves the API. The ways NVIDIA documents
([Set Up Local Network Access](https://build.nvidia.com/spark/connect-to-your-spark)):

- `ssh <user>@<hostname>.local`: every Spark announces its hostname over mDNS (for example `spark-abcd.local`; the
  hostname is on the Quick Start Guide that came in the box);
- `ssh <user>@<ip>`: its address on your network (your router's admin page lists it) when mDNS does not resolve;
- NVIDIA Sync: add the Spark (Sync finds it over mDNS, or takes its IP), then open a terminal on it from Sync;
- or a display, keyboard and mouse on the Spark itself.

```bash
git clone https://github.com/vcruz305/GLM-5.3-EXL3-DGX-Spark-recipe.git && cd GLM-5.3-EXL3-DGX-Spark-recipe
./glm53 init                                    # finds the other three Sparks on the fabric, checks ssh to each
HF_TOKEN=hf_... ./glm53 setup --download-once   # all four in parallel; one 319 GB download, copied over the fabric
./glm53 up                                      # preflight, then ~8 min to READY (first start: longer)
./glm53 chat "Explain RoCE in two sentences."
```

- **`init`** runs on the Spark you are on and needs no names: it finds the ConnectX-7 port that is up with an IPv4
  address, lists the machines on that subnet (`ip neigh`, a ping sweep of a small subnet, mDNS via `avahi-browse` when
  installed), checks each over `ssh -o BatchMode=yes` (GPU must be a GB10, one per `/etc/machine-id`), requires exactly
  three others, ranks them by fabric IP and writes the hosts file with the fabric IPs as ssh targets. Run on a machine
  that is not a Spark, it stops and says how to continue.
- **Passwordless ssh from this Spark to the other three is required** (rank 0's watchdog stops the others through it,
  and `--download-once` copies over it). If it is missing, `init` prints the fix: NVIDIA's
  [`discover-sparks`](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/connect-two-sparks/assets/discover-sparks)
  script from the Connect Two Sparks / Multi Sparks playbooks (run once on this Spark; it asks for your password on
  each Spark and installs one shared key), or `ssh-copy-id` per Spark. NVIDIA Sync's Cluster Assistant also sets up
  ssh between the Sparks.
- **Run every later `./glm53` command on the same Spark.** `setup` copies this clone to the other three, runs
  `tensorfold-four-spark-tp4/setup.sh` on all four with one log each, and prints a summary with the fix for each
  failure; re-running it resumes. With `--model-dir /path/to/pack` an existing copy is checked and used instead.
- The API is `http://127.0.0.1:8890/v1` on rank 0, model id `GLM-5.3-EXL3-3.38bpw`. `./glm53 tunnel` prints the
  `ssh -L` command that brings it to your laptop.
- `./glm53 smoke`, `./glm53 bench` (the 6 reference prompts: ids and tok/s), `./glm53 status`, `./glm53 logs`,
  `./glm53 down`. `--dry-run` on any command prints every command (ssh, rsync, and for `init` each discovery step) and
  runs nothing. `./glm53 help` lists everything.
- **Driving from a laptop instead:** `./glm53 init --via <any one Spark>` runs the same discovery on that Spark and
  reaches the other three through it (your key must be on all four), and `./glm53 init --hosts H0,H1,H2,H3` names
  all four yourself, rank 0 first.

```bash
curl -s http://127.0.0.1:8890/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "GLM-5.3-EXL3-3.38bpw",
  "messages": [{"role": "user", "content": "Explain RoCE in two sentences."}],
  "max_tokens": 1024}'
```

- Requests draft with DFlash2 by default. `"draft": false` decodes serially (the exactness reference).
- Thinking is on by default. `"chat_template_kwargs": {"enable_thinking": false}` turns it off.
- Sampling defaults to the model's (temperature 1.0, top-p 0.95). The numbers above are greedy
  (`"temperature": 0`); the speed-up at temperature > 0 has not been measured.

Requirements, profiles, guards, every knob and the manual path (`setup.sh` on each Spark, then `serve.sh`):
[`tensorfold-four-spark-tp4/README.md`](tensorfold-four-spark-tp4/README.md).

## Upstream status

- **The runtime is [ashhart/TensorFold PR #159](https://github.com/ashhart/TensorFold/pull/159)** (`glm-moe-dsa-tp4`
  by [@drowzeys](https://github.com/drowzeys), open), plus my fixes on two branches of
  [vcruz305/TensorFold](https://github.com/vcruz305/TensorFold):
  [`glm53-tp4-spark`](https://github.com/vcruz305/TensorFold/tree/glm53-tp4-spark) @ `757a851` (the loader fixes this
  pack needs: per-layer handle and page-cache release, fail-closed fp16 → bf16 cast; the default profiles) and
  [`glm53-kv-int4`](https://github.com/vcruz305/TensorFold/tree/glm53-kv-int4) @ `0c858e3` (those plus the int4 / int8
  latent KV cache; `int4-262k`).
- **The same changes are open as PRs into PR #159's branch** (`drowzeys/TensorFold` `glm-moe-dsa-tp4`):
  [#1](https://github.com/drowzeys/TensorFold/pull/1) `glm53-gb10-loading` (GB10 loading fixes),
  [#2](https://github.com/drowzeys/TensorFold/pull/2) `glm53-dflash-default` (DFlash2 by default),
  [#3](https://github.com/drowzeys/TensorFold/pull/3) `glm53-pinned-tiles` (pinned tile table),
  [#4](https://github.com/drowzeys/TensorFold/pull/4) `glm53-roce-health` (RoCE health check, RoCE setup barrier) and
  [#5](https://github.com/drowzeys/TensorFold/pull/5) `glm53-kv-cache-int4` (the int4 / int8 KV cache, stacked on #1).
- **This recipe pins the fork commits** (`env.sh`: `757a851`, and `0c858e3` for `int4-262k`), so it works before any
  of that lands. The pins move only after a session re-measures the new tree through these scripts.

## Quants

The pack is published at [vcruz305/GLM-5.3-EXL3-3.38bpw](https://huggingface.co/vcruz305/GLM-5.3-EXL3-3.38bpw); this
table mirrors its model card.

| bpw | size | smallest card | top-1 vs original | mean KLD | p99 KLD | download |
|---|---:|---|---:|---:|---:|---|
| 3.38 (SAGE MixedK, 8-bit head) | 319.0 GB | 4 × DGX Spark | 92.98% | 0.0948 | 1.61 | [`vcruz305/GLM-5.3-EXL3-3.38bpw`](https://huggingface.co/vcruz305/GLM-5.3-EXL3-3.38bpw) (gated) |

Top-1 is next-token agreement with the original BF16 checkpoint on the card's held-out set (10 sequences of 1,024
tokens from the test splits of WikiText-103, HumanEval, GSM8K and UltraChat-200k). KLD is the mean and the
99th-percentile KL(BF16 ‖ pack) over those positions. That evaluation text was not used for quantization. Each result
is tied to one pack and one runtime: the card scored the pack with its own 8-bit head through ExLlamaV3, while this
recipe serves the original BF16 head through TensorFold, so the card's figures are not this server's.

## Quality

TensorFold (this engine, BF16 head and bf16 cache) against exllamav3 TP=4 on the same pack, teacher-forced on 43,903
prompt positions (16 natural 2,560-token rows plus 3 SixCat rows) and 2,424 decode positions: **PASS, with two
REVIEW flags** on the synthetic SixCat rows (top-1 agreement at margin ≥ 0.5 is 3–4 points below exllamav3's own
rounding control there).

| Measurement | Result |
|---|---:|
| Prompt path, all rows: mean NLL change vs exllamav3 | −0.0030 nats (95% CI −0.0243 … +0.0215) |
| Prompt path, natural rows: mean NLL change | +0.0068 nats (95% CI −0.0033 … +0.0241) |
| Prompt path, all rows: top-1 change | −0.0006 |
| Decode path, natural rows: mean NLL change | +0.0054 nats (95% CI −0.0038 … +0.0156) |
| Greedy, first 4 tokens on 3 SixCat prompts | 3 / 3 equal to the baseline |

Read with these limits ([`bench/records/2026-10-05-quality-g1/`](bench/records/2026-10-05-quality-g1/REPORT_RESULTS.md)):
it compares deployments (BF16 head + bf16 cache against the pack's 8-bit head + Q4 cache), ran NCCL reductions at
context 4,096 without drafting, and used that boot's own tile picks, **not** the pinned tile table this recipe
serves. Drafting does not change outputs (exactness above), but the pinned table changes summation order. A larger
gate (G2) on the pinned table was started and stopped before it finished: TODO(confirm).

## Repository layout

| Path | What it is |
|---|---|
| [`glm53`](glm53) | **The entry point**: `init`, `setup`, `up`, `chat`, `smoke`, `bench`, `status`, `logs`, `tunnel`, `down` for all four Sparks from one machine (runs `tensorfold-four-spark-tp4/run.sh`) |
| [`tensorfold-four-spark-tp4/`](tensorfold-four-spark-tp4/README.md) | **The recipe.** `run.sh` (the launcher), `setup.sh`, `serve.sh`, `rank.sh`, `chat.sh`, `drop-model-cache.sh`, `env.sh`, `profiles/`, `hosts.example`, the engine launcher and its wrappers (`lib/`), the pinned tile table (`tiles/`), `tools/` |
| [`bench/`](bench/) | Engine-neutral `/v1` clients: `bench_v1.py` (smoke + the 6-prompt benchmark), `longctx_run.py` + `make_prompts.py` (long prompts), the reference prompts and ids |
| [`bench/records/`](bench/records/README.md) | The measurement records behind every number in this README |
| [`docs/`](docs/README.md) | A static benchmark viewer over `docs/benchmark-data.json` |
| [`AGENTS.md`](AGENTS.md) | The exact commands and the "do not" list for an AI agent running this recipe |

## Do not

- **Do not guess host names or addresses**, and do not hand-write the hosts file. `./glm53 init` on a Spark finds the
  others; use `--hosts` only with addresses read from the Sparks themselves.
- **Do not ask for passwords or tokens in chat, and do not type them for the user.** `discover-sparks`,
  `ssh-copy-id`, `sudo` and `hf auth login` prompt in the user's terminal.
- **Do not install TensorFold from PyPI, a release tag or `ashhart/TensorFold` main.** None of them has the
  `glm_moe_dsa` family yet (PR #159 is open), and PR #159 alone does not load this pack (host memory and fp16
  tensors, see the folder README). `setup.sh` checks out the pinned `vcruz305/TensorFold` commits; every launcher
  refuses another tree.
- **Do not repoint the pins at the upstream PR branches** (`glm53-gb10-loading` and the rest, open as
  [drowzeys/TensorFold #1 to #5](https://github.com/drowzeys/TensorFold/pulls)). They rename functions
  the runtime check looks for, and none of them has been measured through these scripts.
- **Do not `pip install b12x`.** PyPI 1.3.0 has no `comm.roce` module. `setup.sh` stages commit `b58f34e`.
- **Do not serve the pack directory.** TensorFold needs the BF16 `lm_head.weight` and the fixed chat template; both
  live in the serve view `setup.sh` builds. The pack is never edited.
- **Do not run bf16 at 262,144 tokens with `TF_GLM53_DCP=1`.** It swapped and tripped the watchdog. Use the default
  profile, `--profile int4-262k --allow-unvalidated` or `--profile dcp4-262k`.
- **Do not lower `TF_GLM53_CACHE_RESERVE_GB`, disable the watchdog or allow swap growth** to make a context fit. A
  Spark that swaps under this load can hang until its watchdog reboots it.
- **Do not pass `--parallel`.** The server answers one request at a time; `--parallel N>1` switches TensorFold to a
  different scheduler that was never measured with this recipe.
- **Do not use `--profile int4-262k` / `--allow-unvalidated`** unless asked to: its quality gate is not finished.
- **Do not edit the TensorFold checkouts** under `~/glm53-tensorfold/`. `verify_runtime` refuses a dirty or moved tree.
- **Do not start while another job holds a GPU** on any of the four Sparks. `preflight` and `rank.sh start` refuse.
- **Do not put a Hugging Face token on a command line** that is logged or shared; `HF_TOKEN=... ./glm53 setup` passes
  it to the Sparks over ssh stdin only.
- **Do not expose port 8890 beyond loopback** unless the user asks, and then only with `TFS_API_KEY_FILE`.
- **Do not quote the SixCat decode figure as user-facing speed**, and do not redistribute the DFlash2 weights
  (CC BY-NC-ND 4.0: non-commercial use only).

## Troubleshooting

Every `./glm53` command that fails prints a `To fix:` block with the command for each failure it recognises; the full
output stays in `tensorfold-four-spark-tp4/runs/`.

| Symptom | Cause / fix |
|---|---|
| `no hosts file yet` | `./glm53 init` on one of the four Sparks (it finds the other three), then run every `./glm53` command there |
| `init`: `This machine ... is not a DGX Spark` | `init` must run on a Spark: ssh to any one of the four (step 0 of [Quick start](#quick-start)) and run it there, or `./glm53 init --via <one Spark>` from here |
| `init`: `Permission denied`, `found 0 of the other three` / `rank 0 -> rank N: FAILED` | passwordless ssh from rank 0 to the others is missing: run the printed NVIDIA `discover-sparks` line (or `ssh-copy-id` per Spark) in a terminal on rank 0, then `./glm53 init --force`. Rank 0's watchdog stops the peers over ssh, and `--download-once` copies over it |
| `init`: `found 2 of the other three` / `only N ... answered` | a Spark is off, cabled on another port, or has no address on the fabric port; on a 169.254.x.x (link-local) fabric install `avahi-utils` so `init` can use mDNS. Or `./glm53 init --hosts local,<ip>,<ip>,<ip> --force` |
| `init`: `exactly four Sparks and 5 answered` | more than four Sparks on the fabric: choose four with `./glm53 init --hosts local,<ip>,<ip>,<ip> --force` |
| `init`: `no ConnectX-7 port is Up` / `has no IPv4 address yet` | cable and address the fabric first: NVIDIA Sync's Cluster Assistant or the [Multi Sparks Through a Switch](https://build.nvidia.com/spark/multi-sparks-through-switch) playbook; `FABRIC_IFNAME=<port> ./glm53 init` picks a port |
| `setup`: `no Hugging Face token` | request access to the pack, then `HF_TOKEN=hf_... ./glm53 setup` |
| `setup`: `no Python.h`, `gcc + libibverbs`, `nvcc not found` | `sudo apt install -y libpython3.12-dev` / `gcc libibverbs-dev`, or `CUDA_HOME=<CUDA 13>`; then `./glm53 setup` again |
| `TensorFold at ... is <sha>, the recipe pins ...` / `lacks the measured loader fixes` | another TensorFold tree; `./glm53 setup` |
| `recipe clone at <sha>, here <sha>` | `./glm53 sync` |
| `no b12x RoCE module staged` | `./glm53 setup` (needs `gcc` + `libibverbs-dev`), or `TFS_ROCE=0` for NCCL reductions (measured 37.07 vs 41.42 tok/s) |
| `REFUSE: no RoCE v2 GID for <ip>` | the hosts file's `fabric_ip` is not this Spark's address on `FABRIC_IFNAME`, or the RDMA device is not `ROCE_HCA`: `./glm53 init --force` re-detects both |
| `3/4 clients joined` (or a peer's `client socket has timed out`), then `FATAL ... RoCE requested` | a rank loaded minutes after the others and missed b12x's 120 s setup rendezvous. Fixed by the rendezvous barrier in `tensorfold-four-spark-tp4/lib/tf_serve_patches.py` (every rank waits until all four have loaded); seeing it means this clone predates it: update the recipe (`git pull`, `./glm53 sync`), then `./glm53 down` and `./glm53 up` |
| `context ... needs ... GiB of caches a rank, ... is free` / `LOW MEMORY` | page cache from a download or another load: `./glm53 fadvise`, stop other jobs |
| `profile int4-262k is not validated yet` | intended; `--allow-unvalidated` only when asked to run it |
| `START FAILED` / `FATAL` in a rank log | `./glm53 logs`, read `rank*.log` (`[tf_serve] FATAL ...`) and the watchdog log, then `./glm53 down` and `./glm53 preflight` |
| replies end with `finish_reason: length`, only `reasoning_content` | the reply spent `max_tokens` thinking: raise it or send thinking off |
| decode far below the tables | check the reply's `tensorfold` stats (`mtp_mode` `dflash`, `depth` 7, `confidence` 0.6) and the rank 0 log line `decode-window reductions: RoCE one-shot` |

More, per profile and per guard: [`tensorfold-four-spark-tp4/README.md`](tensorfold-four-spark-tp4/README.md#troubleshooting).

## Credits

- **TensorFold** by [@ashhart](https://github.com/ashhart/TensorFold) and contributors (Apache-2.0 from 0.6.0): the
  exact-drafting engine, its CUDA kernels and server.
- **PR #159** by [@drowzeys](https://github.com/drowzeys) (keys): the full GLM-5.3 `glm_moe_dsa` family and its TP=4
  CUDA engine on four Sparks, including the b12x RoCE integration. This recipe adds two loader fixes on top and wraps
  the PR's own `tensorfold serve`.
- **b12x** by [local-inference-lab](https://github.com/local-inference-lab/b12x) (Luke Alonso, Apache-2.0): the RoCE
  one-shot all-reduce ("RoCEnante").
- **GLM-5.3-DFlash2** by [incoai](https://huggingface.co/incoai/GLM-5.3-DFlash2) (CC BY-NC-ND 4.0, non-commercial):
  the drafter. Downloaded from its own repo, never redistributed here.
- **GLM-5.3** by [zai-org](https://huggingface.co/zai-org/GLM-5.3) (GLM-5.3 license): the model, its chat template
  and the BF16 lm_head this recipe serves.
- **EXL3** by [turboderp](https://github.com/turboderp-org/exllamav3): the quantization format of the pack.

## License

MIT for the scripts and notes in this repo (see [`LICENSE`](LICENSE)). Weights are **not** redistributed here: the
pack follows the GLM-5.3 license, the DFlash2 drafter is CC BY-NC-ND 4.0 (non-commercial use only), and TensorFold
and b12x are Apache-2.0.
