# GLM-5.3 EXL3 on NVIDIA DGX Spark

Serves [zai-org/GLM-5.3](https://huggingface.co/zai-org/GLM-5.3) (78 layers, 256 routed experts per MoE layer, DSA
sparse attention) from the SAGE MixedK EXL3 pack
[vcruz305/GLM-5.3-EXL3-3.38bpw](https://huggingface.co/vcruz305/GLM-5.3-EXL3-3.38bpw) (3.38 bpw, 8-bit head,
319 GB) on **four NVIDIA DGX Sparks**, as an OpenAI-compatible `/v1` API.

The engine is [TensorFold](https://github.com/ashhart/TensorFold)'s full-GLM-5.3 tensor-parallel CUDA engine
([PR #159](https://github.com/ashhart/TensorFold/pull/159) by [@drowzeys](https://github.com/drowzeys), still open),
plus two loader fixes this pack needs. It runs on the
[`glm53-tp4-spark`](https://github.com/vcruz305/TensorFold/tree/glm53-tp4-spark) branch of my TensorFold fork. Decode
reductions go over the ConnectX-7 fabric with [b12x](https://github.com/local-inference-lab/b12x)'s RoCE one-shot
all-reduce. Drafting uses the [DFlash2](https://huggingface.co/incoai/GLM-5.3-DFlash2) drafter. **Drafted replies are
token-identical to serial decoding:** every measured drafted reply equals the `"draft": false` reply, token for token.

> **Using an AI agent to set this up?** It should read [Do not](#do-not) and
> [Troubleshooting](#troubleshooting) before running anything, then follow [Quick start](#quick-start) literally.

| Sparks | Folder | Model | Engine | Status |
|---|---|---|---|---|
| **4** | [`tensorfold-four-spark-tp4/`](tensorfold-four-spark-tp4/README.md) | [`GLM-5.3-EXL3-3.38bpw`](https://huggingface.co/vcruz305/GLM-5.3-EXL3-3.38bpw) + [DFlash2](https://huggingface.co/incoai/GLM-5.3-DFlash2) drafter | TensorFold PR #159 (`glm_moe_dsa`, TP=4) + b12x RoCE | **Measured** (2026-10-05) on the stack this folder packages; this repo's own scripts are statically checked only, see [Measurement status](#measurement-status) |

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
repo. The other profiles (262K with decode context parallelism, measured; int4 KV cache, pending validation) are in
the [folder README](tensorfold-four-spark-tp4/README.md#profiles).

### Measurement status

Every number above was measured on 2026-10-05 with the operator stack this folder packages: the same TensorFold tree
(byte-identical diff, checked by hash), the same `lib/` wrappers, pinned tile table, b12x commit, drafter, BF16
lm_head and chat template. The raw records are in [`bench/records/`](bench/records/README.md). This repo's
`setup.sh` → `serve.sh up` → `serve.sh bench` sequence has been checked statically (`bash -n`, `py_compile`,
`DRY_RUN=1` of every step and profile) but has **not** yet been run end to end on fresh Sparks:
TODO(confirm) by one session that follows [Quick start](#quick-start) literally.

## Quick start

On **each of the four Sparks**, in a clone at the same path:

```bash
git clone https://github.com/vcruz305/GLM-5.3-EXL3-DGX-Spark-recipe.git
cd GLM-5.3-EXL3-DGX-Spark-recipe
hf auth login        # the pack is gated: request access on its Hugging Face page first
bash tensorfold-four-spark-tp4/setup.sh
```

`setup.sh` builds a venv (torch cu130), checks out the pinned TensorFold commit, stages b12x, downloads the pack
(319 GB), the drafter and the BF16 lm_head (1.9 GB, range-read from zai-org/GLM-5.3), and builds the serve view.
Re-running it is safe; downloads resume.

Then, on the machine that will drive the cluster (rank 0 itself is fine):

```bash
cp tensorfold-four-spark-tp4/hosts.example tensorfold-four-spark-tp4/hosts
$EDITOR tensorfold-four-spark-tp4/hosts                # rank, ssh target, fabric IP for each Spark
bash tensorfold-four-spark-tp4/serve.sh preflight      # must print "preflight OK on all four"
bash tensorfold-four-spark-tp4/serve.sh up             # ~7-8 min: watchdogs, ranks 1-3, rank 0, "READY"
bash tensorfold-four-spark-tp4/serve.sh smoke
```

The API is `http://127.0.0.1:8890/v1` on rank 0, model id `GLM-5.3-EXL3-3.38bpw` (`serve.sh tunnel` prints an ssh
tunnel command for another machine):

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
- `bash tensorfold-four-spark-tp4/serve.sh down` stops all four. `DRY_RUN=1` on any `serve.sh` step prints the commands.

Requirements, profiles, guards and every knob: [`tensorfold-four-spark-tp4/README.md`](tensorfold-four-spark-tp4/README.md).

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
| [`tensorfold-four-spark-tp4/`](tensorfold-four-spark-tp4/README.md) | **The recipe.** `setup.sh`, `serve.sh`, `rank.sh`, `chat.sh`, `drop-model-cache.sh`, `env.sh`, `profiles/`, `hosts.example`, the launcher and its wrappers (`lib/`), the pinned tile table (`tiles/`), `tools/` |
| [`bench/`](bench/) | Engine-neutral `/v1` clients: `bench_v1.py` (smoke + the 6-prompt benchmark), `longctx_run.py` + `make_prompts.py` (long prompts), the reference prompts and ids |
| [`bench/records/`](bench/records/README.md) | The measurement records behind every number in this README |
| [`docs/`](docs/README.md) | A static benchmark viewer over `docs/benchmark-data.json` |

## Do not

- **Do not install TensorFold from PyPI, a release tag or `ashhart/TensorFold` main.** None of them has the
  `glm_moe_dsa` family yet (PR #159 is open), and PR #159 alone does not load this pack (host memory and fp16
  tensors, see the folder README). `setup.sh` checks out the pinned `vcruz305/TensorFold` commit; every launcher
  refuses another tree.
- **Do not `pip install b12x`.** PyPI 1.3.0 has no `comm.roce` module. `setup.sh` stages commit `b58f34e`.
- **Do not serve the pack directory.** TensorFold needs the BF16 `lm_head.weight` and the fixed chat template; both
  live in the serve view `setup.sh` builds. The pack is never edited.
- **Do not run bf16 at 262,144 tokens with `TF_GLM53_DCP=1`.** It swapped and tripped the watchdog. Use the default
  profile, or `PROFILE=dcp4-262k`.
- **Do not lower `TF_GLM53_CACHE_RESERVE_GB`, disable the watchdog or allow swap growth** to make a context fit. A
  Spark that swaps under this load can hang until its watchdog reboots it.
- **Do not pass `--parallel`.** The server answers one request at a time; `--parallel N>1` switches TensorFold to a
  different scheduler that was never measured with this recipe.
- **Do not use `PROFILE=int4-262k`** unless asked to validate it. It is pending validation and needs
  `ALLOW_UNVALIDATED=1`.
- **Do not edit the TensorFold checkout** under `~/glm53-tensorfold/`. `verify_runtime` refuses a dirty or moved tree.
- **Do not start while another job holds a GPU** on any of the four Sparks. `preflight` and `rank.sh start` refuse.
- **Do not quote the SixCat decode figure as user-facing speed**, and do not redistribute the DFlash2 weights
  (CC BY-NC-ND 4.0: non-commercial use only).

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `TensorFold at ... is <sha>, the recipe pins ...` / `lacks the measured loader fixes` | another TensorFold tree; re-run `setup.sh` |
| `no b12x RoCE module staged` | re-run `setup.sh` (needs `gcc` + `libibverbs-dev`), or `TFS_ROCE=0` for NCCL reductions (measured 37.07 vs 41.42 tok/s) |
| `REFUSE: no RoCE v2 GID for <ip>` | the hosts file's `fabric_ip` is not this Spark's address on `FABRIC_IFNAME`, or the RDMA device is not `ROCE_HCA` |
| `context ... needs ... GiB of caches a rank, ... is free` | page cache from a download or another load: `serve.sh fadvise` (or `drop-model-cache.sh` on that Spark), stop other jobs |
| `rank 0 cannot 'ssh -o BatchMode=yes <peer>'` | rank 0's watchdog stops the peers over ssh: set up keys from rank 0 to ranks 1-3, or a `peer_ssh` column in the hosts file |
| `START FAILED` / `FATAL` in a rank log | `serve.sh logs`, read `rank*.log` (`[tf_serve] FATAL ...`) and the watchdog log, then `serve.sh down` and `preflight` |
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
