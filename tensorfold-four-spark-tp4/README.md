# Four DGX Sparks: TensorFold TP=4

**Model:** [`vcruz305/GLM-5.3-EXL3-3.38bpw`](https://huggingface.co/vcruz305/GLM-5.3-EXL3-3.38bpw) (SAGE MixedK,
3.38 bpw, revision `cc64e77`, 58 shards, 319 GB) + the original BF16 `lm_head.weight` from
[zai-org/GLM-5.3](https://huggingface.co/zai-org/GLM-5.3) (revision `aca966e`)
**Drafter:** [`incoai/GLM-5.3-DFlash2`](https://huggingface.co/incoai/GLM-5.3-DFlash2) (revision `425aa61`,
CC BY-NC-ND 4.0: non-commercial use only)
**Engine:** TensorFold PR #159 (`glm_moe_dsa`, TP=4) + two loader fixes =
[`vcruz305/TensorFold@757a851`](https://github.com/vcruz305/TensorFold/tree/glm53-tp4-spark); for `int4-262k` also the
int4 KV cache, [`vcruz305/TensorFold@0c858e3`](https://github.com/vcruz305/TensorFold/tree/glm53-kv-int4); b12x
`b58f34e` RoCE reductions
**Entry point:** [`../glm53`](../glm53) (this folder's `run.sh`): `init`, `setup`, `up`, `chat`, ... from one of the
Sparks (`init` finds the other three)
**Status:** `fast-160k` and `dcp4-262k` measured 2026-10-05 on the stack these scripts package; `int4-262k` measured
through a copy of `lib/` (TODO(confirm)) with its quality gate unfinished; the scripts themselves are statically
checked only (see the root README's [Measurement status](../README.md#measurement-status))

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

`./glm53 up --profile NAME` (or `PROFILE=NAME` for `serve.sh`) selects `profiles/<name>.env` on every Spark.
`setup` installs both TensorFold trees, so switching profiles needs only `./glm53 down` and `./glm53 up --profile ...`.

| Profile | Context | KV cache | TensorFold | Status |
|---|---:|---|---|---|
| `fast-160k` (default) | 163,840 | bf16, whole on every rank (`TF_GLM53_DCP=1`) | `glm53-tp4-spark` `757a851` | **measured**, numbers above |
| `int4-262k` | 262,144 | int4 latent, whole on every rank | `glm53-kv-int4` `0c858e3` | speed measured through a copy of `lib/` (TODO(confirm)); quality gate not finished: opt-in (`--allow-unvalidated`) |
| `dcp4-262k` | 262,144 | bf16, split across ranks (`TF_GLM53_DCP=4`) | `glm53-tp4-spark` `757a851` | measured, below |

The default is the `DEFAULT_PROFILE=` line in [`env.sh`](env.sh). The default profile is always treated as validated,
so making `int4-262k` the default once its quality gate passes is that one line.

### `int4-262k`: 262,144 tokens with an int4 KV cache

An int4 MLA latent cache in exllamav3's Q4 cache format (the 512-wide latent in 32-value groups with fp16 scales; RoPE
dims and indexer keys stay 16-bit): 9.234 GiB a rank at 262,144 tokens, against 23.25 GiB for bf16, which does not
fit unsplit. The writer is bit-identical to exllamav3's quantizer on all four Sparks (with the approximate reciprocal
exllamav3's fast-math build uses). int4 changes the numerics, so its greedy ids are not the bf16 ids; drafted replies
still equal this server's own serial replies, and the pack's quality figures do not carry over.

**TODO(confirm):** measured 2026-10-05 with the same TensorFold tree (`0c858e3`, checked file by file) through a copy
of `lib/` and the operator's own launcher, not through `./glm53`; record:
[`bench/records/2026-10-05-serve-262k-int4/`](../bench/records/2026-10-05-serve-262k-int4/README.md).

| Measurement | `int4-262k` | `fast-160k` |
|---|---:|---:|
| Decode with DFlash2, mean of 6 prompts | 40.62 tok/s (median of 3) | 41.11 tok/s (one run) |
| code / prose / math | 38.94 / 33.07 / 57.98 | 40.64 / 35.71 / 56.50 |
| chat_explain / chat_multiturn / long_doc | 37.66 / 33.39 / 42.71 | 37.67 / 32.57 / 43.54 |
| Decode without drafting, mean of 6 prompts | 25.31 tok/s | 26.46 tok/s |
| Drafted == this server's serial | 18 / 18 | 6 / 6 |
| SixCat decode, C=1, p50 | 68.18 tok/s | 69.19 tok/s |
| SixCat prefill, ~2,474-token prompt, p50 | 307.9 tok/s | 313.4 tok/s |
| SixCat TTFT on that prompt, p50 | 8.04 s | 7.90 s |
| KV cache per rank | 9.234 GiB | |
| Lowest MemAvailable while serving | 10.18 GiB | 4.86 GiB |
| Lowest MemAvailable during the first boot | 6.35 GiB (kernel builds + graph capture) | |

Serial decode is ~5% slower than bf16 (int4 decode attention dequantizes and rotates every key: +1.8 ms a token);
drafted decode lands within 2% of bf16. Quality: teacher-forced against exllamav3 TP=4 with its Q4 cache, the prompt
path **passed** (160 rows, mean NLL +0.126%, 95% CI −0.017 … +0.264%); the decode path is still being finished, which
is why the profile needs `--allow-unvalidated`. Not measured: prompts longer than 2,842 tokens at this context.

```bash
./glm53 up --profile int4-262k --allow-unvalidated     # setup already installed the int4 tree
./glm53 bench                                          # drafted == this server's serial (BENCH_IDS=self)
```

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

`bench` on this profile compares drafted replies with the server's own serial replies (`BENCH_IDS=self`), not with
the 32K reference.

**bf16 at 262,144 with `TF_GLM53_DCP=1` does not fit.** The cache guard refused on ranks 0 and 1 (21.9 GiB of
caches needed, 27.6 / 27.8 GiB free with the 6 GiB reserve); on ranks 2 and 3 it passed, MemAvailable fell to
4.60 / 4.69 GiB, swap grew 4 kB, and the watchdog stopped all four
([`bench/records/2026-10-05-serve-262k-dcp1-failed/`](../bench/records/2026-10-05-serve-262k-dcp1-failed/)).

## Runtime identity

| | |
|---|---|
| engine | [ashhart/TensorFold PR #159](https://github.com/ashhart/TensorFold/pull/159) (`drowzeys/TensorFold` `glm-moe-dsa-tp4` @ `689596d`) + 2 commits = [`vcruz305/TensorFold` `glm53-tp4-spark` @ `757a851`](https://github.com/vcruz305/TensorFold/commits/glm53-tp4-spark): per-layer `safe_open` release + `malloc_trim`, per-shard page-cache drop before the cache guard, fail-closed fp16 → bf16 cast. `git -c core.abbrev=7 diff 689596d 757a851` has sha256 `1555d8ad…`, byte-identical to the measured tree's diff |
| engine, `int4-262k` | [`vcruz305/TensorFold` `glm53-kv-int4` @ `0c858e3`](https://github.com/vcruz305/TensorFold/commits/glm53-kv-int4) = `757a851` + `--kv-dtype int8/int4` for the MLA latent cache + two Triton compile fixes found on the GPU. `git -c core.abbrev=7 diff 689596d 0c858e3` has sha256 `d3866de2…`, the tree that was measured |
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
`rsync`, ~330 GB free disk for the pack, drafter and head, and at least 100 GiB of MemAvailable before a start. The
four Sparks sit on one QSFP switch (four Sparks need one) on the same ConnectX-7 port, with an IPv4 address on that
port, as NVIDIA Sync's [Cluster Assistant](https://docs.nvidia.com/sync/latest/cluster-assistant.html) or the
[Multi Sparks Through a Switch](https://build.nvidia.com/spark/multi-sparks-through-switch) playbook leaves them (one
RoCE rail was measured). The same user account on all four, and rank 0 can ssh to ranks 1-3 without a password (its
watchdog stops them on a breach). Every rank reads every shard, so **each Spark holds the full pack**.

**The driver** (where you type `./glm53`) is normally one of the Sparks: `init` run there finds the other three and that
Spark becomes rank 0. Any Linux / macOS machine with `bash` 4 or newer, `ssh` and `rsync` can drive instead, with
`init --via <one Spark>` or `init --hosts H0,H1,H2,H3` and passwordless ssh to all four.

```bash
git clone https://github.com/vcruz305/GLM-5.3-EXL3-DGX-Spark-recipe.git && cd GLM-5.3-EXL3-DGX-Spark-recipe
./glm53 init                                            # on a Spark: finds the other three, checks ssh, writes hosts
HF_TOKEN=hf_... ./glm53 setup --download-once           # all four in parallel; logs in tensorfold-four-spark-tp4/runs/
./glm53 up                                              # preflight, page cache, watchdogs, ranks 1-3, rank 0, READY
./glm53 chat "Explain RoCE in two sentences."
```

- **`init`** ([`tools/discover.sh`](tools/discover.sh) does the looking, `run.sh` decides):
  1. this machine must report a GB10 in `nvidia-smi`; anywhere else `init` stops and says to run it on a Spark, or to
     use `--via` / `--hosts`;
  2. the fabric port is a ConnectX-7 port (`ibdev2netdev`, else `/sys/class/infiniband`) that is Up with an IPv4
     address: `enp1s0f*` before `enP2p1s0f*`, a static address before a 169.254 link-local one, or the
     `FABRIC_IFNAME` you set;
  3. candidates on that port's subnet come from `ip -4 neigh`, `avahi-browse _ssh._tcp` (the mDNS service NVIDIA's
     `discover-sparks` uses; only when `avahi-utils` is installed) and one ping per address when the subnet has at
     most 1,024 addresses (a 169.254.0.0/16 link-local fabric is not swept);
  4. each candidate is probed over `ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new` (the first contact
     records the host key, as `discover-sparks` does): `/etc/machine-id`, GPU, its address on the same port, the RDMA
     device. A candidate that refuses the key is retried through any `~/.ssh/config` alias whose `HostName` is that
     address. Non-GB10 machines and addresses without ssh (a switch) are listed and skipped; one Spark per machine-id;
  5. exactly three other Sparks are required (two or five stop with the reason and the fix); ranks 1-3 are ordered
     by fabric IP; `hosts` gets `local` for rank 0 and the fabric IPs as ssh targets; `cluster.env` gets
     `FABRIC_IFNAME` and `ROCE_HCA`.

  Missing passwordless ssh shows up as `Permission denied` in step 4; `init` then prints NVIDIA's fix, run once on
  rank 0 in a terminal (it asks for the account password of each Spark):
  `curl -fsSLO https://raw.githubusercontent.com/NVIDIA/dgx-spark-playbooks/refs/heads/main/nvidia/connect-two-sparks/assets/discover-sparks && bash ./discover-sparks`
  ([script](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/connect-two-sparks/assets/discover-sparks):
  finds the Sparks over avahi, needs `avahi-utils`, and puts one shared key, `~/.ssh/id_ed25519_shared`, on all of
  them), or `ssh-copy-id <ip>` per Spark. `--dry-run` prints each discovery command and runs none.
- **`init --via SPARK`** (from a machine that is not a Spark) runs the same discovery on `SPARK` over ssh, makes it
  rank 0 and writes `ssh_config` (git-ignored, recorded as `SSH_CONFIG` in `cluster.env`) that reaches ranks 1-3
  through it with `ProxyJump`; your key must be on all four (`ssh-copy-id -o ProxyJump=SPARK <ip>` is printed when it
  is not).
- **`init --hosts H0,H1,H2,H3`** skips discovery: the four ssh targets in rank order (`local` = this machine). It probes
  each one (GB10, distinct machine-id, the fabric port rank 0 reports Up with an address unless `FABRIC_IFNAME` is set)
  and checks ssh from rank 0 to ranks 1-3. `--fabric-ips` skips the address detection; `--peer-ssh` fills the optional
  fourth column (how rank 0 reaches a peer, default its fabric IP).
- **`setup`** copies this clone to every Spark (`rsync`, into the same path below the home directory; `RECIPE_SYNC=git`
  clones it instead), then runs `setup.sh` on all four in parallel with a log per Spark and a summary that names the
  fix for every failure. `setup.sh` builds a venv (torch cu130), checks out both pinned TensorFold trees, stages b12x,
  downloads the pack, drafter and BF16 lm_head, and builds the serve view. Re-running it is safe; downloads resume.
  - `--download-once`: rank 0 downloads (319 GB) while ranks 1-3 build their runtime; then rank 0 `rsync`s the pack,
    drafter and lm_head to ranks 1-3 over the fabric (resumable) and they build their views.
  - `--model-dir /path/to/GLM-5.3-EXL3-3.38bpw`: the pack is already there (on every Spark, or on rank 0 with
    `--download-once`); it is checked (58 shards, each as long as its safetensors header says) instead of downloaded.
    The path is kept in `cluster.env` for every later step.
  - The pack is gated: request access on its Hugging Face page first. `HF_TOKEN=hf_... ./glm53 setup` writes your
    token (over ssh stdin, never a command line) to the Sparks that download and have none; or run
    `~/glm53-tensorfold/venv/bin/hf auth login` on them.
- **`up`** runs `preflight` (recipe commit, idle GPU, MemAvailable, ports, RoCE GID, runtime pin, view shas, launcher dry
  run on every Spark) and then `serve.sh up`. Ranks 1-3 start first, then rank 0. Each rank loads 78 layers (about
  400 s), sets up RoCE, loads the pinned tiles and the drafter and captures its decode graphs; rank 0 then runs a
  16-token serial and a 16-token drafted warm-up and starts HTTP. The first start on a Spark also JIT-builds
  TensorFold's CUDA kernels.
- Then `./glm53 smoke` (`/v1` checks), `./glm53 bench` (the 6 reference prompts: ids and tok/s against the tables
  above), `./glm53 status`, `./glm53 logs` (rank logs, watchdog logs and bench JSON into `runs/<time>/`),
  `./glm53 tunnel` (on rank 0: the `ssh -L` command for your laptop; from another driver, `--open` runs it),
  `./glm53 down`. Those steps reuse the profile
  of the last `up`. `./glm53 check` re-verifies every install; `./glm53 sync` re-copies this clone after a `git pull`.
- `--dry-run` on any command prints every ssh and rsync command (and, for `init`, each discovery command) and runs
  nothing.

**Without the launcher**, the same thing by hand: `bash tensorfold-four-spark-tp4/setup.sh` on each Spark (same clone
path on all four), `cp hosts.example hosts` and edit it on the driver, then `serve.sh preflight`, `serve.sh up`,
`serve.sh smoke` / `bench`, `chat.sh` on rank 0, `serve.sh down`. `DRY_RUN=1` before any `serve.sh` step prints its
commands.

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
| `../glm53` → `run.sh` | the launcher: `init` / `setup` / `sync` / `check` on all four from one machine, then every `serve.sh` step, `chat`, `tunnel`; `--dry-run` |
| `env.sh` | every pin (TensorFold trees, b12x, pack, drafter, head, shas), `DEFAULT_PROFILE`, paths and knobs; `verify_runtime`, `verify_view`, `check_pack`, the hosts-file reader |
| `profiles/*.env` | `fast-160k` (default), `int4-262k` (opt-in), `dcp4-262k` |
| `setup.sh` | per Spark: venv + torch cu130, both pinned TensorFold checkouts, b12x stage, downloads, serve view, checks (`--check`, `--runtime-only`) |
| `serve.sh` | driver: preflight / up / status / smoke / bench / down / logs / tunnel over ssh, `DRY_RUN=1` |
| `rank.sh` | one Spark, one rank: preflight, fadvise, watch, start, state, stop, unwatch, status, smoke, bench |
| `chat.sh` | one streamed chat request and the engine's stats for it |
| `drop-model-cache.sh` | `posix_fadvise(DONTNEED)` of the model files (GB10 counts page cache as used) |
| `hosts.example` | the four Sparks: rank, ssh target, fabric IP (`./glm53 init` writes `hosts`) |
| `tools/discover.sh` | what `init` runs on rank 0: the fabric port and the other Sparks on it (`discover`), and the per-Spark facts (`probe`) |
| `cluster.env` (git-ignored) | per-cluster settings `./glm53 init` / `setup` record (`FABRIC_IFNAME`, `ROCE_HCA`, `MODEL_DIR`, `SSH_CONFIG` after `init --via`); `run.sh` and `serve.sh` forward them to every Spark |
| `lib/tf_serve_rank.py` | the launcher: environment before imports, refusals, the wrappers, then `tensorfold.cli.main(["serve", ...])` |
| `lib/tf_serve_patches.py` | serving wrappers: DFlash2 default, startup + per-request rank consensus, fatal exit on a failed round, warm-up |
| `lib/tf_speed_patches.py`, `lib/tf_speed_common.py` | the speed sweep's wrappers: tile pinning, the RoCE rank-order hard stop, the per-round health guard and consensus |
| `lib/tf_serve_memwatch.py` | per-Spark memory / liveness watchdog; rank 0's also polls the peers and stops all four |
| `tiles/tiles.json` | the pinned tile table |
| `tools/make_view.sh`, `tools/fix_chat_template.py`, `tools/fetch_lm_head.py` | the serve view: pack symlinks + BF16 lm_head + fixed template |
| `tools/b12x_env.sh` | the RoCE environment (stage path, HCA, spin limit, per-host GID) |
| `tools/check_pack.py` | pack completeness without hashing: 58 shards, each exactly as long as its safetensors header says |

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
11. **RoCE setup race.** b12x's setup rendezvous (master port + 11, a fixed 120 s timeout) starts as each rank finishes
    loading, and ranks finish minutes apart, so a follower could give up before rank 0 opened it (`3/4 clients
    joined`) and every rank exited 3. The wrapper holds every rank at a barrier on the NCCL store until all four have
    loaded (the same barrier is in [drowzeys/TensorFold#4](https://github.com/drowzeys/TensorFold/pull/4)).

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

Export the variable before `./glm53` (or `serve.sh`); it forwards every `env.sh` variable you set, plus `cluster.env`,
to all four Sparks and prints the resolved profile on start. Values marked `(measured)` in `env.sh` are the measured configuration: change one only to
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

- **This repo's scripts end to end** on fresh Sparks: `./glm53 init`, `setup` (venv, TensorFold checkouts, b12x stage,
  downloads or the `--download-once` copy, view), `up`, `smoke`, `bench`. Every number above came from the predecessor
  stack; the launcher has run only with ssh and rsync replaced by stand-ins. `init`'s fabric discovery (with and
  without `--via`) has run only against simulated Sparks (stand-ins for `ssh`, `ip`, `ping`, `nvidia-smi`,
  `ibdev2netdev`, `avahi-browse`), not on a real ConnectX-7 fabric, NVIDIA Sync cluster or `discover-sparks` setup.
- **Sampled requests.** Everything was greedy; the speed-up at temperature > 0 is unmeasured.
- **The thinking-on stress sequence** (long prompts with tool calls, streamed channels) has not been run.
- **HTTP overhead, directly.** Streaming does per-token work on rank 0 inside the decode round. The 6-prompt mean
  through `/v1` (41.11, one run per prompt) is 0.75% below the in-process sweep's (41.42, median of 3).
- **DFlash2 at 162.5K**: the run was stopped before the drafted arm.
- **Quality on the pinned tile table** (the bf16 G2 gate was scored on its recorded arms: inconclusive by 0.007
  points on the prompt path), and the decode half of `int4-262k`'s gate. `int4-262k` at prompts longer than 2,842
  tokens.
- `vm.compaction_proactiveness` is 20 on the measured Sparks; PR #159's own recipe uses 0. No stalls were seen.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `no hosts file yet` / `no hosts file at ...` | `./glm53 init` on one of the Sparks (or `cp hosts.example hosts` and fill it in) |
| `init`: `is not a DGX Spark` | run `init` on one of the four Sparks, or `./glm53 init --via <one Spark>` / `--hosts` from where you are |
| `init`: `Permission denied` / `found N of the other three` / `rank 0 -> rank N: FAILED` | the `To fix:` lines: NVIDIA's `discover-sparks` (or `ssh-copy-id` per Spark) once on rank 0, a changed host key (`ssh-keygen -R <ip>`), a Spark off or on another port; then `./glm53 init --force` |
| `init`: `only N of the other three Sparks answered` on a 169.254.x.x fabric | a /16 is not ping-swept: `sudo apt install -y avahi-utils` on rank 0 (mDNS), or `--hosts local,<ip>,<ip>,<ip>` |
| `init`: `5 answered` / `same /etc/machine-id` | name the four: `./glm53 init --hosts local,<ip>,<ip>,<ip> --force` |
| `init`: `no ConnectX-7 port is Up` / `has no IPv4 address` | cable the same port of all four to the switch and give it an address (Cluster Assistant or the switch playbook), or `FABRIC_IFNAME=<port> ./glm53 init --force`; with `--hosts`, `--fabric-ips` too |
| `setup`: `FAIL rank N ...` | the summary prints the fix; the full log is `runs/setup-<time>/setup-rankN.log`; re-run `./glm53 setup` (finished steps are kept) |
| `no Hugging Face token` | request access to the pack, then `HF_TOKEN=hf_... ./glm53 setup` (or `hf auth login` on that Spark) |
| `the pack at ... is incomplete` / `truncated` | `./glm53 setup` (downloads resume; `--download-once` copies resume) |
| `rsync not found` | `sudo apt install -y rsync` there (or `RECIPE_SYNC=git` for the recipe copy) |
| `recipe clone at <sha>, here <sha>` | `./glm53 sync` (or `git pull` on every Spark): the four ranks must run the same `lib/` |
| `TensorFold at ... is <sha>, the recipe pins ...` / `no TensorFold checkout at ...TensorFold-0c858e3402ad` | `./glm53 setup` (installs both trees; `--no-int4` skipped the second) |
| `lacks the measured loader fixes` | a TensorFold tree without the `glm53-tp4-spark` commits; `setup.sh` |
| `no Python.h under ...` | `sudo apt install -y libpython3.12-dev` (or the matching version), then `./glm53 setup` |
| `b12x builds a small RDMA proxy with gcc + libibverbs` | `sudo apt install -y gcc libibverbs-dev`, then `./glm53 setup` (or `TFS_ROCE=0` for NCCL) |
| `b12x tarball sha256 ... !=` | GitHub served a different archive for `b58f34e`; inspect it, then `B12X_TARBALL_SHA=<sha> ./glm53 setup` to accept |
| `REFUSE: NCCL_IB_HCA='=rocep1s0f0'` | start through `./glm53` / `serve.sh` / `rank.sh`, never by hand: `tools/b12x_env.sh` strips the `=` |
| `REFUSE: no RoCE v2 GID for <ip>` | the hosts file's `fabric_ip` is not on `FABRIC_IFNAME`, or the RDMA device is not `ROCE_HCA`: `./glm53 init --force` re-detects both |
| `decode-window reductions: NCCL` then `FATAL ... RoCE requested` | b12x stage or GID problem; `rank.sh preflight --rank R` prints the resolved GID |
| `3/4 clients joined` / `client socket has timed out after 120000ms`, then `FATAL ... RoCE requested` | b12x's setup rendezvous started before the slowest rank had loaded. Fixed by the rendezvous barrier (`align_roce_rendezvous` in `lib/tf_serve_patches.py`; rank logs show `all 4 ranks loaded; b12x RoCE rendezvous ... now`); if you see it, update the recipe (`git pull`, `./glm53 sync`) and start again |
| `context ... x 1 streams needs ... is free` / `LOW MEMORY` | page cache or another job's memory: `./glm53 fadvise`, check `preflight`; never lower `TF_GLM53_CACHE_RESERVE_GB` |
| `GPU BUSY` / `a server rank already runs` / `PORT ... in use` | another job or this server: `./glm53 status`, `./glm53 down`, or stop the other job |
| `REFUSE: no watchdog` / `watchdog flagged` | `./glm53 up` starts the watchdogs; after a trip `./glm53 logs`, read `/tmp/tf_serve_memwatch.VIOLATION`, then `./glm53 down` |
| `START FAILED` / `TIMEOUT` during `up` | `./glm53 logs` (rank logs: `[tf_serve] FATAL ...`), `./glm53 down`, `./glm53 preflight` before the next `up` |
| `profile int4-262k is not validated yet` | intended until its quality gate passes: `--allow-unvalidated` (`ALLOW_UNVALIDATED=1`) to run it anyway |
| first request slow | the warm-up runs before HTTP starts; the first long prompt can still build prompt kernels once |
| client gets `reasoning_content` only, `finish_reason: length` | the reply spent `max_tokens` thinking: raise it or send thinking off |
| decode far below the tables | the reply's `tensorfold` block must show `mtp_mode` `dflash`, `depth` 7, `confidence` 0.6; rank 0's log must show `decode-window reductions: RoCE one-shot`; nothing else on the GPUs |
| `bench` fails on ids with `dcp4-262k` or `int4-262k` | expected against the 32K reference; those profiles set `BENCH_IDS=self` (drafted vs this server's serial), and `./glm53 bench` reuses the profile of the last `up` |
