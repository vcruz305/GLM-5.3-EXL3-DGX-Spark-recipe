# int4 KV cache at 262,144 tokens (`PROFILE=int4-262k`), 2026-10-05

**Provenance: TODO(confirm).** These files come from one session on four DGX Sparks that served the recipe's TensorFold
tree for this profile, [`vcruz305/TensorFold` `glm53-kv-int4` @ `0c858e3`](https://github.com/vcruz305/TensorFold/tree/glm53-kv-int4)
(`git -c core.abbrev=7 diff 689596d` sha256 `d3866de2…`; the seven changed files are byte-identical to the served
tree), through a **copy** of this folder's `lib/` started by the operator's own scripts, not through `./glm53` /
`serve.sh`. Of that copy, `tf_serve_patches.py`, `tf_speed_patches.py` and `tf_speed_common.py` are byte-identical to
this folder's (line endings aside); its `tf_serve_rank.py` differs only in paths and messages (the same argv and the
same `--kv-dtype` post-load check), and its watchdog is the operator's variant of `tf_serve_memwatch.py`. Settings that change numbers were the profile's: `--context 262144`, `TF_GLM53_DCP=1`,
`--kv-dtype int4` (`TF_GLM53_KV_RCP=approx`, `TF_GLM53_KV_TILE=fp16`), b12x RoCE, pinned tiles `db409731…`,
`TF_GLM53_PROMPT_ROWS=2048`, DFlash2 depth 7 / confidence 0.60, the fixed chat template. The HTTP and rendezvous
ports differed (8891 / 29850). Every figure stays TODO(confirm) until one session reproduces it by following the
README with `./glm53 up --profile int4-262k --allow-unvalidated` and `./glm53 bench`.

Edits as in the other records: LF line endings, operator paths rewritten to `/workspace/...`
(`/workspace/TensorFold-int4` is the int4 checkout, `/workspace/serve_q4` the copied serve directory, `/workspace/view`
the serve view), hosts and fabric addresses replaced by `spark-0` … `spark-3` and `10.0.0.10` … `10.0.0.13` in rank
order.

| File | What |
|---|---|
| [`bench_q4_262k_int4.txt`](bench_q4_262k_int4.txt), [`.json`](bench_q4_262k_int4.json) | the 6 reference prompts (thinking on, greedy, 512 tokens): per prompt the `/tokenize` ids against the sweep's, one `"draft": false` reply and three drafted replies that must equal it token for token; server decode tok/s. Client: `bench_v1.py`'s client and prompts in a wrapper (`bench_q4.py`, not in this repo) |
| [`smoke.json`](smoke.json) | `bench/bench_v1.py --smoke` against this server (21 / 21) |
| [`rank0.log`](rank0.log) … [`rank3.log`](rank3.log) | start-up (RoCE `equals the NCCL rank-order sum: True`, tiles in use, `KV cache int4 latent, 9.234 GiB for 262146 positions (dcp 1)`), then one `done req-...` line per request on rank 0 |
| [`sixcat_speed_all.log`](sixcat_speed_all.log) | SixCat v0.7.0 `speed --policy strict --thinking off --profile all` (decode and balanced; the suite's 600 s budget ran out before prefill, so no JSON) |
| [`sixcat_speed_prefill.json`](sixcat_speed_prefill.json), [`.log`](sixcat_speed_prefill.log) | the same flags with `--profile prefill` |
| [`memwatch_events.jsonl`](memwatch_events.jsonl) | *derived*: each node's watchdog session for this boot: start, minimum MemAvailable over the whole session and from the first request on, swap growth, stop |
| [`parity_spark-0.json`](parity_spark-0.json) … [`parity_spark-3.json`](parity_spark-3.json) | per-host kernel parity before the load: the int4/int8 writer against each host's compiled exllamav3 `quant_cache_cont` (0 byte and 0 scale mismatches with the approximate reciprocal), row invariance of the readers |

## Results

| Measurement | Result |
|---|---:|
| Decode with DFlash2, mean of 6 prompts (median of 3 per prompt) | 40.62 tok/s |
| code / prose / math | 38.94 / 33.07 / 57.98 tok/s |
| chat_explain / chat_multiturn / long_doc (2,842 tokens) | 37.66 / 33.39 / 42.71 tok/s |
| Decode without drafting, mean of 6 prompts | 25.31 tok/s |
| Drafted reply == this server's serial reply | 18 / 18 |
| Serial ids == the bf16 sweep's | no (int4 numerics; first differences at tokens 32 / 35 / 24 / 119 / 45 / 12) |
| SixCat decode, C=1, per-stream p50 | 68.18 tok/s |
| SixCat prefill, ~2,474-token prompt, p50 | 307.9 tok/s |
| SixCat TTFT on that prompt, p50 | 8.04 s |
| KV cache per rank | 9.234 GiB (262,146 positions) |
| Lowest MemAvailable from the first request on (any rank) | 10.18 GiB |
| Lowest MemAvailable during the first boot (kernel builds + graph capture) | 6.35 GiB |
| Swap growth | 0 on all four |
| Load, then serving after warm-up (first boot of a fresh kernel cache) | 451 s, 570 s |

Not measured: prompts longer than 2,842 tokens at this context (TTFT, prefill, decode and memory transients), and
sampled requests. The int4 path's quality gate (teacher-forced against exllamav3 with its Q4 cache) passed on the
prompt path (160 rows, mean NLL +0.126%, 95% CI −0.017 … +0.264%); the decode path is still being finished, so the
profile stays opt-in.
