# AGENTS.md

Instructions for an AI agent setting up this recipe. Read the README for the numbers and the reasons; run only this.

## Run exactly

From the machine that drives the cluster (one of the Sparks is fine), in this clone. The pack is gated: the user must
have requested access on Hugging Face.

```bash
./glm53 init --hosts H0,H1,H2,H3          # the four Sparks' ssh targets, rank 0 first; 'local' = this machine
./glm53 setup --download-once             # all four in parallel; one pack download on rank 0, copied over the fabric
./glm53 up                                # preflight, then the four ranks; ends with "GLM-5.3 is up"
./glm53 smoke
```

- If `init` or any later command prints `To fix:`, run those lines (or show them to the user when they need sudo or a
  password), then repeat the same `./glm53` command. `init` needs `--force` to replace its hosts file.
- If `setup` stops for a token, the user runs `HF_TOKEN=<their token> ./glm53 setup --download-once` (never paste a
  token into a logged command yourself), or `~/glm53-tensorfold/venv/bin/hf auth login` on rank 0. Re-running
  `setup` resumes; nothing finished is redone.
- A pack already on disk: add `--model-dir /path/to/GLM-5.3-EXL3-3.38bpw` to `setup` (absolute path; on every Spark,
  or on rank 0 with `--download-once`).
- `./glm53 check` verifies every install; `./glm53 preflight` checks a start without starting.

API: `http://127.0.0.1:8890/v1` on rank 0, model id `GLM-5.3-EXL3-3.38bpw` (`./glm53 tunnel --open` brings it to the
driver). `./glm53 chat "..."` sends one request. Stop with `./glm53 down`. `--dry-run` on any command prints the ssh
and rsync commands without running them. Logs: `tensorfold-four-spark-tp4/runs/` and `./glm53 logs`.

Profiles: the default (`fast-160k`, 163,840 tokens) is the measured one. `--profile dcp4-262k` is measured and slower.
`--profile int4-262k --allow-unvalidated` only when the user asks for it.

## Do not

- **Do not install TensorFold from PyPI, a release tag or `ashhart/TensorFold` main.** None of them has the
  `glm_moe_dsa` family yet (PR #159 is open), and PR #159 alone does not load this pack (host memory and fp16
  tensors, see the folder README). `setup.sh` checks out the pinned `vcruz305/TensorFold` commits; every launcher
  refuses another tree.
- **Do not repoint the pins at the upstream PR branches** (`glm53-gb10-loading` and the rest). They rename functions
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
- **Do not quote the SixCat decode figure as user-facing speed**, and do not redistribute the DFlash2 weights
  (CC BY-NC-ND 4.0: non-commercial use only).

If something fails, read the `To fix:` block, the README's Troubleshooting table and `./glm53 logs` before changing
any setting.
