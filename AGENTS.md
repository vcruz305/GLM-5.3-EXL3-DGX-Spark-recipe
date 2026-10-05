# AGENTS.md

Instructions for an AI agent setting up this recipe. Read the README for the numbers and the reasons; run only this.

## Run exactly

On each of the four Sparks, in a clone at the same path:

```bash
bash tensorfold-four-spark-tp4/setup.sh
```

The pack is gated. If `setup.sh` stops for a token, run `~/glm53-tensorfold/venv/bin/hf auth login`, then run
`setup.sh` again (downloads resume). `bash tensorfold-four-spark-tp4/setup.sh --check` verifies the runtime.

Then on the driver machine (rank 0 is fine):

```bash
cp tensorfold-four-spark-tp4/hosts.example tensorfold-four-spark-tp4/hosts   # edit: rank, ssh target, fabric IP
bash tensorfold-four-spark-tp4/serve.sh preflight    # must print "preflight OK on all four"
bash tensorfold-four-spark-tp4/serve.sh up           # ~7-8 min, ends with READY
bash tensorfold-four-spark-tp4/serve.sh smoke
```

API: `http://127.0.0.1:8890/v1` on rank 0, model id `GLM-5.3-EXL3-3.38bpw`. Stop with `serve.sh down`.
`DRY_RUN=1` on any `serve.sh` step prints the commands without running them.

## Do not

- Install TensorFold from PyPI, a release or `ashhart/TensorFold` main, or edit the checkout under
  `~/glm53-tensorfold/`. The recipe pins a `vcruz305/TensorFold` commit and refuses any other tree.
- `pip install b12x` (PyPI has no `comm.roce`). `setup.sh` stages the pinned commit.
- Serve the pack directory. Serve the view `setup.sh` builds (BF16 lm_head and fixed chat template).
- Run bf16 at 262,144 tokens with `TF_GLM53_DCP=1`. It swaps. Use the default profile or `PROFILE=dcp4-262k`.
- Lower `TF_GLM53_CACHE_RESERVE_GB`, disable the watchdog or allow swap growth to make a context fit.
- Pass `--parallel`. The server answers one request at a time.
- Use `PROFILE=int4-262k` unless asked to validate it (pending, needs `ALLOW_UNVALIDATED=1`).
- Start while another job holds a GPU on any of the four Sparks.
- Quote the SixCat decode figure as user-facing speed.
- Redistribute the DFlash2 weights (CC BY-NC-ND 4.0, non-commercial use only).

If something fails, read the README's Troubleshooting table and `serve.sh logs` before changing any setting.
