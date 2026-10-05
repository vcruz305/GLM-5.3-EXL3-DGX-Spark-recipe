#!/usr/bin/env python3
"""One rank of the GLM-5.3 TP=4 TensorFold server (PR #159 `tensorfold serve`), in the measured fastest exact
configuration. Started by rank.sh (which sources env.sh, the profile and tools/b12x_env.sh first).

This is the launcher the measurements ran (the tf_serve stack, 2026-10-05) with three recipe changes: the b12x stage
directory comes from B12X_STAGE instead of a fixed path, the RoCE GID message names no subnet, and an opt-in
TFS_KV_DTYPE (int8 / int4, the glm53-kv-int4 TensorFold branch only) adds `--kv-dtype` plus a post-load check. With
TFS_KV_DTYPE=bf16 (every measured profile) the argv and the engine are exactly the measured ones. Since then every
RoCE start also waits at a barrier before b12x's setup rendezvous (step 3, lib/tf_serve_patches.py
align_roce_rendezvous); it changes when the RoCE setup starts, not what any rank computes.

usage: tf_serve_rank.py --rank R [--dry-run]

What it does, in order (nothing in the clone is edited; every patch is applied here at import):
 1. environment, before torch / tensorfold are imported (several knobs are module constants): TF_NCCL_LIB,
    CUDA_VISIBLE_DEVICES=0, TF_GLM53_{ROCE,DFLASH,PROMPT_ROWS,VERIFY_ROWS,DFLASH_DEPTH,DFLASH_CONFIDENCE,PROFILE_FLAG},
    NCCL/GLOO socket interfaces - exactly the sweep's (tf_speed_rank.py) values;
 2. refusals (exit 3): RoCE env not as measured (NCCL_IB_HCA with '=', no RoCE v2 GID, spin limit), tile table or
    chat template bytes not the pinned ones, a stale DFLASH_CFG / PROFILE in the control dir is removed;
 3. the sweep's wrappers (lib/tf_speed_patches.py, verbatim): pin_tiles(load:...), patch_roce_check (hard stop);
    with RoCE also align_roce_rendezvous (every rank loaded before b12x's 120 s setup rendezvous starts);
 4. lib/tf_serve_patches.install: post-load checks, health guard + consensus (Ops), DFlash2 default, warm-up;
 5. tensorfold.cli.main(["serve", VIEW, "--backend", "cuda", "--tp", "4", "--rank", R, ...]): rank 0 serves the
    OpenAI-compatible /v1 API, ranks 1-3 follow it.
--dry-run prints the environment and the tensorfold argv after the same checks, imports nothing heavy, and exits 0.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import posixpath
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

ENV_PREFIXES = ("TF_", "TFS_", "NCCL_", "B12X_", "GLOO_", "CUDA_VISIBLE", "TENSORFOLD_", "PYTHON")
LOOPBACK = ("127.0.0.1", "localhost", "::1")


def log(msg: str) -> None:
    print(msg, flush=True)


def bail(code: int, why: str) -> None:
    log(f"[tf_serve] REFUSE: {why}")
    log(f"[tf_serve] EXIT {code}")
    sys.stdout.flush()
    os._exit(code)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def get(env: dict, key: str) -> str:
    v = env.get(key)
    if v is None or v == "":
        raise KeyError(f"{key} is not set (rank.sh sources env.sh)")
    return v


def serve_env(env: dict) -> dict:
    """The variables this rank sets before importing tensorfold, from the TFS_* settings (env.sh) and the b12x
    env.sh already in ``env``. Pure: returns a new dict of the values to set."""
    out = {
        "TF_NCCL_LIB": get(env, "TFS_NCCL_LIB"),
        "CUDA_VISIBLE_DEVICES": "0",
        "TENSORFOLD_NO_UPDATE_CHECK": "1",
        "PYTHONUNBUFFERED": "1",
        "PYTHONFAULTHANDLER": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "TF_GLM53_DFLASH": get(env, "TFS_DFLASH"),
        "TF_GLM53_PROMPT_ROWS": get(env, "TFS_PROMPT_ROWS"),
        "TF_GLM53_VERIFY_ROWS": get(env, "TFS_VERIFY_ROWS"),
        "TF_GLM53_DFLASH_DEPTH": get(env, "TFS_DFLASH_DEPTH"),
        "TF_GLM53_DFLASH_CONFIDENCE": get(env, "TFS_DFLASH_CONFIDENCE"),
        "TF_GLM53_PROFILE_FLAG": posixpath.join(get(env, "TFS_CTL"), "PROFILE"),   # DFLASH_CFG lives next to it
        "GLOO_SOCKET_IFNAME": get(env, "TFS_IFNAME"),
    }
    if env.get("TFS_ROCE", "1") == "1":
        out["TF_GLM53_ROCE"] = "1"
        out["NCCL_SOCKET_IFNAME"] = env.get("NCCL_SOCKET_IFNAME") or "=" + get(env, "TFS_IFNAME")
    else:                                            # the sweep's NCCL arm: the 2026-10-05 baseline launcher's values
        out["TF_GLM53_ROCE"] = "0"
        out["NCCL_SOCKET_IFNAME"] = "=" + get(env, "TFS_IFNAME")
        out["NCCL_IB_HCA"] = "=" + get(env, "TFS_HCA")
    return out


def roce_problems(env: dict) -> list[str]:
    """The sweep's RoCE refusals (tf_speed_rank.py / tf_speed_start_rank.sh) on the environment env.sh produced."""
    if env.get("TFS_ROCE", "1") != "1":
        return []
    bad = []
    if env.get("NCCL_IB_HCA", "") != env.get("TFS_HCA", "rocep1s0f0"):
        bad.append(f"NCCL_IB_HCA={env.get('NCCL_IB_HCA', '')!r}: needs {env.get('TFS_HCA', 'rocep1s0f0')!r} "
                   "(b12x compares it with strcmp; '=' disables RoCE) - source b12x env.sh")
    if not env.get("B12X_ROCE_GID_INDEX"):
        bad.append("B12X_ROCE_GID_INDEX unresolved (tools/b12x_env.sh found no RoCE v2 GID for this rank's fabric "
                   "address in the hosts file)")
    if env.get("B12X_ROCE_SPIN_LIMIT") != "300000000":
        bad.append(f"B12X_ROCE_SPIN_LIMIT={env.get('B12X_ROCE_SPIN_LIMIT')!r}: must be 300000000 on every rank")
    stage = env.get("B12X_STAGE", "")
    if not stage or stage.rstrip("/") + "/site" not in env.get("PYTHONPATH", ""):
        bad.append("PYTHONPATH lacks the b12x stage $B12X_STAGE/site (source tools/b12x_env.sh; setup.sh stages it)")
    return bad


def tensorfold_argv(env: dict, rank: int) -> list[str]:
    """The `tensorfold serve` command line every rank runs (only --rank differs)."""
    host = get(env, "TFS_HTTP_HOST")
    argv = ["serve", get(env, "TFS_VIEW"), "--backend", "cuda", "--tp", "4", "--rank", str(int(rank)),
            "--master", get(env, "TFS_MASTER"), "--master-port", str(int(get(env, "TFS_MASTER_PORT"))),
            "--context", str(int(get(env, "TFS_CONTEXT"))),
            "--host", host, "--port", str(int(get(env, "TFS_HTTP_PORT"))),
            "--name", get(env, "TFS_NAME"), "--max-tokens", str(int(get(env, "TFS_MAX_TOKENS"))),
            "--thinking" if env.get("TFS_THINKING", "1") == "1" else "--no-thinking",
            "--no-update-check"]
    if env.get("TFS_ALIAS"):
        argv += ["--alias", env["TFS_ALIAS"]]
    if env.get("TFS_API_KEY_FILE"):
        argv += ["--api-key-file", env["TFS_API_KEY_FILE"]]
    kv = env.get("TFS_KV_DTYPE", "bf16")                 # bf16 (every measured profile) adds nothing to the argv
    if kv != "bf16":
        argv += ["--kv-dtype", kv]
    return argv


def config_problems(env: dict, rank: int, check_files: bool = True) -> list[str]:
    bad = []
    if rank not in (0, 1, 2, 3):
        bad.append(f"rank {rank}: 0..3")
    if env.get("TFS_HTTP_HOST", "127.0.0.1") not in LOOPBACK and not env.get("TFS_API_KEY_FILE"):
        bad.append(f"TFS_HTTP_HOST={env.get('TFS_HTTP_HOST')} is not loopback: set TFS_API_KEY_FILE (tensorfold "
                   "--api-key-file), or keep 127.0.0.1 and use an ssh tunnel")
    try:
        ctx = int(get(env, "TFS_CONTEXT"))
        dcp = env.get("TF_GLM53_DCP", "")
        if ctx > 200_000 and dcp not in ("1", "4"):
            bad.append(f"TFS_CONTEXT={ctx}: past 200K the engine turns on decode context parallelism unless "
                       f"TF_GLM53_DCP says otherwise (got {dcp!r}): set TF_GLM53_DCP=1 (whole KV cache on every rank, "
                       "the 32K code path) or 4 (DCP: other bits, label it)")
        elif dcp not in ("", "1", "4"):
            bad.append(f"TF_GLM53_DCP={dcp!r}: 1 or 4")
    except (KeyError, ValueError) as exc:
        bad.append(str(exc))
    try:
        conf = float(get(env, "TFS_DFLASH_CONFIDENCE"))
        depth = int(get(env, "TFS_DFLASH_DEPTH"))
        if not (0.0 <= conf <= 1.0 and 0 <= depth <= 7):
            bad.append(f"DFlash2 policy depth {depth} / confidence {conf} out of range")
    except (KeyError, ValueError) as exc:
        bad.append(str(exc))
    kv = env.get("TFS_KV_DTYPE", "bf16")
    if kv not in ("bf16", "int8", "int4"):
        bad.append(f"TFS_KV_DTYPE={kv!r}: bf16, int8 or int4")
    if not check_files:
        return bad
    if kv != "bf16" and not os.path.isfile(os.path.join(env.get("TFS_CLONE", ""), "tensorfold", "families",
                                                         "glm_moe_dsa", "cuda", "kvq.py")):
        bad.append(f"TFS_KV_DTYPE={kv} but TFS_CLONE={env.get('TFS_CLONE')!r} has no glm_moe_dsa/cuda/kvq.py "
                   "(the quantized latent cache is on the vcruz305/TensorFold glm53-kv-int4 branch: PROFILE=int4-262k)")
    for key in ("TFS_CLONE", "TFS_VIEW", "TFS_DFLASH", "TFS_NCCL_LIB", "TFS_TILES"):
        p = env.get(key, "")
        if not p or not os.path.exists(p):
            bad.append(f"{key}={p!r} does not exist")
    view = env.get("TFS_VIEW", "")
    tpl = os.path.join(view, "chat_template.jinja")
    if os.path.isfile(tpl):
        got = sha256_file(tpl)
        if got != env.get("TFS_TEMPLATE_SHA"):
            bad.append(f"{tpl} sha {got[:16]} != the fixed template {env.get('TFS_TEMPLATE_SHA', '')[:16]} "
                       "(tools/make_view.sh)")
    elif view:
        bad.append(f"{tpl} missing (tools/make_view.sh)")
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors.index.json"):
        if view and not os.path.exists(os.path.join(view, name)):
            bad.append(f"{view}/{name} missing")
    tiles = env.get("TFS_TILES", "")
    if os.path.isfile(tiles):
        got = sha256_file(tiles)
        if got != env.get("TFS_TILES_SHA"):
            bad.append(f"tile table {tiles} sha {got[:16]} != pinned {env.get('TFS_TILES_SHA', '')[:16]}")
    dpath = env.get("TFS_DFLASH", "")
    if dpath and not os.path.isfile(os.path.join(dpath, "config.json")):
        bad.append(f"{dpath}/config.json missing (DFlash2 drafter)")
    return bad


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rank", type=int, required=True, choices=(0, 1, 2, 3))
    ap.add_argument("--dry-run", action="store_true", help="print env + tensorfold argv after the checks; no load")
    ap.add_argument("--no-file-checks", action="store_true", help="with --dry-run: skip path/sha checks (CPU tests)")
    a = ap.parse_args(argv)
    env = dict(os.environ)
    try:
        new = serve_env(env)
        targv = tensorfold_argv(env, a.rank)
    except KeyError as exc:
        bail(3, str(exc))
    env.update(new)
    bad = config_problems(env, a.rank, check_files=not a.no_file_checks) + roce_problems(env)
    shown = {k: env[k] for k in sorted(env) if k.startswith(ENV_PREFIXES)}
    if a.dry_run:
        log(f"[tf_serve] DRY RUN rank {a.rank}")
        for k, v in shown.items():
            log(f"  {k}={v}")
        log("  argv: tensorfold " + " ".join(targv))
        for b in bad:
            log(f"  WOULD REFUSE: {b}")
        log(f"[tf_serve] dry run {'ok' if not bad else 'found ' + str(len(bad)) + ' problem(s)'}")
        return 0 if not bad else 3
    if bad:
        bail(3, "; ".join(bad))
    os.environ.update(new)
    ctl = env["TFS_CTL"]
    os.makedirs(ctl, exist_ok=True)
    for fn in ("PROFILE", "PROFILE_PREFILL", "DFLASH_CFG"):     # a stale file would change one rank's policy / trace
        try:
            os.remove(os.path.join(ctl, fn))
        except FileNotFoundError:
            pass
    log(f"[tf_serve] rank {a.rank} env: " + " ".join(f"{k}={v}" for k, v in shown.items()
                                                    if not k.startswith("PYTHONPATH")))
    log(f"[tf_serve] rank {a.rank} argv: tensorfold " + " ".join(targv))
    sys.path.insert(1, env["TFS_CLONE"])

    from tensorfold.families import glm_moe_dsa as family          # noqa: E402  (after the environment)
    from tensorfold.families.glm_moe_dsa.cuda import fused         # noqa: E402

    import tf_speed_patches as Pt                                   # noqa: E402  the sweep's wrappers, verbatim
    import tf_serve_patches as Sp                                   # noqa: E402

    policy = Sp.Policy.from_env()
    Pt.pin_tiles(fused, "load:" + env["TFS_TILES"], log)
    if policy.roce:
        Pt.patch_roce_check(log)
        Sp.align_roce_rendezvous(log)                               # all ranks loaded before b12x's 120 s gloo join
    Sp.install(family, a.rank, policy, ctl, log)
    kv = env.get("TFS_KV_DTYPE", "bf16")
    if kv != "bf16":                                                # opt-in, unvalidated: prove the cache it built
        inner = family.cuda_engine

        def cuda_engine(model_dir, **options):
            if options.get("kv_dtype", "bf16") != kv:
                bail(3, f"tensorfold passed kv_dtype {options.get('kv_dtype', 'bf16')!r}, TFS_KV_DTYPE is {kv!r}")
            eng = inner(model_dir, **options)
            st = eng.runner.st
            got = getattr(st, "kv_dtype", "bf16")
            if got != kv:
                bail(3, f"engine caches are {got}, TFS_KV_DTYPE is {kv}")
            log(f"[tf_serve] rank {a.rank}: KV cache {got} latent, {st.nbytes() / 2**30:.3f} GiB for {st.local} "
                f"positions (dcp {eng.runner.w.dcp})")
            return eng

        family.cuda_engine = cuda_engine
    from tensorfold import cli                                      # noqa: E402

    rc = cli.main(targv)
    log(f"[tf_serve] rank {a.rank}: tensorfold returned {rc}")
    log(f"[tf_serve] EXIT {rc}")
    sys.stdout.flush()
    os._exit(int(rc or 0))


if __name__ == "__main__":
    raise SystemExit(main())
