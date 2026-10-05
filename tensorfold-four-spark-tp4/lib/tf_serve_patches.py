"""Serving-only patches for TensorFold PR #159 GLM-5.3 TP=4 behind `tensorfold serve` (applied at import by
tf_serve_rank.py; the clone is never edited).

The sweep's own wrappers are reused verbatim from tf_speed_patches.py (sha aa3f882c...): tile pinning (`pin_tiles`),
the RoCE "equals the NCCL rank-order sum: False" hard stop (`patch_roce_check`), and `Ops` - the per-round all-rank
b12x poisoned guard around `runner._verify` and the after-request all-gather of [depth, confidence, tokens, reply
hash, poisoned]. This module adds only what `tensorfold serve` needs to run that measured configuration:

* DFlash2 by default. The engine's default mode is MTP (fused.MTP_MODE, "normed/normed") and this pack has no MTP
  layer (num_nextn_predict_layers 0), so a request without "tf_mtp": "dflash" would decode serially (~27 tok/s).
  TF_GLM53_MTP=dflash cannot be used instead: Runner.__init__ calls set_mode() before the drafter is attached and
  raises. Requests now draft with DFlash2 unless they send "draft": false (serial reference); other tf_mtp values
  are refused before any collective.
* Startup consensus: every rank's [DFlash2 depth, confidence, verify rows, RoCE on, tile-table hash] must equal rank
  0's and the configured policy, else every rank exits 3.
* Per-request consensus: Ops' all-gather after every request (rank 0 in generate, ranks 1-3 in the follow loop); any
  disagreement exits every rank with 4 - a rank that decoded a different reply must never serve again.
* Fatal rounds: an exception inside a request's collectives (e.g. the health guard's B12X_UNHEALTHY) exits every
  rank with 5. Without this rank 0's HTTP server answers 500 and hangs the next request against dead followers.
* Warm-up: rank 0 runs the sweep's warm-up (a 16-token serial and a 16-token drafted reply to "Hello") before the
  HTTP server starts, so the first client request does not pay first-use kernel builds.

Pure helpers (resolve_mode, check_rows, check_startup, Policy) import nothing heavy and are unit-tested on a CPU.
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
import sys
from dataclasses import dataclass

MICRO = 1_000_000
DFLASH = "dflash"
EXIT_REFUSED, EXIT_MISMATCH, EXIT_ROUND = 3, 4, 5


# ------------------------------------------------------------------------------------------------- policy ---
@dataclass(frozen=True)
class Policy:
    depth: int = 7
    confidence: float = 0.60
    verify_rows: int = 8
    roce: bool = True
    draft_default: bool = True
    tiles: str = ""
    tiles_sha: str = ""
    health: str = "round"
    warmup: bool = True

    @property
    def conf_micro(self) -> int:
        return round(self.confidence * MICRO)

    @property
    def depth_cap(self) -> int:
        """runner._generate_dflash: min(drafter.block - 1 = 7, vrows - 1, depth)."""
        return min(7, self.verify_rows - 1, self.depth)

    @classmethod
    def from_env(cls, env=None) -> "Policy":
        e = os.environ if env is None else env
        return cls(depth=int(e.get("TF_GLM53_DFLASH_DEPTH", "7")),
                   confidence=float(e.get("TF_GLM53_DFLASH_CONFIDENCE", "0.3")),
                   verify_rows=int(e.get("TF_GLM53_VERIFY_ROWS", "8")),
                   roce=e.get("TF_GLM53_ROCE", "1") != "0",
                   draft_default=e.get("TFS_DRAFT_DEFAULT", "1") == "1",
                   tiles=e.get("TFS_TILES", ""), tiles_sha=e.get("TFS_TILES_SHA", ""),
                   health=e.get("TFS_HEALTH", "round"), warmup=e.get("TFS_WARMUP", "1") == "1")


def resolve_mode(draft: bool, mtp_mode, draft_default: bool):
    """The engine mtp_mode for a request: None = serial (k = 0), "dflash" = DFlash2 rounds. "draft": false always wins
    (the serial reference); no tf_mtp = the server's default; any other tf_mtp is refused (ValueError) before the
    engine shares the request with ranks 1-3."""
    if not draft:
        return None
    if mtp_mode is None:
        return DFLASH if draft_default else None
    if mtp_mode == DFLASH:
        return DFLASH
    raise ValueError(f"tf_mtp {mtp_mode!r}: this server drafts with DFlash2 only (\"tf_mtp\": \"dflash\", the default); "
                     "send \"draft\": false for the serial reference")


def check_rows(rows: list[list[int]], policy: Policy) -> list[str]:
    """After a request, every rank's [depth, conf_micro, n, sha0, sha1, poisoned] (Ops.consensus): all equal to rank 0's,
    nobody poisoned, and a drafted request ran the configured policy."""
    bad = []
    if not rows:
        return ["no consensus rows"]
    ref = [int(x) for x in rows[0][:5]]
    for r, v in enumerate(rows):
        v = [int(x) for x in v[:6]]
        if len(v) < 6:
            bad.append(f"rank {r}: short consensus row {v}")
            continue
        if v[5]:
            bad.append(f"rank {r}: RoCE runtime poisoned")
        if v[:5] != ref:
            bad.append(f"rank {r}: [depth, conf, n, sha0, sha1] {v[:5]} != rank 0's {ref}")
    if ref[0] and (ref[0], ref[1]) != (policy.depth_cap, policy.conf_micro):
        bad.append(f"drafted with depth {ref[0]} / confidence {ref[1] / MICRO}, configured "
                   f"{policy.depth_cap} / {policy.conf_micro / MICRO}")
    return bad


def tiles_hash_ints(sha_hex: str) -> list[int]:
    h = int(sha_hex[:15], 16) if sha_hex else 0
    return [h & 0x7FFFFFFF, (h >> 31) & 0x7FFFFFFF]


def startup_vector(depth: int, conf: float, vrows: int, fast: bool, tiles_sha_hex: str, draft_default: bool) -> list[int]:
    return [int(depth), round(float(conf) * MICRO), int(vrows), 1 if fast else 0, *tiles_hash_ints(tiles_sha_hex),
            1 if draft_default else 0]


def check_startup(rows: list[list[int]], policy: Policy) -> list[str]:
    bad = []
    want = startup_vector(policy.depth_cap, policy.confidence, policy.verify_rows, policy.roce, policy.tiles_sha,
                          policy.draft_default)
    names = ["dflash depth", "dflash confidence (micro)", "verify rows", "RoCE reductions", "tiles sha[0]",
             "tiles sha[1]", "draft default"]
    for r, v in enumerate(rows):
        v = [int(x) for x in v]
        for i, (got, exp) in enumerate(zip(v, want)):
            if got != exp:
                bad.append(f"rank {r}: {names[i]} = {got}, configured {exp}")
    return bad


def tiles_in_use_sha(rows: list[list]) -> str:
    """sha256 of the engine's tile table serialized exactly as tf_speed_patches.save_tiles writes the pinned file."""
    blob = json.dumps({"count": len(rows), "linears": rows}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


# ----------------------------------------------------------------------------------------------- runtime ---
def _say(msg: str) -> None:
    print(msg, flush=True)


def fatal(rank: int, code: int, why: str) -> None:
    _say(f"[tf_serve] FATAL rank {rank}: {why}")
    _say(f"[tf_serve] EXIT {code}")
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


def _gather_ints(engine, values: list[int]) -> list[list[int]]:
    import torch

    world = engine.runner.w.world
    mine = torch.tensor([int(v) for v in values], dtype=torch.int32, device="cuda")
    allv = torch.empty((world * len(values),), dtype=torch.int32, device="cuda")
    engine.comm.all_gather(mine, allv)
    return allv.view(world, len(values)).tolist()


def _consensus_rows(ops) -> list[list[int]]:
    ops.consensus()                                  # every rank: all-gather into ops._all (rank 0 also gets a copy)
    return ops._all.view(ops.world, 6).tolist()


def _warm_prompt(model_dir) -> list[int]:
    """The sweep's warm-up prompt: "Hello", thinking on, effort low (13 tokens), rendered by the served template."""
    from pathlib import Path

    from tokenizers import Tokenizer

    from tensorfold.cuda.chat_template import ChatTemplate

    md = Path(model_dir)
    text = ChatTemplate(md).render([{"role": "user", "content": "Hello"}], tools=None, enable_thinking=True,
                                   extra={"reasoning_effort": "low"})
    return Tokenizer.from_file(str(md / "tokenizer.json")).encode(text, add_special_tokens=False).ids


def post_load(engine, rank: int, policy: Policy, ctl: str, model_dir, log=_say) -> None:
    """Every rank, right after Glm53Engine is built (inside cuda_engine, before the CLI serves or follows)."""
    import tf_speed_patches as Pt

    r = engine.runner
    if r is None:
        fatal(rank, EXIT_REFUSED, "no fused runner (TF_GLM53_FUSED=0?)")
    fw = r.w
    if policy.roce and fw.fast is None:
        fatal(rank, EXIT_REFUSED, "RoCE requested but decode reductions run on NCCL (see the RoCE lines above)")
    if not policy.roce and fw.fast is not None:
        fatal(rank, EXIT_REFUSED, "TF_GLM53_ROCE=0 but RoCE came up")
    if r.drafter is None:
        fatal(rank, EXIT_REFUSED, "no DFlash2 drafter loaded (TF_GLM53_DFLASH)")
    in_use = tiles_in_use_sha(Pt.tile_rows(fw.tunable))
    log(f"[tf_serve] rank {rank}: tiles in use sha {in_use[:16]} (pinned {policy.tiles_sha[:16]})")
    if policy.tiles_sha and in_use != policy.tiles_sha:
        fatal(rank, EXIT_REFUSED, f"tile table in use {in_use} != pinned {policy.tiles_sha}")
    ops = Pt.Ops(engine, rank, ctl, health=policy.health, log=log)
    depth, conf = r._dflash_cfg()                    # what a request would read now: env + any DFLASH_CFG file
    vec = startup_vector(depth, conf, r.vrows, fw.fast is not None, in_use, policy.draft_default)
    rows = _gather_ints(engine, vec)
    bad = check_startup(rows, policy)
    if bad:
        fatal(rank, EXIT_REFUSED, "startup consensus: " + "; ".join(bad))
    log(f"[tf_serve] rank {rank}: startup consensus ok - DFlash2 depth {depth} confidence {conf}, verify rows "
        f"{r.vrows}, drafter block {r.drafter.block}, health guard {ops.health}, decode reductions "
        f"{'RoCE one-shot' if fw.fast else 'NCCL'}, default {'DFlash2' if policy.draft_default else 'serial'}")

    orig_run = engine._run

    @functools.wraps(orig_run)
    def run_or_die(*a, **k):                          # inside a request's collectives: any failure stops every rank
        try:
            return orig_run(*a, **k)
        except BaseException as exc:                  # noqa: BLE001
            import traceback

            traceback.print_exc()
            fatal(rank, EXIT_ROUND, f"request failed inside its collectives: {exc!r}")

    engine._run = run_or_die

    def after_request(tag: str) -> None:
        rows = _consensus_rows(ops)
        problems = check_rows(rows, policy)
        if problems:
            fatal(rank, EXIT_MISMATCH, f"consensus after {tag}: " + "; ".join(problems))

    if rank == 0:
        orig_gen = engine.generate

        @functools.wraps(orig_gen)                     # keeps the signature App inspects (draft, stop_eos, mtp_mode)
        def generate(prompt, max_tokens, sampling, on_tokens, draft=True, stop_eos=True, mtp_mode=None, **kw):
            mode = resolve_mode(draft, mtp_mode, policy.draft_default)      # ValueError: before any collective
            st = orig_gen(prompt, max_tokens, sampling, on_tokens, draft=draft, stop_eos=stop_eos, mtp_mode=mode,
                          **kw)
            after_request("a request")
            if isinstance(st, dict):
                st["tf_serve"] = {"mode": mode or "serial", "consensus": "ok"}
            return st

        engine.generate = generate
        if policy.warmup:
            ids = _warm_prompt(model_dir)
            for draft in (False, True):
                st = engine.generate(ids, 16, None, lambda new: None, draft=draft)
                log(f"[tf_serve] warm-up {'DFlash2' if draft else 'serial'}: {st.get('tokens')} tokens, "
                    f"tok/s {st.get('tok_s')}, sha {st.get('sha256')}")
    else:
        orig_follow = engine.follow

        @functools.wraps(orig_follow)
        def follow(requests=None):
            done = 0
            try:
                while requests is None or done < requests:
                    orig_follow(requests=1)
                    after_request(f"request {done}")
                    done += 1
            except BaseException as exc:              # noqa: BLE001  NCCL failure, KeyboardInterrupt, ...
                import traceback

                traceback.print_exc()
                fatal(rank, EXIT_ROUND, f"follow loop ended: {exc!r}")

        engine.follow = follow
    log(f"[tf_serve] rank {rank}: READY ({'serving HTTP next' if rank == 0 else 'following rank 0'})")


def install(family_module, rank: int, policy: Policy, ctl: str, log=_say) -> None:
    """Wrap the family's cuda_engine (tensorfold.cli calls family.package.cuda_engine) with post_load."""
    orig = family_module.cuda_engine

    @functools.wraps(orig)
    def cuda_engine(model_dir, **options):
        engine = orig(model_dir, **options)
        post_load(engine, rank, policy, ctl, model_dir, log)
        return engine

    family_module.cuda_engine = cuda_engine
