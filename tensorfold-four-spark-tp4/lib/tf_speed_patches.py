"""GPU-side pieces of the speed sweep, applied from the launcher only (the TensorFold clone is never edited):

* tile pinning: save every tuned EXL3 linear's (K splits, warps) after a boot, or replace ``fused.tune_linears`` /
  ``fused.tune_groups`` with a loader of that table (fail closed on any shape mismatch);
* RoCE: ``RoceReduce._check`` copied verbatim except that ``equals the NCCL rank-order sum: False`` now raises (the
  engine then falls back to NCCL on every rank, and the launcher refuses to run a RoCE arm on NCCL);
* ``Ops``: what the protocol asks of every rank - DFlash2 config files, the after-request consensus all-gather, the
  per-round health guard (all ranks agree no b12x runtime is poisoned before a round's tokens are emitted), fixed-row
  verify timings, drafter timings, an eager expert-overlap probe, and the rank-0 chain logger.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import types

import numpy as np
import torch

import tf_speed_common as C


# ------------------------------------------------------------------------------------------------ tiles ---
def tile_rows(lins) -> list[list]:
    return [[int(l.k), int(l.n), float(l.bits), str(l.codebook), str(l.layout), int(l.split[0]), int(l.split[1])]
            for l in lins]


def save_tiles(fw, path: str) -> str:
    """Every rank (after share_tiles they all hold rank 0's picks): the table, written atomically; returns its sha."""
    data = {"count": len(fw.tunable), "linears": tile_rows(fw.tunable)}
    blob = json.dumps(data, sort_keys=True)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + f".tmp{os.getpid()}"
    with open(tmp, "w") as f:
        f.write(blob)
    os.replace(tmp, path)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def apply_tiles(lins, saved: dict) -> int:
    rows = saved["linears"]
    if len(rows) != len(lins):
        raise RuntimeError(f"tile table has {len(rows)} linears, this engine {len(lins)}: refusing")
    changed = 0
    for i, (lin, row) in enumerate(zip(lins, rows)):
        k, n, bits, cb, layout, sk, wk = row
        if (int(lin.k), int(lin.n), float(lin.bits), str(lin.codebook), str(lin.layout)) != (k, n, bits, cb, layout):
            raise RuntimeError(f"tile table linear {i}: shape {row[:5]} != engine {[lin.k, lin.n, lin.bits]}")
        if tuple(lin.split) != (sk, wk):
            changed += 1
        lin.split = (int(sk), int(wk))
    return changed


def pin_tiles(fused, spec: str, log=print) -> None:
    """spec 'load:PATH': fused.Weights applies the saved table instead of timing tiles (tune_groups becomes a no-op:
    the table already holds the group picks). 'save:PATH' / 'tune' / 'notune': nothing to patch here."""
    if not spec.startswith("load:"):
        return
    path = spec[5:]
    with open(path) as f:
        saved = json.load(f)
    sha = hashlib.sha256(json.dumps(saved, sort_keys=True).encode()).hexdigest()[:16]

    def tune_linears(lins, rows=3, iters=20):
        n = apply_tiles(lins, saved)
        log(f"[tf_speed] tiles pinned from {path} (sha {sha}): {len(lins)} linears, {n} differ from plan()")
        return {}

    def tune_groups(gs, rows=3):
        return {}

    fused.tune_linears, fused.tune_groups = tune_linears, tune_groups


# ------------------------------------------------------------------------------------------------- RoCE ---
def patch_roce_check(log=print) -> None:
    from tensorfold.families.glm_moe_dsa.cuda import roce

    def _check(self, nccl) -> None:                      # roce.RoceReduce._check + a hard stop on order False
        from tensorfold.families.glm_moe_dsa.cuda.fused import DECODE_ROWS

        same = order = True
        for rows in (3, DECODE_ROWS):
            g = torch.Generator(device="cpu").manual_seed(1000 + self.rank + rows)
            x = (torch.randn(rows, 6144, generator=g) * 10.0 ** (self.rank - 1)).cuda()
            s = torch.empty_like(x)
            self.all_reduce(x, s)
            allv = torch.empty((self.world, rows, 6144), device="cuda")
            nccl.all_gather(s, allv)
            parts = torch.empty((self.world, rows, 6144), device="cuda")
            nccl.all_gather(x, parts)
            ref = parts[0].clone()
            for r in range(1, self.world):
                ref += parts[r]
            same &= all(torch.equal(allv[r], allv[0]) for r in range(self.world))
            order &= torch.equal(allv[0], ref)
        if not same:
            raise RuntimeError("RoCE all-reduce: ranks hold different bits")
        print(f"[tensorfold] RoCE one-shot reduce ready (ranks bit-equal: {same}; equals the NCCL rank-order sum: "
              f"{order}; windows of 3 and {DECODE_ROWS} rows)", flush=True)
        if not order:
            log("[tf_speed] FATAL ROCE_ORDER_FALSE: the RoCE sum is not the NCCL rank-order sum")
            raise RuntimeError("tf_speed: RoCE sum != NCCL rank-order sum (hard stop)")

    roce.RoceReduce._check = _check


# -------------------------------------------------------------------------------------------- chain log ---
class ChainLog:
    """Rank 0: every greedy Drafter.chain call's full chain (confidence 0, i.e. every candidate row the block pass
    produced) with its cumulative confidences, next to what the engine kept. Drafts are unchanged (the original chain
    runs first and its result is returned)."""

    def __init__(self) -> None:
        self.active, self.rounds, self.bad = False, [], 0

    def install(self, dr) -> None:
        from tensorfold.families.glm5_next.cuda import dflash2 as d2

        orig = dr.chain                                  # bound method of the drafter instance
        log = self

        def chain(self_, tokens, values, proj, anchor, first, sampling, confidence=0.0):
            out = orig(tokens, values, proj, anchor, first, sampling, confidence)
            if log.active and (sampling is None or getattr(sampling, "temperature", 0) <= 0):
                full, cum, prev, c = [], [], int(anchor), 1.0
                for d in range(tokens.shape[0]):
                    edge = dr.succ[tokens[d]].astype(np.float64) @ (dr.pred[prev].astype(np.float64) * proj[d])
                    score = values[d] + d2.EDGE * edge
                    j = int(np.argmax(score))
                    p = np.exp(score - score.max())
                    c *= float(p[j] / p.sum())
                    prev = int(tokens[d, j])
                    full.append(prev)
                    cum.append(c)
                ok = full[:len(out)] == list(out)
                log.bad += 0 if ok else 1
                log.rounds.append({"first": int(first), "kept": len(out), "full": full, "cum": cum, "ok": ok})
            return out

        dr.chain = types.MethodType(chain, dr)

    def begin(self) -> None:
        self.active, self.rounds = True, []

    def end(self) -> list:
        self.active = False
        r, self.rounds = self.rounds, []
        return r


# -------------------------------------------------------------------------------------------------- ops ---
class Ops:
    def __init__(self, engine, rank: int, ctl_dir: str, health: str = "round", log=print) -> None:
        self.e, self.rank, self.ctl, self.log = engine, rank, ctl_dir, log
        self.runner = r = engine.runner
        self.w = r.w
        self.world = self.w.world
        self.vrows = r.vrows
        self.last: dict = {}
        self.verify_raw = r._verify                     # bound original (rowtime times graphs without the guard)
        from tensorfold.families.glm5_next.cuda import dflash2 as d2

        self.d2, self.edge_default = d2, d2.EDGE
        self.chain_log = ChainLog() if rank == 0 and r.drafter is not None else None
        if self.chain_log is not None:
            self.chain_log.install(r.drafter)
        orig_gen = r.generate

        def generate(*a, **k):                           # stash this rank's reply for the consensus
            st = orig_gen(*a, **k)
            out = st.get("out") or []
            self.last = {"depth": int(st.get("depth") or 0), "conf": round(float(st.get("confidence") or 0) * C.MICRO),
                         "n": len(out), "sha": C.sha_ints(out)}
            return st

        r.generate = generate
        self._flag = torch.zeros((6,), dtype=torch.int32, device="cuda")
        self._all = torch.zeros((self.world * 6,), dtype=torch.int32, device="cuda")
        self._h = torch.zeros((1,), dtype=torch.int32, device="cuda")
        self._hall = torch.zeros((self.world,), dtype=torch.int32, device="cuda")
        self.health = health
        if health == "round":
            orig_verify = r._verify

            def guarded(R, P, T, pick):                  # every round, every rank: the reductions are done (sync),
                orig_verify(R, P, T, pick)               # then all ranks agree nobody's RoCE runtime is poisoned
                torch.cuda.synchronize()                 # before the round's picks are read and emitted
                self._h.fill_(1 if self.poisoned() else 0)
                self.w.comm.all_gather(self._h, self._hall)
                if int(self._hall.max().item()):
                    self.log(f"[tf_speed] FATAL B12X_UNHEALTHY (ranks {self._hall.tolist()})")
                    if self.w.fast is not None:
                        self.w.fast.rt.check_health()     # raises with b12x's own message on the poisoned rank
                    raise RuntimeError("tf_speed: a rank's RoCE runtime is poisoned; tokens of this round dropped")

            r._verify = guarded

    # -- protocol ops -------------------------------------------------------------------------------------
    def poisoned(self) -> bool:
        f = self.w.fast
        return bool(f is not None and f.rt.poisoned)

    def depth_cap(self, depth: int) -> int:
        block = self.runner.drafter.block if self.runner.drafter is not None else C.BLOCK
        return C.effective_depth(depth, self.vrows, block)

    def apply_cfg(self, cfg: dict) -> None:
        C.write_dflash_cfg(self.ctl, cfg)
        self.d2.EDGE = self.edge_default if cfg.get("edge") is None else float(cfg["edge"])
        if cfg.get("profile"):                           # runner.RoundProfiler: the next 8 rounds, then it deletes it
            open(os.path.join(self.ctl, "PROFILE"), "w").close()

    def read_profile(self):
        """Rank 0, after a profiled request: runner.RoundProfiler's per-kernel table (prof/rank0.txt)."""
        p = os.path.join(self.ctl, "prof", f"rank{self.rank}.txt")
        try:
            with open(p) as f:
                lines = f.read().splitlines()
            os.replace(p, p + f".{int(time.time())}")
            return lines[:80]
        except OSError:
            return None

    def consensus(self):
        l = self.last or {}
        v = [l.get("depth", 0), l.get("conf", 0), l.get("n", 0), *(l.get("sha") or [0, 0]), 1 if self.poisoned() else 0]
        self._flag.copy_(torch.tensor(v, dtype=torch.int32))
        self.w.comm.all_gather(self._flag, self._all)
        got = self._all.view(self.world, 6).tolist()
        self.last = {}
        return got if self.rank == 0 else None

    def mem(self) -> dict:
        with open("/proc/meminfo") as f:
            m = {ln.split(":")[0]: int(ln.split()[1]) for ln in f}
        return {"avail_gib": round(m["MemAvailable"] / 2**20, 2),
                "swap_used_mib": round((m["SwapTotal"] - m["SwapFree"]) / 1024, 1)}

    def chain_log_begin(self) -> None:
        if self.chain_log is not None:
            self.chain_log.begin()

    def chain_log_end(self):
        return self.chain_log.end() if self.chain_log is not None else None

    @torch.no_grad()
    def rowtime(self, R: int, P0: int, reps: int, ids: list[int]):
        """Graph replays of an R-row verify window at P0 (the reply's own tokens: the cache rows it rewrites get the
        same bits), each followed by the host read of the picks as in the decode loop; rank 0's wall ms."""
        r = self.runner
        vb = r.vb
        T = r._T(P0 + R)
        vb.ids[:R].copy_(torch.tensor(ids[:R], dtype=torch.long))
        torch.cuda.synchronize()
        ms = []
        for _ in range(reps + 2):
            t0 = time.perf_counter()
            self.verify_raw(R, P0, T, "argmax")
            vb.argmax[:R].tolist()
            ms.append(1e3 * (time.perf_counter() - t0))
        # same window through the decode loop's (possibly guarded) _verify: guarded - raw = health-check cost.
        # Every rank runs it (the guard all-gathers), so the collectives stay in step.
        mg = []
        if self.health == "round":
            for _ in range(reps + 2):
                t0 = time.perf_counter()
                r._verify(R, P0, T, "argmax")
                vb.argmax[:R].tolist()
                mg.append(1e3 * (time.perf_counter() - t0))
        self.last_guarded = [round(x, 3) for x in mg[2:]] if self.rank == 0 and mg else None
        return [round(x, 3) for x in ms[2:]] if self.rank == 0 else None

    @torch.no_grad()
    def draft_time(self, pending: int, reps: int):
        dr = self.runner.drafter
        if dr is None:
            return None
        ms = []
        for _ in range(reps + 2):
            t0 = time.perf_counter()
            dr.candidates(int(pending), dr.block - 1)
            ms.append(1e3 * (time.perf_counter() - t0))
        return [round(x, 3) for x in ms[2:]] if self.rank == 0 else None

    @torch.no_grad()
    def probe(self, R: int, P0: int, ids: list[int]):
        """One eager R-row window (fused.compute, the graphs' own kernels) with fused.route wrapped to record every MoE
        layer's picks; distinct experts per layer for each prefix of 1..R rows, and the rows' argmax (teacher forced:
        must equal the serial reply)."""
        from tensorfold.families.glm_moe_dsa.cuda import fused

        r = self.runner
        w, st, vb = r.w, r.st, r.vb
        K = w.cfg.num_experts_per_tok
        rec = torch.full((len(w.layers) + 1, R, K), -1, dtype=torch.int32, device="cuda")
        orig = fused.route

        def route(w_, L, b, r0, n):
            orig(w_, L, b, r0, n)
            if b is vb and L.index < rec.shape[0]:
                rec[L.index, r0:r0 + n].copy_(b.pick[r0:r0 + n])

        fused.route = route
        try:
            vb.ids[:R].copy_(torch.tensor(ids[:R], dtype=torch.long))
            st.pos.fill_(P0)
            fused.compute(w, st, vb, R, r._T(P0 + R), logits="all", pick="argmax")
            torch.cuda.synchronize()
        finally:
            fused.route = orig
        if self.rank != 0:
            return None
        picks = rec.cpu().numpy()
        argmax = vb.argmax[:R].tolist()
        per_prefix = []
        for p in range(1, R + 1):
            counts = [len(set(picks[l, :p].ravel().tolist()) - {-1}) for l in range(picks.shape[0])
                      if picks[l, 0, 0] >= 0]
            per_prefix.append({"rows": p, "mean": round(sum(counts) / len(counts), 3), "min": min(counts),
                               "max": max(counts), "layers": len(counts)})
        return {"distinct": per_prefix, "argmax": [int(t) for t in argmax]}
