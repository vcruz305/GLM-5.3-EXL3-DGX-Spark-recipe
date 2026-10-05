"""Stdlib-only helpers for the TensorFold GLM-5.3 TP=4 speed sweep (plans, DFlash2 config files, exactness, tables,
off-policy evaluation of DFlash2 depth/confidence from chain logs). Imported by the rank launcher, the protocol, the
CLIs and the CPU tests; nothing here touches torch or a GPU."""

from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
from typing import Any

DEFAULT_DEPTH, DEFAULT_CONF, DEFAULT_EDGE = 7, 0.3, 0.6
BLOCK = 8                       # GLM-5.3-DFlash2 block_size: at most BLOCK - 1 = 7 drafts a round, whatever VERIFY_ROWS
MICRO = 1_000_000
CORE = ("code", "prose", "math")    # the three prompts of the 2026-10-05 baseline (mean 38.55 tok/s)


# ------------------------------------------------------------------------------------------------- plans ---
def load_plan(path: str) -> dict:
    with open(path) as f:
        plan = json.load(f)
    validate_plan(plan)
    return plan


def validate_plan(plan: dict) -> None:
    """Raise ValueError on anything the sweep would trip over later (names, ranges, references)."""
    for k in ("name", "max_tokens", "reps", "prompts", "configs"):
        if k not in plan:
            raise ValueError(f"plan: missing {k!r}")
    if not (1 <= int(plan["max_tokens"]) <= 4096):
        raise ValueError("plan: max_tokens out of range")
    if not (1 <= int(plan["reps"]) <= 9):
        raise ValueError("plan: reps out of range")
    names = [p["name"] for p in plan["prompts"]]
    if len(set(names)) != len(names) or not names:
        raise ValueError("plan: prompt names must be unique and non-empty")
    for p in plan["prompts"]:
        msgs = p.get("messages")
        if not msgs or not all(m.get("role") in ("system", "user", "assistant") and isinstance(m.get("content"), str)
                               for m in msgs) or msgs[-1]["role"] != "user":
            raise ValueError(f"plan: prompt {p['name']!r} needs messages ending with a user turn")
        if p.get("effort") not in (None, "low", "high"):
            raise ValueError(f"plan: prompt {p['name']!r} effort {p.get('effort')!r} (template: low | high | None = max)")
    cnames = [c["name"] for c in plan["configs"]]
    if len(set(cnames)) != len(cnames):
        raise ValueError("plan: config names must be unique")
    for c in plan["configs"]:
        encode_cfg(c)                                    # range checks
    rt = plan.get("rowtime") or {}
    if rt.get("enabled"):
        for r in rt.get("rows", []):
            if not 1 <= int(r) <= BLOCK:
                raise ValueError("plan: rowtime rows must be 1..8")
        for at in rt.get("at", []):
            if at["prompt"] not in names:
                raise ValueError(f"plan: rowtime prompt {at['prompt']!r} unknown")
    for item in plan.get("profile", []):
        if item.get("prompt") not in names or item.get("mode") not in ("off", "dflash"):
            raise ValueError(f"plan: profile item {item}")
        if item["mode"] == "dflash":
            encode_cfg(item.get("config") or {})
    ep = plan.get("expert_probe") or {}
    if ep.get("enabled"):
        for n in ep.get("prompts", []):
            if n not in names:
                raise ValueError(f"plan: expert_probe prompt {n!r} unknown")
        if not 1 <= int(ep.get("rows", 8)) <= BLOCK:
            raise ValueError("plan: expert_probe rows must be 1..8")
    if plan.get("health", "round") not in ("round", "request", "off"):
        raise ValueError("plan: health must be round | request | off")


# ------------------------------------------------------------------------------------- DFlash2 configs ---
def encode_cfg(cfg: dict | None) -> list[int]:
    """[depth, confidence in millionths, edge in millionths or -1 (the drafter's default), profile 0/1] - ints for the
    all-gather. profile 1: every rank touches PROFILE before the request (runner.RoundProfiler traces 8 rounds)."""
    if cfg is None:
        return [DEFAULT_DEPTH, round(DEFAULT_CONF * MICRO), -1, 0]
    d = int(cfg.get("depth", DEFAULT_DEPTH))
    c = float(cfg.get("confidence", DEFAULT_CONF))
    e = cfg.get("edge")
    if not 0 <= d <= BLOCK - 1:
        raise ValueError(f"depth {d}: 0..{BLOCK - 1} (block {BLOCK})")
    if not 0.0 <= c <= 1.0:
        raise ValueError(f"confidence {c}: 0..1")
    if e is not None and not 0.0 <= float(e) <= 4.0:
        raise ValueError(f"edge {e}: 0..4")
    return [d, round(c * MICRO), -1 if e is None else round(float(e) * MICRO), 1 if cfg.get("profile") else 0]


def decode_cfg(v: list[int]) -> dict:
    d, c, e = (int(x) for x in v[:3])
    return {"depth": d, "confidence": c / MICRO, "edge": None if e < 0 else e / MICRO,
            "profile": bool(len(v) > 3 and int(v[3]))}


def write_dflash_cfg(ctl_dir: str, cfg: dict) -> str:
    """The file runner._generate_dflash reads at the start of every request (dirname(TF_GLM53_PROFILE_FLAG)/DFLASH_CFG),
    written atomically (tmp + rename): a half-written file would make that rank fall back to the env defaults while the
    others use the file - different verify windows on different ranks."""
    os.makedirs(ctl_dir, exist_ok=True)
    path = os.path.join(ctl_dir, "DFLASH_CFG")
    tmp = path + f".tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump({"depth": int(cfg["depth"]), "confidence": float(cfg["confidence"])}, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return path


def read_dflash_cfg(ctl_dir: str) -> dict | None:
    try:
        with open(os.path.join(ctl_dir, "DFLASH_CFG")) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def effective_depth(depth: int, vrows: int, block: int = BLOCK) -> int:
    """runner._generate_dflash: depth_max = min(drafter.block - 1, vrows - 1, depth)."""
    return min(block - 1, vrows - 1, int(depth))


# ------------------------------------------------------------------------------------------- exactness ---
def sha_ids(ids: list[int]) -> str:
    """The engine's own reply hash: sha256 of json.dumps(out), first 16 hex digits."""
    return hashlib.sha256(json.dumps([int(t) for t in ids]).encode()).hexdigest()[:16]


def sha_ints(ids: list[int]) -> list[int]:
    """Two non-negative int32 words of the reply hash (for an int all-gather)."""
    h = int(hashlib.sha256(json.dumps([int(t) for t in ids]).encode()).hexdigest()[:15], 16)
    return [h & 0x7FFFFFFF, (h >> 31) & 0x7FFFFFFF]


def first_divergence(a: list[int], b: list[int]) -> int | None:
    for j in range(max(len(a), len(b))):
        if j >= len(a) or j >= len(b) or a[j] != b[j]:
            return j
    return None


# ------------------------------------------------------------------------------------------------ stats ---
def median(v: list[float]) -> float | None:
    v = [x for x in v if x is not None]
    return statistics.median(v) if v else None


def summarize(results: dict) -> dict:
    """runs -> {config: {prompt: medians}} plus per-config means; serial from the 'off' runs."""
    runs = results.get("runs", [])
    prompts = [p for p in results.get("prompt_order", [])] or sorted({r["prompt"] for r in runs})
    table: dict[str, dict] = {}
    for r in runs:
        if r.get("phase") not in ("serial", "config", "bookend"):
            continue
        key = "serial" if r["mode"] == "off" else r["config"]
        cell = table.setdefault(key, {}).setdefault(r["prompt"], {"tok_s": [], "tpr": [], "verify": [], "draft": [],
                                                                  "round_p50": [], "identical": [], "n": []})
        st = r.get("stats") or {}
        cell["tok_s"].append(st.get("tok_s"))
        cell["tpr"].append(st.get("tokens_per_round"))
        cell["verify"].append((st.get("ms_per_round") or {}).get("verify"))
        cell["draft"].append((st.get("ms_per_round") or {}).get("draft"))
        cell["round_p50"].append((st.get("round_ms") or {}).get("p50"))
        cell["identical"].append(r.get("identical"))
        cell["n"].append(r.get("n"))
    out: dict[str, Any] = {"prompts": prompts, "rows": {}}
    for key, cells in table.items():
        row = {}
        for p, c in cells.items():
            row[p] = {"tok_s": median(c["tok_s"]), "tpr": median(c["tpr"]), "verify_ms": median(c["verify"]),
                      "draft_ms": median(c["draft"]), "round_p50_ms": median(c["round_p50"]),
                      "reps": len(c["tok_s"]), "identical": all(x is not False for x in c["identical"]),
                      "n": median(c["n"])}
        core = [row[p]["tok_s"] for p in CORE if p in row and row[p]["tok_s"]]
        allp = [row[p]["tok_s"] for p in prompts if p in row and row[p]["tok_s"]]
        row["_mean_core"] = sum(core) / len(core) if len(core) == len(CORE) else None
        row["_mean_all"] = sum(allp) / len(allp) if allp else None
        row["_exact"] = all(row[p]["identical"] for p in cells)
        out["rows"][key] = row
    return out


def _f(x, nd=2):
    return "-" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def table_md(results: dict, title: str = "") -> str:
    s = summarize(results)
    prompts = s["prompts"]
    lines = []
    if title:
        lines.append(f"### {title}")
    hdr = "| config | " + " | ".join(f"{p} tok/s (tok/round, verify ms)" for p in prompts) + \
          " | mean core3 | mean all | exact |"
    lines += [hdr, "|" + "---|" * (len(prompts) + 4)]
    order = ["serial"] + [c["name"] for c in results.get("configs", []) if c["name"] in s["rows"]]
    order += [k for k in s["rows"] if k not in order]
    for key in order:
        row = s["rows"].get(key)
        if row is None:
            continue
        cells = []
        for p in prompts:
            c = row.get(p)
            cells.append("-" if c is None else f"{_f(c['tok_s'])} ({_f(c['tpr'], 2)}, {_f(c['verify_ms'], 1)})")
        lines.append(f"| {key} | " + " | ".join(cells) + f" | {_f(row['_mean_core'])} | {_f(row['_mean_all'])} | "
                     f"{'yes' if row['_exact'] else 'NO'} |")
    return "\n".join(lines)


# ----------------------------------------------------------------------------- off-policy DFlash2 model ---
def truncate(cum: list[float], depth: int, conf: float) -> int:
    """How many drafts Drafter.chain keeps from a full chain with cumulative confidences ``cum``: at most ``depth``;
    the first is always kept; stop at the first later one whose cumulative confidence is < conf (conf > 0 only)."""
    k = 0
    for d in range(min(depth, len(cum))):
        if conf > 0 and d > 0 and cum[d] < conf:
            break
        k += 1
    return k


def accepted_len(full: list[int], ref: list[int], start: int) -> int:
    """Leading drafts that equal the serial reply: full[i] vs ref[start + i] (ref[start] = the first draft's slot)."""
    n = 0
    for i, t in enumerate(full):
        if start + i >= len(ref) or ref[start + i] != t:
            break
        n += 1
    return n


def linear_verify_ms(intercept: float = 29.69, slope: float = 10.72):
    """ms(R) fitted to the 2026-10-05 NCCL run (serial 40.6 ms at R = 1; code/prose/math verify windows)."""
    return lambda R: intercept + slope * R


def eval_policies(rounds: list[dict], ref: list[int], L0: int, verify_ms, draft_ms: float, taps_ms: float,
                  policies: list[tuple[int, float]]) -> dict:
    """Replay logged rounds (full chains + cumulative confidences, one per round of a drafted run) under other
    (depth, confidence) policies: tokens a round = min(accepted, kept) + 1, cost = draft + verify(kept + 1) + taps.
    Off-policy: the rounds start where the logging run's rounds started."""
    out = {}
    for depth, conf in policies:
        tok = ms = 0.0
        rows = 0
        for rd in rounds:
            full, cum = rd["full"], rd["cum"]
            start = rd["first"] - L0                      # reply index of the first draft
            k = truncate(cum, depth, conf)
            acc = min(accepted_len(full, ref, start), k)
            tok += acc + 1
            ms += draft_ms + verify_ms(k + 1) + taps_ms
            rows += k + 1
        n = max(len(rounds), 1)
        out[f"d{depth}c{conf:.2f}"] = {"depth": depth, "confidence": conf, "tokens_per_round": tok / n,
                                       "rows_per_round": rows / n, "ms_per_round": ms / n,
                                       "tok_s": 1e3 * tok / ms if ms else None}
    return out


def verify_table(rowtime: list[dict], ctx: str = "short") -> dict[int, float] | None:
    """{R: median ms} from rowtime records (rank 0, graph replay + host read of the picks)."""
    by: dict[int, list[float]] = {}
    for r in rowtime:
        if r.get("ctx", "short") != ctx:
            continue
        by.setdefault(int(r["R"]), []).extend(r["ms"])
    return {R: statistics.median(v) for R, v in sorted(by.items())} if by else None


def interp_verify(tab: dict[int, float]):
    keys = sorted(tab)

    def f(R):
        if R in tab:
            return tab[R]
        lo = max([k for k in keys if k <= R], default=keys[0])
        hi = min([k for k in keys if k >= R], default=keys[-1])
        if lo == hi:
            return tab[lo]
        return tab[lo] + (tab[hi] - tab[lo]) * (R - lo) / (hi - lo)
    return f


def iid_distinct(R: float, E: int = 256, k: int = 8) -> float:
    """Expected distinct experts a MoE layer reads for R rows if every row picked k of E independently."""
    return E * (1 - (1 - k / E) ** R)


def atomic_json(path: str, obj) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def render_chat(template: str, messages: list[dict], effort: str | None) -> str:
    """The model's chat_template.jinja rendered as the 2026-10-05 launcher did (jinja2 sandbox + loopcontrols,
    add_generation_prompt; reasoning_effort only when given: the template's default is 'max')."""
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"])

    def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
        return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)

    def raise_exception(msg):
        raise RuntimeError(msg)

    env.filters["tojson"] = tojson
    env.globals["raise_exception"] = raise_exception
    kw = dict(messages=messages, add_generation_prompt=True)
    if effort:
        kw["reasoning_effort"] = effort
    return env.from_string(template).render(**kw)


def isfinite(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)
