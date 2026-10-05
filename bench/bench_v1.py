#!/usr/bin/env python3
"""/v1 smoke test and benchmark for the GLM-5.3 TP=4 TensorFold server (stdlib only: runs on rank 0's system python3
or anywhere with a tunnel). serve.sh smoke / serve.sh bench run it on rank 0.

  python3 bench/bench_v1.py --smoke
  python3 bench/bench_v1.py [--reps 3] [--no-serial] [--ids reference|self] [--base http://127.0.0.1:8890/v1]

Benchmark: the 2026-10-05 speed sweep's 6 prompts exactly as the sweep sent them (thinking on, no effort = Max,
greedy, 512 tokens), streamed with "return_token_ids": true. For each prompt /tokenize must give the reference prompt
ids. With --ids reference (fast-160k) the "draft": false reply and every drafted reply (the server default, DFlash2
d7/c0.60) must equal the reference serial ids token for token; with --ids self (dcp4-262k, int4-262k: other numeric
paths) every drafted reply must equal this server's own serial reply. Speed: the server's own decode rate (the
"tensorfold" stats block: tokens after the first over decode time) and the client's (tokens after the first over
first-to-last delta), medians over --reps, next to the sweep's d7c0.60 medians (arms A and C). Exit 1 on any id
mismatch or failed check.

Smoke: /v1/models, /health, /tokenize parity (thinking on = official template, thinking off = `</think>` prompt),
a thinking-off reply that must stop at a turn-end id with clean content, a thinking-on streamed reply whose reasoning
and content stay split with no `</think>` / turn-end markup in either channel, drafted == "draft": false ids, and the
refusal of an unsupported "tf_mtp".
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request

LEAKS = ("</think>", "<think>", "<|user|>", "<|observation|>", "<|assistant|>", "<|endoftext|>", "<|system|>")


class Client:
    def __init__(self, base: str, model: str, key: str | None = None, timeout: float = 1800.0):
        self.base, self.model, self.key, self.timeout = base.rstrip("/"), model, key, timeout
        self.root = self.base[:-3] if self.base.endswith("/v1") else self.base

    def _req(self, url: str, body: dict | None):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(url, data=data, method="GET" if body is None else "POST")
        req.add_header("Content-Type", "application/json")
        if self.key:
            req.add_header("Authorization", f"Bearer {self.key}")
        return urllib.request.urlopen(req, timeout=self.timeout)

    def get(self, path: str) -> dict:
        with self._req(self.base + path, None) as r:
            return json.loads(r.read())

    def post(self, path: str, body: dict, root: bool = False) -> tuple[int, dict]:
        try:
            with self._req((self.root if root else self.base) + path, body) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            try:
                return e.code, json.loads(e.read())
            except ValueError:
                return e.code, {}

    def stream(self, body: dict) -> dict:
        """POST a streamed chat completion; per-chunk deltas with arrival times, the final stats and usage."""
        body = dict(body, stream=True, model=self.model)
        t0 = time.perf_counter()
        chunks, final, err = [], None, None
        with self._req(self.base + "/chat/completions", body) as r:
            for raw in r:
                line = raw.decode("utf-8").strip()
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    break
                obj = json.loads(payload)
                now = time.perf_counter() - t0
                if "error" in obj:
                    err = obj["error"]
                    continue
                if obj.get("choices"):
                    ch = obj["choices"][0]
                    d = ch.get("delta") or {}
                    if d.get("content") or d.get("reasoning_content"):
                        chunks.append({"t": now, "content": d.get("content") or "",
                                       "reasoning": d.get("reasoning_content") or ""})
                    if ch.get("finish_reason"):
                        final = obj
                elif obj.get("usage") and final is not None:
                    final["usage"] = obj["usage"]
        total = time.perf_counter() - t0
        if err is not None:
            return {"error": err, "total_s": total}
        stats = (final or {}).get("tensorfold") or {}
        ids = stats.get("token_ids") or []
        ts = [c["t"] for c in chunks]
        n = len(ids)
        return {"ids": ids, "n": n, "finish": ((final or {}).get("choices") or [{}])[0].get("finish_reason"),
                "usage": (final or {}).get("usage"), "stats": {k: v for k, v in stats.items() if k != "token_ids"},
                "content": "".join(c["content"] for c in chunks), "reasoning": "".join(c["reasoning"] for c in chunks),
                "chunks": len(chunks), "both_in_one_chunk": sum(1 for c in chunks if c["content"] and c["reasoning"]),
                "ttft_s": ts[0] if ts else None, "total_s": total,
                "client_tok_s": (n - 1) / (ts[-1] - ts[0]) if n > 1 and len(ts) > 1 and ts[-1] > ts[0] else None}


def chat_body(messages, max_tokens, thinking: bool | None, **extra) -> dict:
    b = {"messages": messages, "max_tokens": max_tokens, "temperature": 0, "return_token_ids": True}
    if thinking is not None:
        b["chat_template_kwargs"] = {"enable_thinking": bool(thinking)}
    b.update(extra)
    return b


def first_diff(a, b):
    for i in range(max(len(a), len(b))):
        if i >= len(a) or i >= len(b) or a[i] != b[i]:
            return i
    return None


class Checks:
    def __init__(self):
        self.items = []

    def add(self, name: str, ok: bool, detail="") -> bool:
        self.items.append({"check": name, "ok": bool(ok), "detail": detail})
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""), flush=True)
        return ok

    @property
    def ok(self) -> bool:
        return all(i["ok"] for i in self.items)


def tokenize(cl: Client, messages, thinking: bool) -> list[int]:
    code, r = cl.post("/tokenize", {"model": cl.model, "messages": messages,
                                    "chat_template_kwargs": {"enable_thinking": thinking}}, root=True)
    if code != 200:
        raise RuntimeError(f"/tokenize HTTP {code}: {r}")
    return r["tokens"]


def smoke(cl: Client, ref: dict, ck: Checks) -> dict:
    out = {}
    models = cl.get("/models")
    ids = [m["id"] for m in models.get("data", [])]
    ck.add("/v1/models lists the served name", cl.model in ids, f"{ids}")
    with urllib.request.urlopen(cl.root + "/health", timeout=30) as r:
        ck.add("/health 200", r.status == 200, r.read().decode()[:200])
    eos = set(ref["eos_ids"])
    for p in ref["prompts"]:
        got = tokenize(cl, p["messages"], True)
        ck.add(f"/tokenize {p['name']} thinking on == sweep prompt ids", got == p["prompt_ids"],
               f"{len(got)} vs {p['prompt_len']} tokens, first diff {first_diff(got, p['prompt_ids'])}")
    q = [{"role": "user", "content": "Reply with exactly the word OK and nothing else."}]
    off = tokenize(cl, q, False)
    on = tokenize(cl, q, True)
    ck.add("thinking off prompt ends <|assistant|></think>", off[-2:] == [154828, ref["think_end_id"]], f"{off[-3:]}")
    ck.add("thinking on prompt ends <|assistant|><think>", on[-2:] == [154828, ref["think_open_id"]], f"{on[-3:]}")
    r = cl.stream(chat_body(q, 64, False))
    out["thinking_off"] = r
    ck.add("thinking off: no error", "error" not in r, str(r.get("error", ""))[:200])
    if "error" not in r:
        ck.add("thinking off: finish stop at a turn-end id", r["finish"] == "stop" and r["ids"] and r["ids"][-1] in eos,
               f"finish {r['finish']}, last id {r['ids'][-1:]}, n {r['n']}")
        ck.add("thinking off: empty reasoning, content has OK", not r["reasoning"] and "OK" in r["content"],
               f"content {r['content'][:80]!r}")
        ck.add("thinking off: no markup leak", not any(t in r["content"] for t in LEAKS), r["content"][:80])
    r = cl.stream(chat_body([{"role": "user", "content": "What is 17 * 23? Answer with the number."}], 1024, True))
    out["thinking_on"] = r
    ck.add("thinking on: no error", "error" not in r, str(r.get("error", ""))[:200])
    if "error" not in r:
        ck.add("thinking on: reasoning non-empty", bool(r["reasoning"].strip()), f"{len(r['reasoning'])} chars")
        ck.add("thinking on: no markup leak in either channel",
               not any(t in r["content"] or t in r["reasoning"] for t in LEAKS), "")
        if r["finish"] == "stop":
            ck.add("thinking on: stops at a turn-end id with an answer", r["ids"][-1] in eos and "391" in r["content"],
                   f"content {r['content'][:80]!r}")
        else:
            ck.add("thinking on: finish", r["finish"] == "length", f"{r['finish']} (1024 tokens still thinking?)")
        print(f"       (chunks carrying both channels: {r['both_in_one_chunk']}; TF sends one delta per token, so the "
              f"token that closes </think> may carry both - read both fields)")
    short = [{"role": "user", "content": "Name three prime numbers above 100."}]
    a = cl.stream(chat_body(short, 96, False, draft=False))
    b = cl.stream(chat_body(short, 96, False))
    ok = "error" not in a and "error" not in b and a["ids"] == b["ids"] and a["n"] > 0
    ck.add("drafted ids == \"draft\": false ids", ok,
           f"n {a.get('n')} / {b.get('n')}, modes {a.get('stats', {}).get('tf_serve')} / {b.get('stats', {}).get('tf_serve')}")
    ck.add("default request drafts with DFlash2", (b.get("stats") or {}).get("mtp_mode") == "dflash",
           f"mtp_mode {(b.get('stats') or {}).get('mtp_mode')}, depth {(b.get('stats') or {}).get('depth')}, "
           f"confidence {(b.get('stats') or {}).get('confidence')}")
    bad = cl.stream(chat_body(short, 8, False, tf_mtp="auto"))
    ck.add("tf_mtp other than dflash refused", "error" in bad, str(bad.get("error", ""))[:160])
    return out


def bench(cl: Client, ref: dict, ck: Checks, reps: int, serial: bool, names: list[str] | None,
          ids_mode: str = "reference") -> dict:
    """ids_mode "reference": every reply (serial and drafted) must equal the pinned reference serial ids - the
    measured fast path (fast-160k / 32K) reproduces them token for token. "self": drafted replies must equal this
    server's own "draft": false reply (a profile on another numeric path: DCP=4, int4 cache)."""
    rows = []
    if ids_mode == "self" and not serial:
        raise SystemExit("--ids self needs the serial run (drop --no-serial)")
    warm = cl.stream(chat_body([{"role": "user", "content": "Hello"}], 16, True))
    ck.add("warm-up request", "error" not in warm, str(warm.get("error", ""))[:200])
    for p in ref["prompts"]:
        if names and p["name"] not in names:
            continue
        got = tokenize(cl, p["messages"], True)
        ck.add(f"{p['name']}: prompt ids == reference", got == p["prompt_ids"],
               f"first diff {first_diff(got, p['prompt_ids'])}")
        body = chat_body(p["messages"], p["max_tokens"], True)
        row = {"prompt": p["name"], "prompt_len": p["prompt_len"], "runs": []}
        want = p["serial_ids"]
        if serial:
            r = cl.stream(dict(body, draft=False))
            same = r.get("ids") == p["serial_ids"]
            if ids_mode == "reference":
                ck.add(f"{p['name']}: serial ids == reference serial", same,
                       f"n {r.get('n')}, first diff {first_diff(r.get('ids') or [], p['serial_ids'])}, "
                       f"tok/s {r.get('stats', {}).get('tok_s')}")
            else:
                want = r.get("ids") or []
                ck.add(f"{p['name']}: serial reply complete", "error" not in r and bool(want),
                       f"n {r.get('n')}, tok/s {r.get('stats', {}).get('tok_s')}, equals the reference: {same} "
                       f"(first diff {first_diff(want, p['serial_ids'])}; not required on this profile)")
            row["serial"] = {"tok_s": r.get("stats", {}).get("tok_s"), "client_tok_s": r.get("client_tok_s"),
                             "identical": same if ids_mode == "reference" else True, "equals_reference": same,
                             "finish": r.get("finish"), "ttft_s": r.get("ttft_s")}
        for rep in range(reps):
            r = cl.stream(body)
            same = r.get("ids") == want
            st = r.get("stats") or {}
            ck.add(f"{p['name']} rep {rep}: drafted ids == {'reference serial' if ids_mode == 'reference' else 'this server serial'}",
                   same,
                   f"n {r.get('n')}, first diff {first_diff(r.get('ids') or [], want)}, tok/s "
                   f"{st.get('tok_s')}, tokens/round {st.get('tokens_per_round')}, mode {st.get('mtp_mode')} "
                   f"d{st.get('depth')} c{st.get('confidence')}")
            row["runs"].append({"tok_s": st.get("tok_s"), "client_tok_s": r.get("client_tok_s"),
                                "tokens_per_round": st.get("tokens_per_round"), "ttft_s": r.get("ttft_s"),
                                "ms_per_round": st.get("ms_per_round"), "identical": same, "finish": r.get("finish"),
                                "mtp_mode": st.get("mtp_mode"), "depth": st.get("depth"),
                                "confidence": st.get("confidence")})
        med = lambda k: statistics.median([x[k] for x in row["runs"] if x.get(k) is not None]) if row["runs"] else None  # noqa: E731
        row["tok_s"], row["client_tok_s"], row["tpr"] = med("tok_s"), med("client_tok_s"), med("tokens_per_round")
        row["sweep_A"], row["sweep_C"] = p["tok_s"].get("A_roce_d7c0.60"), p["tok_s"].get("C_roce_d7c0.60")
        rows.append(row)
    print("\n| prompt | tokens | server tok/s (median) | client tok/s | tokens/round | sweep A d7c0.60 | sweep C | "
          "server / A | serial tok/s | exact |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        ratio = r["tok_s"] / r["sweep_A"] if r["tok_s"] and r["sweep_A"] else None
        ex = all(x["identical"] for x in r["runs"]) and (r.get("serial") or {}).get("identical", True)
        print(f"| {r['prompt']} | {r['prompt_len']} | {fmt(r['tok_s'])} | {fmt(r['client_tok_s'])} | {fmt(r['tpr'])} | "
              f"{fmt(r['sweep_A'])} | {fmt(r['sweep_C'])} | {fmt(ratio, 3)} | {fmt((r.get('serial') or {}).get('tok_s'))} "
              f"| {'yes' if ex else 'NO'} |")
    full = [r for r in rows if r["tok_s"]]
    if len(full) == len(ref["prompts"]):
        m = sum(r["tok_s"] for r in full) / len(full)
        print(f"\nmean over all 6 prompts: server {m:.2f} tok/s vs sweep A {ref['mean_all']['A_roce_d7c0.60']} / "
              f"C {ref['mean_all']['C_roce_d7c0.60']}")
    return {"rows": rows}


def fmt(x, nd=2):
    return "-" if x is None else f"{x:.{nd}f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8890/v1")
    ap.add_argument("--model", default="GLM-5.3-EXL3-3.38bpw")
    ap.add_argument("--ref", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "sweep_reference.json"),
                    help="the 6 reference prompts with their prompt ids and serial reply ids (default: next to this file)")
    ap.add_argument("--ids", choices=("reference", "self"), default="reference",
                    help="reference: replies must equal the reference serial ids (fast-160k); self: drafted must "
                         "equal this server's own serial reply (dcp4-262k, int4-262k)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--no-serial", action="store_true", help="skip the \"draft\": false run per prompt")
    ap.add_argument("--prompts", default="", help="comma list (default: all 6)")
    ap.add_argument("--api-key", default=None)
    a = ap.parse_args()
    with open(a.ref) as f:
        ref = json.load(f)
    cl = Client(a.base, a.model, a.api_key)
    ck = Checks()
    t0 = time.time()
    if a.smoke:
        res = smoke(cl, ref, ck)
    else:
        res = bench(cl, ref, ck, a.reps, not a.no_serial, [n for n in a.prompts.split(",") if n] or None, a.ids)
    res.update(checks=ck.items, ok=ck.ok, base=a.base, model=a.model, ids_mode=None if a.smoke else a.ids,
               started=t0, wall_s=round(time.time() - t0, 1))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)
        print(f"wrote {a.out}")
    print(f"{'ALL CHECKS PASSED' if ck.ok else 'FAILED'}: {sum(i['ok'] for i in ck.items)}/{len(ck.items)}")
    return 0 if ck.ok else 1


if __name__ == "__main__":
    sys.exit(main())
