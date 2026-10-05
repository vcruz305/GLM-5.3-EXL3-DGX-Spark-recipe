"""Long-prompt drafter test through the /v1 server: per prompt, greedy, thinking off, 512 new tokens,
"draft": false (serial) then the default (DFlash2 d7/c0.60). Records client TTFT, effective prefill tok/s
(usage.prompt_tokens / TTFT), client decode tok/s ((n-1)/(last-first delta)), the server's "tensorfold" stats
(tok_s, tokens_per_round, ...), /health counters before/after each request, and whether drafted ids == serial ids.

usage: python3 bench/longctx_run.py BASE MODEL PROMPTS_JSON OUT_JSON [names,comma]
  e.g. python3 bench/make_prompts.py ~/glm53-tensorfold/view       (writes bench/longctx_prompts.json)
       python3 bench/longctx_run.py http://127.0.0.1:8890/v1 GLM-5.3-EXL3-3.38bpw bench/longctx_prompts.json \\
           longctx_128k.json book_128k
  THINKING=1 python3 bench/longctx_run.py ... bench/sweep6_prompts.json sweep6.json   (the 6 sweep prompts, thinking
  on as measured: serial vs drafted ids and tok/s per prompt)
A 128K-token prompt takes ~8.5 minutes to its first token at ~250 tok/s prefill; the server answers one request at a
time, so run nothing else against it meanwhile.
"""
import json
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_v1 as B  # noqa: E402

base, model, pj, out = sys.argv[1:5]
names = sys.argv[5].split(",") if len(sys.argv) > 5 and sys.argv[5] else None
cl = B.Client(base, model, timeout=3600)
THINK = os.environ.get("THINKING", "0") == "1"      # the sweep prompts were measured with thinking on


def health():
    with urllib.request.urlopen(cl.root + "/health", timeout=30) as r:
        return json.loads(r.read())


res = {"base": base, "model": model, "started": time.strftime("%Y-%m-%d %H:%M:%S"), "rows": []}
for p in json.load(open(pj, encoding="utf-8"))["prompts"]:
    if names and p["name"] not in names:
        continue
    served = B.tokenize(cl, p["messages"], THINK)
    row = {"name": p["name"], "source": p.get("source", "sweep_reference.json"), "prompt_tokens_tokenize": len(served),
           "thinking": THINK, "runs": {}}
    if p.get("serial_ids"):
        row["sweep_serial_ids"] = p["serial_ids"]
    print(f"{p['name']}: {len(served)} prompt tokens served ({p.get('source', 'sweep')})", flush=True)
    for arm, extra in (("serial", {"draft": False}), ("dflash", {})):
        h0 = health()
        body = B.chat_body(p["messages"], p.get("max_tokens", 512), THINK, **extra)
        t = time.time()
        r = cl.stream(body)
        h1 = health()
        if "error" in r:
            row["runs"][arm] = {"error": r["error"], "total_s": r.get("total_s")}
            print(f"  {arm}: ERROR {r['error']}", flush=True)
            continue
        u = r.get("usage") or {}
        pt = u.get("prompt_tokens")
        d = {k: (h1.get(k, 0) or 0) - (h0.get(k, 0) or 0) for k in
             ("prefill_seconds_total", "decode_seconds_total", "prompt_tokens_total", "completion_tokens_total")}
        run = {"wall_start": t, "ttft_s": r["ttft_s"], "total_s": r["total_s"], "usage": u, "n": r["n"],
               "finish": r["finish"], "client_decode_tok_s": r["client_tok_s"],
               "effective_prefill_tok_s": (pt / r["ttft_s"]) if pt and r["ttft_s"] else None,
               "server_stats": r["stats"], "health_delta": d,
               "server_prefill_tok_s": (d["prompt_tokens_total"] / d["prefill_seconds_total"])
               if d.get("prefill_seconds_total") else None,
               "content_head": r["content"][:400], "ids": r["ids"]}
        row["runs"][arm] = run
        st = r["stats"]
        print(f"  {arm}: TTFT {r['ttft_s']:.2f} s, prompt {pt}, eff prefill {run['effective_prefill_tok_s'] or 0:.1f} tok/s,"
              f" n {r['n']} {r['finish']}, client decode {r['client_tok_s']}, server tok_s {st.get('tok_s')}, "
              f"tokens/round {st.get('tokens_per_round')}, mode {st.get('mtp_mode')}", flush=True)
    a, b = row["runs"].get("serial", {}), row["runs"].get("dflash", {})
    row["ids_equal"] = bool(a.get("ids")) and a.get("ids") == b.get("ids")
    row["first_diff"] = B.first_diff(a.get("ids") or [], b.get("ids") or [])
    if p.get("serial_ids"):
        row["serial_eq_sweep"] = a.get("ids") == p["serial_ids"]
        row["serial_first_diff_vs_sweep"] = B.first_diff(a.get("ids") or [], p["serial_ids"])
    print(f"  drafted ids == serial ids: {row['ids_equal']} (first diff {row['first_diff']}); serial == sweep serial: "
          f"{row.get('serial_eq_sweep')} (first diff {row.get('serial_first_diff_vs_sweep')})", flush=True)
    res["rows"].append(row)
    json.dump(res, open(out, "w", encoding="utf-8"), indent=1)
json.dump(res, open(out, "w", encoding="utf-8"), indent=1)
print("wrote", out)
