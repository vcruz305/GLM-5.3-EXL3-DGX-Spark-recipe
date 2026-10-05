#!/usr/bin/env bash
# One chat request against the running server, streamed to the terminal, then the engine's own stats for the reply
# (decode tok/s, tokens per DFlash2 round, draft mode). Run it on rank 0, or anywhere with a tunnel
# (bash tensorfold-four-spark-tp4/serve.sh tunnel). stdlib python3 only.
#
#   bash tensorfold-four-spark-tp4/chat.sh "Explain RoCE in two sentences."
#   THINKING=0 bash ... chat.sh "..."       thinking off (the server default is on)
#   DRAFT=0 bash ... chat.sh "..."          serial decode ("draft": false), the exactness reference
#   TEMPERATURE=0 bash ... chat.sh "..."    greedy (what the benchmarks ran); default: the model's sampling
#   BASE=http://127.0.0.1:8890/v1 MAX_TOKENS=1024 API_KEY=...
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"
BASE="${BASE:-http://127.0.0.1:$TFS_HTTP_PORT/v1}"
[[ $# -ge 1 ]] || { sed -n 2,11p "$0"; exit 2; }
BASE="$BASE" MODEL="$TFS_NAME" PROMPT="$*" python3 - <<'PY'
import json, os, sys, time, urllib.request
body = {"model": os.environ["MODEL"], "messages": [{"role": "user", "content": os.environ["PROMPT"]}],
        "max_tokens": int(os.environ.get("MAX_TOKENS", "1024")), "stream": True,
        "stream_options": {"include_usage": True}}
if os.environ.get("THINKING") in ("0", "1"):
    body["chat_template_kwargs"] = {"enable_thinking": os.environ["THINKING"] == "1"}
if os.environ.get("DRAFT") == "0":
    body["draft"] = False
if os.environ.get("TEMPERATURE"):
    body["temperature"] = float(os.environ["TEMPERATURE"])
req = urllib.request.Request(os.environ["BASE"].rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                             headers={"Content-Type": "application/json"})
if os.environ.get("API_KEY"):
    req.add_header("Authorization", "Bearer " + os.environ["API_KEY"])
t0, first, stats, usage, finish, in_reason = time.perf_counter(), None, {}, {}, None, False
with urllib.request.urlopen(req, timeout=3600) as r:
    for raw in r:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        obj = json.loads(data)
        if "error" in obj:
            sys.exit(f"\nserver error: {obj['error']}")
        for ch in obj.get("choices") or []:
            d = ch.get("delta") or {}
            for key, dim in (("reasoning_content", True), ("content", False)):   # one chunk can carry both
                txt = d.get(key)
                if txt:
                    first = first or time.perf_counter()
                    if dim and not in_reason:
                        sys.stdout.write("\033[2m"); in_reason = True
                    if not dim and in_reason:
                        sys.stdout.write("\033[0m\n"); in_reason = False
                    sys.stdout.write(txt); sys.stdout.flush()
            finish = ch.get("finish_reason") or finish
        stats = obj.get("tensorfold") or stats
        usage = obj.get("usage") or usage
if in_reason:
    sys.stdout.write("\033[0m")
print()
ttft = (first - t0) if first else None
print(f"\n[finish {finish}; prompt {usage.get('prompt_tokens')} / completion {usage.get('completion_tokens')} tokens; "
      f"TTFT {ttft:.2f} s; server decode {stats.get('tok_s')} tok/s; tokens/round {stats.get('tokens_per_round')}; "
      f"mode {stats.get('mtp_mode')} d{stats.get('depth')} c{stats.get('confidence')}]" if ttft is not None else
      f"\n[finish {finish}; no tokens]", file=sys.stderr)
PY
