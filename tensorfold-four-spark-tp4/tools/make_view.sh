#!/usr/bin/env bash
# Build the serve view on this Spark: a directory of symlinks to every file of the EXL3 pack, plus
#   - lm_head.safetensors -> the original BF16 lm_head (tools/fetch_lm_head.py), and a model.safetensors.index.json
#     that maps lm_head.weight to it (TensorFold's glm_moe_dsa engine reads a BF16 lm_head.weight; the pack's own
#     head is 8-bit EXL3 and stays in the index, unused);
#   - chat_template.jinja = the official template with the thinking-off line fixed (tools/fix_chat_template.py).
# The pack directory is never written. Idempotent; refuses on any sha mismatch.
#   bash tensorfold-four-spark-tp4/tools/make_view.sh
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/env.sh"
PY="$VENV/bin/python"; [[ -x "$PY" ]] || PY="$PYTHON_BIN"

[[ -f "$PACK_DIR/config.json" && -f "$PACK_DIR/model.safetensors.index.json" ]] || die "no pack at $PACK_DIR (setup.sh downloads it)"
[[ "$(sha256sum "$PACK_DIR/config.json" | cut -d' ' -f1)" == "$PACK_CONFIG_SHA" ]] \
  || die "$PACK_DIR/config.json is not $PACK_REPO @ ${PACK_REV:0:7}"
[[ "$(stat -L -c %s "$HEAD_DIR/lm_head.safetensors" 2>/dev/null)" == "$HEAD_BYTES" ]] \
  || die "no BF16 lm_head at $HEAD_DIR. Run: $PY $RECIPE_TP4_DIR/tools/fetch_lm_head.py $HEAD_DIR"

mkdir -p "$VIEW_DIR"
find "$VIEW_DIR" -mindepth 1 -maxdepth 1 -type l -delete
n=0
for src in "$PACK_DIR"/* "$PACK_DIR"/.[!.]*; do
  [[ -e "$src" || -L "$src" ]] || continue
  name="$(basename "$src")"
  case "$name" in model.safetensors.index.json|chat_template.jinja) continue ;; esac
  ln -sfn "$src" "$VIEW_DIR/$name"; n=$((n + 1))
done
ln -sfn "$HEAD_DIR/lm_head.safetensors" "$VIEW_DIR/lm_head.safetensors"
"$PY" - "$PACK_DIR/model.safetensors.index.json" "$VIEW_DIR/model.safetensors.index.json" <<'PY'
import json, os, sys
src, dst = sys.argv[1], sys.argv[2]
idx = json.load(open(src))
idx["weight_map"]["lm_head.weight"] = "lm_head.safetensors"
tmp = dst + ".tmp"
with open(tmp, "w") as f:
    json.dump(idx, f, indent=2)
os.replace(tmp, dst)
PY
"$PY" "$RECIPE_TP4_DIR/tools/fix_chat_template.py" "$PACK_DIR/chat_template.jinja" "$VIEW_DIR/chat_template.jinja" >/dev/null
say "view $VIEW_DIR: $n pack symlinks + BF16 lm_head + fixed chat template (sha ${FIXED_TEMPLATE_SHA:0:16})"
