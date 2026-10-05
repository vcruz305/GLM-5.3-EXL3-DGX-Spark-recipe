#!/usr/bin/env bash
# GLM-5.3 (SAGE MixedK EXL3, 3.38 bpw) on four DGX Sparks, TensorFold TP=4: set up and run the whole cluster from ONE
# machine: one of the Sparks (`./glm53 init` there finds the other three by itself), or any Linux / macOS box with ssh
# to all four (`init --via SPARK` or `init --hosts`). The repo root's ./glm53 calls this file.
# It drives setup.sh (per Spark, over ssh), serve.sh (the driver) and rank.sh (per Spark); those still work on their
# own. `./glm53 help` prints the usage below.
set -uo pipefail

usage() {
  cat <<'EOF'
Usage: ./glm53 <command> [options]

First time, on any one of the four Sparks (it becomes rank 0 and serves the API):
  ./glm53 init                                      finds the other three Sparks on the ConnectX-7 fabric by itself,
                                                    checks ssh to them, writes the hosts file: no host names needed
  HF_TOKEN=hf_... ./glm53 setup --download-once     installs on all four in parallel, one pack download copied over
                                                    the fabric; re-runnable, downloads resume
  ./glm53 up [--profile NAME]                       preflight, then all four ranks (~8 min) until READY
Then:
  ./glm53 chat "Explain RoCE in two sentences."     one streamed request on rank 0, with the engine's stats
  ./glm53 smoke | bench                             /v1 checks | the 6 reference prompts (ids + tok/s)
  ./glm53 status | logs | tunnel [--open] | down

Commands:
  init       run on a Spark: finds the other three on its ConnectX-7 fabric port (ip neigh, avahi _ssh._tcp, a ping
             sweep of a small subnet), keeps those that answer ssh -o BatchMode=yes with a GB10 GPU and their own
             /etc/machine-id, needs exactly three, ranks them by fabric IP (this Spark = rank 0) and writes the hosts
             file with fabric IPs as ssh targets. On a machine that is not a Spark it stops and says what to do.
             [--via SPARK]          from a machine that is not a Spark: run the discovery on SPARK (any ssh target of
                                    one of the four); it becomes rank 0 and ranks 1-3 are reached through it
             [--hosts H0,H1,H2,H3]  skip discovery: the four ssh targets in rank order ('local' = this machine)
               [--fabric-ips A,B,C,D] skip address detection   [--peer-ssh P1,P2,P3] how rank 0 reaches ranks 1-3
             [--force]              replace an existing hosts file
             FABRIC_IFNAME=<port> ./glm53 init   use that ConnectX-7 port instead of detecting it
  setup      [--download-once]  download the 319 GB pack once on rank 0 and rsync it to ranks 1-3 over the fabric
             [--model-dir DIR]  the pack already sits at DIR (on every Spark, or on rank 0 with --download-once)
             [--no-int4]        skip the second TensorFold tree (then int4-262k needs another setup)
             HF_TOKEN=hf_... ./glm53 setup   installs your Hugging Face token on the Sparks that download (gated pack)
  sync       copy this recipe clone to every Spark (rsync over ssh; RECIPE_SYNC=git clones and checks out this commit)
  check      setup.sh --check on all four: the profile's runtime pin and the serve view
  preflight  recipe commit, idle GPU, memory, ports, RoCE GID, runtime, view and launcher dry run on all four
  up         [--profile fast-160k|int4-262k|dcp4-262k] [--allow-unvalidated] [--skip-preflight]
  status | logs | smoke | bench | fadvise | down        the serve.sh step of the same name, on all four / rank 0
  chat       "prompt"   (THINKING=0, DRAFT=0, TEMPERATURE=0, MAX_TOKENS=N as in chat.sh)
  tunnel     the ssh tunnel to rank 0's API (port 8890): from another driver it prints it (--open runs it, Ctrl-C
             closes it); on rank 0 itself it prints the command to run on your laptop

Every command takes:
  --dry-run              print every ssh / rsync command instead of running it (no ssh at all; init prints each
                         discovery command)
  --profile NAME         fast-160k   163,840 tokens, bf16 KV cache: the measured fast path (default)
                         int4-262k   262,144 tokens, int4 KV cache (glm53-kv-int4 tree); quality gate not finished,
                                     so it also needs --allow-unvalidated
                         dcp4-262k   262,144 tokens, bf16 KV cache split across ranks: measured, slower
                         smoke / bench / status / chat / down reuse the profile of the last `up`.

Files (git-ignored): tensorfold-four-spark-tp4/hosts (rank ssh_target fabric_ip [peer_ssh]), tensorfold-four-spark-tp4/
cluster.env (MODEL_DIR, FABRIC_IFNAME, ROCE_HCA, ...: forwarded to every Spark), tensorfold-four-spark-tp4/ssh_config
(init --via only), logs in tensorfold-four-spark-tp4/runs/.
Settings: env.sh (FABRIC_IFNAME, ROCE_HCA, SSH_CONFIG, REMOTE_REPO, RECIPE_HOME, TFS_*).
EOF
}

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TP4=tensorfold-four-spark-tp4
fail() { echo "error: $*" >&2; exit 1; }

# ---- arguments ------------------------------------------------------------------------------------------------------
CMD="${1:-help}"; [[ $# -gt 0 ]] && shift
OPT_DRY=0 OPT_FORCE=0 OPT_ONCE=0 OPT_SKIP_PRE=0 OPT_OPEN=0
OPT_HOSTS="" OPT_IPS="" OPT_PEERS="" OPT_MODEL_DIR="" OPT_VIA="" PROFILE_FROM=""
ARGS=()
need_val() { [[ -n "${2:-}" && "${2:0:2}" != -- ]] || fail "$1 needs a value (./glm53 help)"; }
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) OPT_DRY=1 ;;
    --profile) need_val "$@"; export PROFILE="$2"; PROFILE_FROM=flag; shift ;;
    --profile=*) export PROFILE="${1#*=}"; PROFILE_FROM=flag ;;
    --allow-unvalidated) export ALLOW_UNVALIDATED=1 ;;
    --hosts) need_val "$@"; OPT_HOSTS="$2"; shift ;;
    --hosts=*) OPT_HOSTS="${1#*=}" ;;
    --fabric-ips) need_val "$@"; OPT_IPS="$2"; shift ;;
    --fabric-ips=*) OPT_IPS="${1#*=}" ;;
    --peer-ssh) need_val "$@"; OPT_PEERS="$2"; shift ;;
    --peer-ssh=*) OPT_PEERS="${1#*=}" ;;
    --via) need_val "$@"; OPT_VIA="$2"; shift ;;
    --via=*) OPT_VIA="${1#*=}" ;;
    --model-dir) need_val "$@"; OPT_MODEL_DIR="$2"; shift ;;
    --model-dir=*) OPT_MODEL_DIR="${1#*=}" ;;
    --download-once) OPT_ONCE=1 ;;
    --no-int4) export WITH_INT4=0 ;;
    --force) OPT_FORCE=1 ;;
    --skip-preflight) OPT_SKIP_PRE=1 ;;
    --open) OPT_OPEN=1 ;;
    -h|--help) CMD=help ;;
    --) shift; ARGS+=("$@"); break ;;
    -*) fail "unknown option $1 (./glm53 help)" ;;
    *) ARGS+=("$1") ;;
  esac
  shift
done
case "$CMD" in help|-h|--help) usage; exit 0 ;; esac
case "$CMD" in
  init|setup|sync|check|preflight|up|status|logs|smoke|bench|fadvise|down|chat|tunnel) ;;
  *) usage >&2; fail "unknown command '$CMD'" ;;
esac
DRY=0; [[ "$OPT_DRY" == 1 || "${DRY_RUN:-0}" != 0 ]] && DRY=1
[[ "$DRY" == 1 ]] && export DRY_RUN=1
if [[ -n "$OPT_MODEL_DIR" ]]; then
  [[ "$OPT_MODEL_DIR" == /* || "$OPT_MODEL_DIR" == "~/"* ]] \
    || fail "--model-dir needs an absolute path (or '~/...' in quotes: the home directory on each Spark)"
  export MODEL_DIR="${OPT_MODEL_DIR%/}"
fi

# ---- per-cluster state: cluster.env and the profile of the last `up` (an exported variable wins over both) -------------
CLUSTER_ENV="$HERE/cluster.env"
SERVING_ENV="$HERE/runs/serving.env"
load_kv() {                                      # load_kv FILE: export each KEY=value line whose KEY is not set yet
  local f=$1 l
  [[ -f "$f" ]] || return 0
  while IFS= read -r l || [[ -n "$l" ]]; do
    [[ "$l" =~ ^([A-Z_][A-Z0-9_]*)= ]] || continue
    [[ -n "${!BASH_REMATCH[1]+x}" ]] || eval "export $l"
  done < "$f"
}
case "$CMD" in
  smoke|bench|status|logs|chat|tunnel|down|fadvise)
    if [[ -z "${PROFILE+x}" && -f "$SERVING_ENV" ]]; then load_kv "$SERVING_ENV"; PROFILE_FROM=last-up; fi ;;
esac
USER_IFNAME="${FABRIC_IFNAME:-}"                 # a port you named yourself (init detects one otherwise)
load_kv "$CLUSTER_ENV"

# ---- what reaches the Sparks: only variables you set (or cluster.env set), captured before env.sh fills in defaults ---
FORWARD_VARS="PROFILE ALLOW_UNVALIDATED RECIPE_HOME VENV TF_REF TF_SRC_DIR B12X_STAGE STATE_DIR MODEL_ROOT MODEL_DIR PACK_DIR
DRAFTER_DIR HEAD_DIR VIEW_DIR TILES CUDA_HOME FABRIC_IFNAME ROCE_HCA TFS_ROCE TFS_CONTEXT TF_GLM53_DCP
TF_GLM53_CACHE_RESERVE_GB TFS_KV_DTYPE TFS_PROMPT_ROWS TFS_VERIFY_ROWS TFS_DFLASH_DEPTH TFS_DFLASH_CONFIDENCE
TFS_DRAFT_DEFAULT TFS_HEALTH TFS_WARMUP TFS_THINKING TFS_MAX_TOKENS TFS_NAME TFS_ALIAS TFS_HTTP_HOST TFS_HTTP_PORT
TFS_API_KEY_FILE TFS_MASTER_PORT TFS_MIN_START_GIB TFS_MIN_AVAIL_GIB TFS_MAX_SWAP_GROWTH_KB TFS_WATCH_GRACE_S
TF_GLM53_KV_RCP TF_GLM53_KV_TILE TORCH_EXTENSIONS_DIR TRITON_CACHE_DIR BENCH_IDS BENCH_ARGS
WITH_INT4 SKIP_DOWNLOADS HF_MAX_WORKERS HF_HUB_DISABLE_XET HF_HOME HF_ENDPOINT PYTHON_BIN TORCH_SPEC TORCH_INDEX_URL
B12X_TARBALL_SHA TF_REPO"
FWD=""
for k in $FORWARD_VARS; do
  if [[ -n "${!k+x}" ]]; then FWD+=" $k=$(printf %q "${!k}")"; fi
done
export SERVE_FWD="$FWD"                          # serve.sh forwards exactly this list (not env.sh's own exports)

# shellcheck source=env.sh
source "$HERE/env.sh"
if [[ -z "${REMOTE_REPO:-}" ]]; then             # the clone on each Spark: same path below the home directory as here
  case "$RECIPE_ROOT" in
    "$HOME"/*) REMOTE_REPO="~/${RECIPE_ROOT#"$HOME"/}" ;;
    *) REMOTE_REPO="$RECIPE_ROOT" ;;
  esac
fi
export REMOTE_REPO
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=20)
SSHP="ssh"                                       # the same, as typed in a fix command
if [[ -n "${SSH_CONFIG:-}" ]]; then SSH+=(-F "$SSH_CONFIG"); SSHP="ssh -F $(printf %q "$SSH_CONFIG")"; fi
ssh_e() {                                        # the ssh command for rsync -e (rsync splits it; quotes keep spaces)
  local a out=""
  for a in "${SSH[@]}"; do if [[ "$a" == *" "* ]]; then out+=" '$a'"; else out+=" $a"; fi; done
  printf '%s\n' "${out# }"
}

# ---- helpers ----------------------------------------------------------------------------------------------------------
qdir() { if [[ "$1" == "~/"* ]]; then printf '~/%q' "${1#\~/}"; else printf %q "$1"; fi; }
local_path() { if [[ "$1" == "~/"* ]]; then printf '%s\n' "$HOME/${1#\~/}"; else printf '%s\n' "$1"; fi; }
hrun() {                                         # hrun TARGET CMD: run CMD on a host (stdin passes through)
  local t=$1; shift
  if [[ "$DRY" == 1 ]]; then echo "DRY [$t]: $*"; return 0; fi
  if [[ "$t" == local ]]; then bash -c "$*"; else "${SSH[@]}" "$t" "$*"; fi
}
rcmd() {                                         # rcmd RANK CMD: run CMD in the recipe clone on that rank's Spark
  local r=$1; shift
  local cmd; cmd="cd $(qdir "$REMOTE_REPO") && env$FWD HOSTS_INLINE=$(printf %q "$HOSTS_INLINE") $*"
  if [[ "$DRY" == 1 ]]; then echo "DRY [rank $r] ${H_SSH[$r]}: $cmd"; return 0; fi
  if [[ "${H_SSH[$r]}" == local ]]; then bash -c "$cmd"; else "${SSH[@]}" "${H_SSH[$r]}" "$cmd"; fi
}
on() {                                           # on TARGET CMD: how a user types CMD for that host (fix lines)
  if [[ "$1" == local ]]; then printf '%s\n' "$2"; else printf "%s %s '%s'\n" "$SSHP" "$1" "$2"; fi
}
need_hosts() {
  [[ -f "$HOSTS_FILE" ]] || fail "no hosts file yet. Run ./glm53 init on one of the four Sparks (it finds the other three), then ./glm53 $CMD there"
  read_hosts
}
gate_profile() {
  if [[ "${PROFILE_VALIDATED:-1}" != 1 && "${ALLOW_UNVALIDATED:-0}" != 1 ]]; then
    fail "profile $PROFILE is not validated yet (${PROFILE_NOTE:-no measured figures}). To run it anyway: ./glm53 $CMD --profile $PROFILE --allow-unvalidated"
  fi
}
set_cluster() {                                  # set_cluster KEY VALUE: record a per-cluster setting in cluster.env
  local k=$1 v=$2 tmp
  if [[ "$DRY" == 1 ]]; then echo "DRY: cluster.env: $k=$(printf %q "$v")"; return 0; fi
  tmp="$(mktemp "$CLUSTER_ENV.XXXXXX")" || fail "cannot write next to $CLUSTER_ENV"
  {
    if [[ -f "$CLUSTER_ENV" ]]; then grep -v "^$k=" "$CLUSTER_ENV"
    else echo "# per-cluster settings for ./glm53 and serve.sh (git-ignored); every Spark gets them. KEY=value, shell-quoted."; fi
    printf '%s=%q\n' "$k" "$v"
  } > "$tmp"
  mv "$tmp" "$CLUSTER_ENV"
}

FIXES=()
fix() { FIXES+=("$*"); }
print_fixes() {
  [[ ${#FIXES[@]} -gt 0 ]] || return 0
  echo >&2
  echo "To fix:" >&2
  printf '%s\n' "${FIXES[@]}" | awk '!seen[$0]++' | sed 's/^/  /' >&2
  FIXES=()
}
hint_from() {                                    # hint_from FILE: queue the fix for every known failure in FILE
  local f=$1
  [[ -f "$f" ]] || return 0
  g() { grep -a -q -E "$1" "$f"; }
  g 'no Hugging Face token' && fix "the pack is gated: request access on https://huggingface.co/$PACK_REPO, then HF_TOKEN=hf_... ./glm53 setup (installs the token on the Sparks that download)"
  g 'Python\.h' && fix "on that Spark: sudo apt install -y libpython3.12-dev   (then ./glm53 setup)"
  g 'libibverbs' && fix "on that Spark: sudo apt install -y gcc libibverbs-dev   (then ./glm53 setup; or TFS_ROCE=0 for NCCL reductions)"
  g 'nvcc not found' && fix "CUDA_HOME=<a CUDA 13.x toolkit, e.g. /usr/local/cuda-13.0> ./glm53 setup"
  g 'recipe clone at' && fix "./glm53 sync   (every Spark must run this same recipe commit)"
  g 'the recipe pins|lacks the measured loader fixes|no TensorFold checkout|no venv at|not the recipe runtime|diff vs PR #159|has no kvq\.py' \
    && fix "./glm53 setup   (re-runnable: installs both pinned TensorFold trees; nothing is downloaded twice)"
  g 'no b12x RoCE module' && fix "./glm53 setup (needs gcc + libibverbs-dev), or TFS_ROCE=0 ./glm53 up for NCCL reductions (measured 37.07 vs 41.42 tok/s)"
  g 'no RoCE v2 GID|NCCL_IB_HCA=' && fix "the hosts file's fabric_ip is not on FABRIC_IFNAME=$FABRIC_IFNAME, or the RDMA device is not ROCE_HCA=$ROCE_HCA: ./glm53 init --force (on rank 0) re-detects both"
  g 'GPU BUSY|GPU busy' && fix "another job holds a GPU: stop it (nvidia-smi on that Spark), or ./glm53 down if it is this server"
  g 'a server rank already runs|PORT [0-9]+ in use' && fix "a server is already up: ./glm53 status, then ./glm53 down"
  g 'LOW MEMORY|MemAvailable [0-9]+ GiB <' && fix "./glm53 fadvise (drops the model files' page cache, which GB10 counts as used), and stop other jobs"
  g 'GiB is free|caches a rank' && fix "./glm53 fadvise, stop other jobs; never lower TF_GLM53_CACHE_RESERVE_GB"
  g "cannot 'ssh -o BatchMode=yes" && fix "./glm53 init --force on rank 0   (prints the commands that let rank 0 ssh to ranks 1-3)"
  g 'watchdog flagged|VIOLATION' && fix "a watchdog stopped the cluster (memory floor or swap growth): ./glm53 logs, then ./glm53 down before the next up"
  g 'not validated yet|NOT VALIDATED' && grep -a -q 'error: .*not validated' "$f" && fix "add --allow-unvalidated to run that profile anyway, or use the default ($DEFAULT_PROFILE)"
  g 'pack download failed|drafter download failed' && fix "./glm53 setup again (downloads resume); if transfers stall: HF_HUB_DISABLE_XET=1 ./glm53 setup"
  g 'b12x tarball sha256' && fix "GitHub served another archive for b12x b58f34e: inspect it, then B12X_TARBALL_SHA=<sha> ./glm53 setup"
  g 'weight shards|truncated|is incomplete|no pack at|shards present' && fix "./glm53 setup (resumes the download), or ./glm53 setup --download-once to copy rank 0's pack again"
  g 'no serve view|not the fixed template|does not map lm_head|lm_head\.safetensors missing|no DFlash2 drafter' && fix "./glm53 setup   (rebuilds the serve view)"
  g 'START FAILED|TIMEOUT|FATAL' && fix "./glm53 logs (read rank*.log for [tf_serve] FATAL and the watchdog logs), then ./glm53 down and ./glm53 preflight"
  g 'Permission denied \(|Host key verification failed|Could not resolve hostname|Connection timed out|Connection refused|No route to host' \
    && fix "ssh failed: ./glm53 init --force (on rank 0) checks every ssh path and prints the commands that fix it"
  g 'rsync: (command )?not found|rsync: not found|command not found: rsync' && fix "sudo apt install -y rsync on that Spark (or RECIPE_SYNC=git ./glm53 sync)"
  unset -f g
  return 0
}
last_line() {                                    # the last meaningful line of a log (progress bars use \r)
  tail -c 4000 "$1" 2>/dev/null | tr '\r' '\n' | grep -a -v -E '^[[:space:]]*$' | tail -n 1 | cut -c1-150
}

# par_run DIR LABEL "RANK:CMD"...: CMD in the recipe clone on each rank, all at once; per-rank logs, progress, summary.
par_run() {
  local dir=$1 label=$2; shift 2
  local item r c i n=0 alive t0 bad=0
  local pids=() ranks=() logs=()
  [[ "$DRY" == 1 ]] || mkdir -p "$dir"
  for item in "$@"; do
    r="${item%%:*}"; c="${item#*:}"
    if [[ "$DRY" == 1 ]]; then rcmd "$r" "$c"; continue; fi
    logs[$n]="$dir/$label-rank$r.log"; ranks[$n]=$r
    rcmd "$r" "$c" > "${logs[$n]}" 2>&1 < /dev/null &
    pids[$n]=$!
    say "rank $r (${H_SSH[$r]}): $label started, log ${logs[$n]}"
    n=$((n + 1))
  done
  [[ "$DRY" == 1 ]] && return 0
  t0=$(date +%s)
  while :; do
    alive=0
    for ((i = 0; i < n; i++)); do kill -0 "${pids[$i]}" 2>/dev/null && alive=1; done
    [[ "$alive" == 0 ]] && break
    sleep 5
    if (( ($(date +%s) - t0) % ${PROGRESS_EVERY_S:-60} < 5 )); then
      for ((i = 0; i < n; i++)); do
        kill -0 "${pids[$i]}" 2>/dev/null && echo "  [$label +$(( $(date +%s) - t0 ))s] rank ${ranks[$i]}: $(last_line "${logs[$i]}")" >&2
      done
    fi
  done
  echo >&2
  for ((i = 0; i < n; i++)); do
    wait "${pids[$i]}"; local rc=$?
    r=${ranks[$i]}
    if [[ $rc == 0 ]]; then
      echo "OK    rank $r (${H_SSH[$r]}): $(last_line "${logs[$i]}")" >&2
    else
      bad=1
      local why; why="$(grep -a -E '^error: |REFUSE|FAILED' "${logs[$i]}" | tail -n 1 | cut -c1-200)"
      echo "FAIL  rank $r (${H_SSH[$r]}), exit $rc: ${why:-$(last_line "${logs[$i]}")}" >&2
      echo "      log: ${logs[$i]}" >&2
      hint_from "${logs[$i]}"
    fi
  done
  print_fixes
  return $bad
}

# run_serve STEP: serve.sh STEP, output kept in runs/<step>-<time>.log, the fix printed on failure
run_serve() {
  local step=$1 log rc
  if [[ "$DRY" == 1 ]]; then DRY_RUN=1 bash "$HERE/serve.sh" "$step"; return $?; fi
  mkdir -p "$HERE/runs"; log="$HERE/runs/$step-$(date +%Y%m%d_%H%M%S).log"
  bash "$HERE/serve.sh" "$step" 2>&1 | tee "$log"; rc=${PIPESTATUS[0]}
  if [[ $rc != 0 ]]; then hint_from "$log"; print_fixes; echo "(output kept in $log)" >&2; fi
  return $rc
}

# ---- init ---------------------------------------------------------------------------------------------------------
# Two ways in. `./glm53 init` (the default) runs ON one of the Sparks, which becomes rank 0: tools/discover.sh finds the
# other three on the ConnectX-7 fabric and checks each over ssh; nobody has to know a host name. `--via SPARK` runs the
# same discovery on SPARK from a machine that is not a Spark. `--hosts H0,H1,H2,H3` skips discovery (the override).
DISCOVER="$HERE/tools/discover.sh"
VIA_CFG="$HERE/ssh_config"                       # written by init --via: ranks 1-3 reached through rank 0 (ProxyJump)
NV_SWITCH_PLAYBOOK=https://build.nvidia.com/spark/multi-sparks-through-switch
NV_DISCOVER_URL=https://raw.githubusercontent.com/NVIDIA/dgx-spark-playbooks/refs/heads/main/nvidia/connect-two-sparks/assets/discover-sparks
RECIPE_URL=https://github.com/vcruz305/GLM-5.3-EXL3-DGX-Spark-recipe.git
probe() {                                        # probe TARGET: KEY=VALUE facts about a Spark (discover.sh on stdin)
  local t=$1
  if [[ "$t" == local ]]; then bash "$DISCOVER" probe "$FABRIC_IFNAME"
  else "${SSH[@]}" "$t" bash -s -- probe "$(printf %q "$FABRIC_IFNAME")" < "$DISCOVER"; fi
}
kv() { printf '%s\n' "$1" | sed -n "s/^$2=//p" | head -n 1; }
fld() { printf '%s\n' "$1" | tr '\t' '\n' | sed -n "s/^$2=//p" | head -n 1; }   # a field of a discover.sh record
pick_port() {                                    # pick_port CX7: Up with an IPv4 address, enp1s0f* before enP2p1s0f*
  local e i s h a
  for e in $1; do
    IFS=: read -r i s h a <<< "$e"
    [[ "$s" == Up && -n "$a" ]] || continue
    if [[ "$i" == enP* ]]; then echo "2 $i"; else echo "1 $i"; fi
  done | sort -k1,1n -k2,2 | head -n 1 | cut -d' ' -f2
}
port_get() {                                     # port_get CX7 IFNAME N: field N of that port (3 = RDMA device, 4 = ip)
  local e i s h a
  for e in $1; do
    IFS=: read -r i s h a <<< "$e"
    [[ "$i" == "$2" ]] || continue
    case "$3" in 3) echo "$h" ;; 4) echo "${a%/*}" ;; esac
    return 0
  done
}
ssh_fix() {                                      # ssh_fix TARGET OUTPUT: the fix for a failed ssh from here
  local t=$1 out=$2
  case "$out" in
    *"Host key verification failed"*) fix "ssh -o StrictHostKeyChecking=accept-new $t true   # accept $t's host key once (ssh-keygen -R $t first if it changed)" ;;
    *"Permission denied"*) fix "ssh-copy-id $t   # passwordless ssh from here to $t (ssh-keygen -t ed25519 first if you have no key)" ;;
    *"Could not resolve hostname"*) fix "$t does not resolve: use the Spark's mDNS name (<hostname>.local) or its IP, as user@host; see NVIDIA's https://build.nvidia.com/spark/connect-to-your-spark" ;;
    *) fix "$SSHP $t true   # must succeed without a password prompt (is $t up and reachable?)" ;;
  esac
}
nv_ssh_fix() {                                   # nv_ssh_fix RANK0_TARGET PEER...: let rank 0 ssh to the peers
  local t0=$1 where; shift
  if [[ "$t0" == local ]]; then where="on this Spark"; else where="on $t0 (rank 0)"; fi
  fix "passwordless ssh from rank 0 to the other Sparks is missing. Once, $where, as your normal user (not sudo), in a terminal: it asks for your password on each Spark, so the user types it, never an agent:"
  fix "    curl -fsSLO $NV_DISCOVER_URL && bash ./discover-sparks"
  fix "  (NVIDIA's discover-sparks from the Connect Two Sparks / Multi Sparks Through a Switch playbooks: it finds the Sparks over avahi on the ConnectX-7 ports, needs avahi-utils (sudo apt install -y avahi-utils), and puts one shared key, ~/.ssh/id_ed25519_shared, on all of them)"
  if [[ $# == 1 ]]; then fix "  or by hand, $where: ssh-copy-id $1   (ssh-keygen -t ed25519 first if ~/.ssh has no key)"
  else fix "  or by hand, $where: for ip in $*; do ssh-copy-id \$ip; done   (ssh-keygen -t ed25519 first if ~/.ssh has no key; one password per Spark)"; fi
}
peer_fix() {                                     # peer_fix PEER_TARGET PEER_ADDR OUTPUT: let rank 0 ssh to that peer
  local tr=$1 p=$2 out=$3 t0=${T[0]}
  case "$out" in
    *"Host key verification failed"*)
      fix "$(on "$t0" "ssh-keygen -R $p; ssh -o StrictHostKeyChecking=accept-new $p true")   # rank 0 accepts $p's (new) host key" ;;
    *) nv_ssh_fix "$t0" "$p" ;;
  esac
}
split4() {                                       # split4 LIST N: comma list -> SPLIT array of exactly N entries
  local IFS=,; SPLIT=($1)
  [[ ${#SPLIT[@]} == "$2" ]] || fail "'$1' must list exactly $2 comma-separated entries"
}
not_a_spark() {                                  # init without --hosts / --via on a machine that is not a Spark
  echo >&2
  echo "This machine ($(hostname)) is not a DGX Spark, so it cannot see the Sparks' ConnectX-7 fabric. Either:" >&2
  echo "  1. run init on any one of the four Sparks (it becomes rank 0 and serves the API):" >&2
  echo "       ssh <user>@<spark-hostname>.local     (its mDNS name, or its IP; NVIDIA Sync can also open a terminal on it)" >&2
  echo "       git clone $RECIPE_URL && cd GLM-5.3-EXL3-DGX-Spark-recipe && ./glm53 init" >&2
  echo "  2. or from here, through any one Spark you can ssh to:   ./glm53 init --via <that Spark's ssh target>" >&2
  echo "  3. or name all four yourself, rank 0 first:              ./glm53 init --hosts H0,H1,H2,H3" >&2
  fail "init needs a Spark: nothing was written"
}
write_hosts() {                                  # write_hosts HOW: the hosts file from T, IPS, PEERCOL
  local body r
  body="# written by ./glm53 init on $(date +%Y-%m-%d) ($1; FABRIC_IFNAME=$FABRIC_IFNAME, ROCE_HCA=$ROCE_HCA)
# rank  ssh_target  fabric_ip  [peer_ssh: how rank 0's watchdog reaches this rank; default the fabric IP]"
  for r in 0 1 2 3; do body+=$'\n'"$r  ${T[$r]}  ${IPS[$r]}${PEERCOL[$r]:+  ${PEERCOL[$r]}}"; done
  if [[ "$DRY" == 1 ]]; then printf '%s\n' "$body" | sed 's/^/  DRY hosts: /'
  else printf '%s\n' "$body" > "$HOSTS_FILE"; printf '%s\n' "$body" | sed 's/^/  /' >&2; fi
}
unset_cluster() {                                # unset_cluster KEY: drop a setting from cluster.env
  [[ "$DRY" == 1 || ! -f "$CLUSTER_ENV" ]] && return 0
  grep -q "^$1=" "$CLUSTER_ENV" || return 0
  local tmp; tmp="$(mktemp "$CLUSTER_ENV.XXXXXX")" || return 0
  grep -v "^$1=" "$CLUSTER_ENV" > "$tmp"; mv "$tmp" "$CLUSTER_ENV"
}
write_via_config() {                             # the ssh config that reaches ranks 1-3 through rank 0 ($OPT_VIA)
  local base="$HOME/.ssh/config" r body
  [[ -n "${SSH_CONFIG:-}" && "$SSH_CONFIG" != "$VIA_CFG" ]] && base="$SSH_CONFIG"   # your own -F file, if you use one
  body="# written by ./glm53 init --via $OPT_VIA on $(date +%Y-%m-%d): ranks 1-3 are reached through rank 0 ($OPT_VIA) over
# the ConnectX-7 fabric (ProxyJump). ./glm53 and serve.sh use this file (SSH_CONFIG in cluster.env); it includes your own
# ssh config for everything else. Delete it, and SSH_CONFIG from cluster.env, to stop using it."
  for r in 1 2 3; do body+=$'\n'"Host glm53-rank$r"$'\n'"  HostName ${IPS[$r]}"$'\n'"  ProxyJump $OPT_VIA"; done
  body+=$'\n'"Host *"$'\n'"  Include $base"$'\n'"  Include /etc/ssh/ssh_config"
  if [[ "$DRY" == 1 ]]; then printf '%s\n' "$body" | sed "s|^|  DRY $VIA_CFG: |"; else printf '%s\n' "$body" > "$VIA_CFG"; fi
}

init_discover() {
  [[ -z "$OPT_IPS$OPT_PEERS" ]] || fail "--fabric-ips and --peer-ssh go with --hosts; ./glm53 init alone discovers both"
  if [[ -f "$HOSTS_FILE" && "$OPT_FORCE" != 1 && "$DRY" != 1 ]]; then
    fail "$HOSTS_FILE already exists. Re-run with --force to replace it"
  fi
  local via="$OPT_VIA" where="this machine" r
  [[ -n "$via" ]] && where="$via"
  T=(local "" "" ""); IPS=("" "" "" ""); PEERCOL=("" "" "" "")
  [[ -n "$via" ]] && T[0]="$via"
  say "1/3 discovering the four Sparks on the ConnectX-7 fabric from $where (rank 0: it serves the API)"
  if [[ "$DRY" == 1 ]]; then
    [[ -n "$via" ]] && echo "DRY [here]: ${SSH[*]} $via bash -s -- discover $USER_IFNAME  < $TP4/tools/discover.sh   (the steps below run on $via)"
    FABRIC_IFNAME= DRY_RUN=1 bash "$DISCOVER" discover "$USER_IFNAME"
    echo "DRY: keep the candidates whose GPU is a GB10, one per /etc/machine-id; exactly three are required; ranks 1-3 in fabric-IP order"
    IPS=("<fabric IP of ${via:-this Spark}>" "<lowest peer fabric IP>" "<second peer fabric IP>" "<highest peer fabric IP>")
    for r in 1 2 3; do if [[ -n "$via" ]]; then T[$r]="glm53-rank$r"; PEERCOL[$r]="${IPS[$r]}"; else T[$r]="${IPS[$r]}"; fi; done
    say "2/3 hosts file $HOSTS_FILE"
    FABRIC_IFNAME="<the fabric port>" ROCE_HCA="<its RDMA device>" write_hosts "discovered"
    set_cluster FABRIC_IFNAME "<the fabric port>"
    set_cluster ROCE_HCA "<its RDMA device>"
    if [[ -n "$via" ]]; then
      write_via_config; set_cluster SSH_CONFIG "$VIA_CFG"
      say "3/3 ssh from here to ranks 1-3 through $via"
      for r in 1 2 3; do echo "DRY [here]: ssh -F $VIA_CFG -o BatchMode=yes -o ConnectTimeout=20 glm53-rank$r true"; done
    else
      say "3/3 rank 0 is this Spark: discovery already checked ssh -o BatchMode=yes to ranks 1-3"
    fi
    return 0
  fi

  local out
  if [[ -z "$via" ]]; then
    out="$(FABRIC_IFNAME= bash "$DISCOVER" discover "$USER_IFNAME")"   # only a port you named, not cluster.env's
  else
    out="$("${SSH[@]}" "$via" true 2>&1)" || {
      echo "  ssh $via failed: $(printf '%s\n' "$out" | grep -v '^$' | tail -n 1)" >&2
      ssh_fix "$via" "$out"; print_fixes; fail "cannot reach $via over ssh without a password: nothing was written"; }
    out="$("${SSH[@]}" "$via" bash -s -- discover "$(printf %q "$USER_IFNAME")" < "$DISCOVER")"
  fi
  local line S="" peers=() fails=() others=() errs=() tab=$'\t' swept=1 p
  while IFS= read -r line; do
    case "${line%%"$tab"*}" in
      SELF) S="$line" ;; PEER) peers+=("$line") ;; FAIL) fails+=("$line") ;; OTHER) others+=("$line") ;;
      NOTE) echo "  note: $(fld "$line" msg)" >&2; [[ "$line" == *"no ping sweep"* ]] && swept=0 ;;
      ERROR) errs+=("$line") ;;
    esac
  done <<< "$out"
  # peers in fabric-IP order (rank 1 = the lowest address)
  mapfile -t peers < <(for p in ${peers[@]+"${peers[@]}"}; do printf '%s\t%s\n' "$(fld "$p" ip)" "$p"; done \
                       | sort -t. -k1,1n -k2,2n -k3,3n -k4,4n | cut -f2-)
  if [[ ${#errs[@]} -gt 0 ]]; then
    local e="${errs[0]}"
    echo "  $(fld "$e" msg)" >&2
    case "$(fld "$e" code)" in
      NOT_SPARK)
        [[ -z "$via" ]] && not_a_spark
        fix "--via must name one of the four Sparks (an ssh target of a GB10 machine); $via is not one" ;;
      NO_CX7) fix "this does not look like a DGX Spark (no ConnectX-7 port): run ./glm53 init on one of the four" ;;
      NO_UP) fix "cable the same QSFP port of all four Sparks to one switch (four Sparks need a switch), then check that ibdev2netdev shows it (Up): $NV_SWITCH_PLAYBOOK" ;;
      NO_IPV4) fix "give that ConnectX-7 port an IPv4 address on all four Sparks: NVIDIA Sync's Cluster Assistant (https://docs.nvidia.com/sync/latest/cluster-assistant.html) or step 4 of $NV_SWITCH_PLAYBOOK; FABRIC_IFNAME=<port> ./glm53 init picks another port" ;;
      TOO_MANY|SAME_MID) fix "name the four Sparks yourself, rank 0 first: ./glm53 init --hosts local,<ip>,<ip>,<ip> --force" ;;
    esac
    print_fixes; fail "init stopped before writing $HOSTS_FILE"
  fi
  [[ -n "$S" ]] || fail "discovery printed nothing usable (above): nothing was written"
  local IF SIP SHCA ip
  IF="$(fld "$S" if)"; SIP="$(fld "$S" ip)"; SHCA="$(fld "$S" hca)"
  [[ -n "$via" ]] || where="this Spark"
  echo "  rank 0: $(fld "$S" host) ($where), $(fld "$S" gpu), fabric $SIP/$(fld "$S" prefix) on $IF, RDMA device ${SHCA:-none}" >&2
  for p in ${peers[@]+"${peers[@]}"}; do
    echo "  found:  $(fld "$p" host) $(fld "$p" ip), $(fld "$p" gpu), RDMA device $(fld "$p" hca), ssh $(fld "$p" ssh) as $(fld "$p" user): ok" >&2
  done
  for p in ${others[@]+"${others[@]}"}; do echo "  not a Spark (not used): $(fld "$p" ip) $(fld "$p" host), GPU '$(fld "$p" gpu)'" >&2; done
  for p in ${fails[@]+"${fails[@]}"}; do echo "  on the fabric, ssh -o BatchMode=yes failed: $(fld "$p" ip): $(fld "$p" why)" >&2; done
  local n=${#peers[@]}
  if (( n > 3 )); then
    local all=""; for p in ${peers[@]+"${peers[@]}"}; do all+="${all:+,}$(fld "$p" ip)"; done
    fix "this recipe runs on exactly four Sparks and $((n + 1)) answered ($SIP + $all). Choose four, rank 0 first: ./glm53 init --hosts local,<ip>,<ip>,<ip> --force"
    print_fixes; fail "init stopped before writing $HOSTS_FILE: found $n other Sparks, need exactly 3"
  fi
  if (( n < 3 )); then
    local denied=() why explained=0 nossh=()
    for p in ${fails[@]+"${fails[@]}"}; do
      ip="$(fld "$p" ip)"; why="$(fld "$p" why)"
      case "$why" in
        *"Permission denied"*) denied+=("$ip"); explained=1 ;;
        *"Host key verification failed"*) explained=1
          fix "$(on "${T[0]}" "ssh-keygen -R $ip")   # $ip's host key changed (a reinstalled Spark?); init then accepts the new one" ;;
        *"no IPv4 address on"*|*"is outside"*) explained=1; fix "$why: $NV_SWITCH_PLAYBOOK" ;;
        *) nossh+=("$ip ($why)") ;;
      esac
    done
    [[ ${#denied[@]} -gt 0 ]] && nv_ssh_fix "${T[0]}" ${denied[@]+"${denied[@]}"}
    if [[ "$explained" == 0 ]]; then
      fix "only $n of the other three Sparks answered on $IF ($SIP/$(fld "$S" prefix)). Check that all four are on, cabled to the switch on the same ConnectX-7 port, and that 'ip -br addr' on each shows an address on $IF in that subnet ($NV_SWITCH_PLAYBOOK)"
      [[ ${#nossh[@]} == 0 ]] || fix "on the fabric without ssh (a switch, or a Spark whose sshd is down): ${nossh[*]+${nossh[*]}}"
      [[ "$swept" == 1 ]] || fix "this subnet is too large to ping-sweep (a 169.254.x.x/16 link-local fabric): install avahi-utils (sudo apt install -y avahi-utils) so init can use mDNS, or ping each Spark's fabric IP once from here, then run init again"
      fix "or name them: ./glm53 init --hosts local,<ip>,<ip>,<ip> --force"
    fi
    fix "then: ./glm53 init --force"
    print_fixes; fail "init stopped before writing $HOSTS_FILE: found $n of the other three Sparks"
  fi
  [[ ${#fails[@]} == 0 ]] || echo "  note: the addresses above that failed ssh are not used (a switch, or a fifth machine on the fabric)" >&2

  # ranks 1-3 by fabric IP
  local hcas=("$SHCA")
  IPS[0]="$SIP"
  for r in 1 2 3; do
    p="${peers[$((r - 1))]}"
    IPS[$r]="$(fld "$p" ip)"; hcas[$r]="$(fld "$p" hca)"
    if [[ -n "$via" ]]; then T[$r]="glm53-rank$r"; [[ "$(fld "$p" ssh)" != "${IPS[$r]}" ]] && PEERCOL[$r]="$(fld "$p" ssh)"
    else T[$r]="$(fld "$p" ssh)"; [[ "${T[$r]}" != "${IPS[$r]}" ]] && PEERCOL[$r]="${T[$r]}"; fi
    [[ "$(fld "$p" rsync)" == 1 ]] || fix "rank $r (${IPS[$r]}) has no rsync: sudo apt install -y rsync there (./glm53 sync and setup --download-once use it)"
    [[ -n "${hcas[$r]}" ]] || echo "  rank $r: warning: no RDMA device behind $IF: RoCE reductions will refuse (TFS_ROCE=0 runs NCCL)" >&2
  done
  FABRIC_IFNAME="$IF"
  if [[ -n "$SHCA" && "${hcas[1]}" == "$SHCA" && "${hcas[2]}" == "$SHCA" && "${hcas[3]}" == "$SHCA" ]]; then
    ROCE_HCA="$SHCA"
  elif [[ -n "$SHCA" ]]; then
    echo "  warning: the RDMA device behind $IF differs between the Sparks (${hcas[*]}); recording rank 0's ($SHCA)" >&2
    ROCE_HCA="$SHCA"
  fi
  say "2/3 hosts file $HOSTS_FILE (ranks 1-3 in fabric-IP order)"
  write_hosts "discovered from $(fld "$S" host)"
  set_cluster FABRIC_IFNAME "$FABRIC_IFNAME"
  set_cluster ROCE_HCA "$ROCE_HCA"
  if [[ -z "$via" ]]; then
    say "3/3 rank 0 is this Spark: discovery already checked ssh -o BatchMode=yes to ranks 1-3"
    if [[ "${SSH_CONFIG:-}" == "$VIA_CFG" ]]; then unset_cluster SSH_CONFIG; rm -f "$VIA_CFG"; fi
    print_fixes
    say "init OK. Run every ./glm53 command on this Spark. Next: ./glm53 setup --download-once   (one 319 GB download, copied over the fabric)"
    return 0
  fi
  write_via_config
  set_cluster SSH_CONFIG "$VIA_CFG"
  say "3/3 ssh from here to ranks 1-3 through $via (ProxyJump: needs your key on every Spark, not only on $via)"
  local bad=0
  for r in 1 2 3; do
    out="$(ssh -F "$VIA_CFG" -o BatchMode=yes -o ConnectTimeout=20 "${T[$r]}" true 2>&1)"
    if [[ $? == 0 ]]; then echo "  here -> rank $r (${IPS[$r]} via $via): ok" >&2
    else
      bad=1; echo "  here -> rank $r (${IPS[$r]} via $via): FAILED: $(printf '%s\n' "$out" | grep -v '^$' | tail -n 1)" >&2
      fix "ssh-copy-id -o ProxyJump=$via ${IPS[$r]}   # your key from this machine on rank $r (it asks for your password there once)"
    fi
  done
  print_fixes
  [[ "$bad" == 0 ]] || fail "the hosts file is written, but this machine cannot reach every Spark through $via yet: run the commands above, then ./glm53 init --via $via --force"
  say "init OK. Drive the cluster from here (or from $via). Next: ./glm53 setup --download-once"
}

init_named() {
  [[ -z "$OPT_VIA" ]] || fail "--via and --hosts do not go together: --hosts names all four, --via discovers them"
  split4 "$OPT_HOSTS" 4; T=("${SPLIT[@]}")
  local r s
  for r in 0 1 2; do for ((s = r + 1; s < 4; s++)); do
    [[ "${T[$r]}" != "${T[$s]}" ]] || fail "--hosts lists ${T[$r]} twice"
  done; done
  IPS=("" "" "" ""); PEERCOL=("" "" "" "")
  local HOSTN=("" "" "" "") HCAS=("" "" "" "") MIDS=("" "" "" "") OUTS=("" "" "" "")
  if [[ -n "$OPT_IPS" ]]; then split4 "$OPT_IPS" 4; IPS=("${SPLIT[@]}"); fi
  if [[ -n "$OPT_PEERS" ]]; then split4 "$OPT_PEERS" 3; PEERCOL=("" "${SPLIT[@]}"); fi
  if [[ -f "$HOSTS_FILE" && "$OPT_FORCE" != 1 && "$DRY" != 1 ]]; then
    fail "$HOSTS_FILE already exists. Re-run with --force to replace it"
  fi
  local bad=0 out ip hca gpu cx port
  say "1/3 ssh from here to each Spark: its GPU, /etc/machine-id and ConnectX-7 ports"
  for r in 0 1 2 3; do
    if [[ "$DRY" == 1 ]]; then
      if [[ "${T[$r]}" == local ]]; then echo "DRY [local]: bash $TP4/tools/discover.sh probe $FABRIC_IFNAME   (hostname, /etc/machine-id, GPU, ConnectX-7 ports and addresses, rsync)"
      else echo "DRY [${T[$r]}]: ${SSH[*]} ${T[$r]} bash -s -- probe $FABRIC_IFNAME  < $TP4/tools/discover.sh   (hostname, /etc/machine-id, GPU, ConnectX-7 ports and addresses, rsync)"; fi
      [[ -n "${IPS[$r]}" ]] || IPS[$r]="<fabric IP of ${T[$r]}>"
      continue
    fi
    out="$(probe "${T[$r]}" 2>&1)"
    if [[ $? != 0 || "$out" != *HOST=* ]]; then
      bad=1; echo "  rank $r: ssh to ${T[$r]} failed: $(printf '%s\n' "$out" | grep -v '^$' | tail -n 1)" >&2
      ssh_fix "${T[$r]}" "$out"; continue
    fi
    OUTS[$r]="$out"
  done
  # the fabric port: FABRIC_IFNAME if you set it, else the port rank 0 reports Up with an IPv4 address (as discovery does)
  if [[ "$DRY" != 1 && -z "$USER_IFNAME" && -n "${OUTS[0]}" ]]; then
    port="$(pick_port "$(kv "${OUTS[0]}" CX7)")"
    if [[ -n "$port" && "$port" != "$FABRIC_IFNAME" ]]; then
      echo "  fabric port: $port (Up with an IPv4 address on rank 0; FABRIC_IFNAME=<port> picks another)" >&2
      FABRIC_IFNAME="$port"
    fi
  fi
  for r in 0 1 2 3; do
    [[ "$DRY" == 1 || -z "${OUTS[$r]}" ]] && continue
    out="${OUTS[$r]}"; cx="$(kv "$out" CX7)"
    HOSTN[$r]="$(kv "$out" HOST)"; MIDS[$r]="$(kv "$out" MID)"; gpu="$(kv "$out" GPU)"
    if [[ -n "$cx" ]]; then ip="$(port_get "$cx" "$FABRIC_IFNAME" 4)"; hca="$(port_get "$cx" "$FABRIC_IFNAME" 3)"
    else ip="$(kv "$out" IP)"; hca="$(kv "$out" HCA)"; fi
    HCAS[$r]="$hca"
    if [[ "$gpu" != *GB10* ]]; then
      bad=1; echo "  rank $r: ${T[$r]} (${HOSTN[$r]}) is not a DGX Spark: GPU '${gpu:-none}'" >&2
      fix "--hosts must name four DGX Sparks (GB10); replace ${T[$r]}"; continue
    fi
    if [[ -n "${IPS[$r]}" ]]; then
      [[ -z "$ip" || "$ip" == "${IPS[$r]}" ]] || echo "  rank $r: note: ${T[$r]} has $ip on $FABRIC_IFNAME; using --fabric-ips ${IPS[$r]}" >&2
    elif [[ -n "$ip" ]]; then
      IPS[$r]="$ip"
    else
      bad=1
      echo "  rank $r: ${T[$r]} (${HOSTN[$r]}) has no IPv4 address on FABRIC_IFNAME=$FABRIC_IFNAME. Its ConnectX-7 ports: ${cx:-none}" >&2
      fix "FABRIC_IFNAME=<the ConnectX-7 port that carries the fabric> ./glm53 init --hosts $OPT_HOSTS --force   (or --fabric-ips A,B,C,D)"
      continue
    fi
    echo "  rank $r: ${T[$r]} (${HOSTN[$r]}) $gpu, fabric ${IPS[$r]} on $FABRIC_IFNAME, RDMA device ${hca:-none}, $(kv "$out" ARCH)" >&2
    [[ -n "$hca" ]] || echo "  rank $r: warning: no RDMA device behind $FABRIC_IFNAME on ${T[$r]}: RoCE reductions will refuse (TFS_ROCE=0 runs NCCL)" >&2
    [[ "$(kv "$out" RSYNC)" == 1 ]] || fix "$(on "${T[$r]}" 'sudo apt install -y rsync')   # ./glm53 sync / setup --download-once use it"
  done
  if [[ "$DRY" != 1 ]]; then
    for r in 0 1 2; do for ((s = r + 1; s < 4; s++)); do
      [[ -z "${IPS[$r]}" || "${IPS[$r]}" != "${IPS[$s]}" ]] || { bad=1; echo "  ranks $r and $s share the fabric IP ${IPS[$r]}" >&2; }
      [[ -z "${MIDS[$r]}" || "${MIDS[$r]}" != "${MIDS[$s]}" ]] \
        || { bad=1; echo "  ranks $r and $s are the same machine (/etc/machine-id ${MIDS[$r]:0:12}...): --hosts needs four different Sparks" >&2; }
    done; done
    # one RDMA device name for all four (the recipe sets one ROCE_HCA); adopt the detected one if it differs
    if [[ -n "${HCAS[0]}" && "${HCAS[0]}" == "${HCAS[1]}" && "${HCAS[0]}" == "${HCAS[2]}" && "${HCAS[0]}" == "${HCAS[3]}" \
          && "${HCAS[0]}" != "$ROCE_HCA" ]]; then
      echo "  RDMA device behind $FABRIC_IFNAME is ${HCAS[0]} on all four (ROCE_HCA was $ROCE_HCA): recording ROCE_HCA=${HCAS[0]}" >&2
      ROCE_HCA="${HCAS[0]}"
    fi
  fi
  if [[ "$bad" == 1 ]]; then print_fixes; fail "init stopped before writing $HOSTS_FILE: fix the above, then run it again"; fi

  say "2/3 hosts file $HOSTS_FILE"
  write_hosts "--hosts"
  set_cluster FABRIC_IFNAME "$FABRIC_IFNAME"
  set_cluster ROCE_HCA "$ROCE_HCA"
  [[ -n "${SSH_CONFIG:-}" ]] && set_cluster SSH_CONFIG "$SSH_CONFIG"

  say "3/3 ssh from rank 0 (${T[0]}) to ranks 1-3: its watchdog stops all four through it, and --download-once copies over it"
  local p
  for r in 1 2 3; do
    p="${PEERCOL[$r]:-${IPS[$r]}}"
    if [[ "$DRY" == 1 ]]; then hrun "${T[0]}" "ssh -o BatchMode=yes -o ConnectTimeout=5 $p true"; continue; fi
    out="$(hrun "${T[0]}" "ssh -o BatchMode=yes -o ConnectTimeout=5 $(printf %q "$p") true" 2>&1)"
    if [[ $? == 0 ]]; then echo "  rank 0 -> rank $r ($p): ok" >&2
    else bad=1; echo "  rank 0 -> rank $r ($p): FAILED: $(printf '%s\n' "$out" | grep -v '^$' | tail -n 1)" >&2; peer_fix "${T[$r]}" "$p" "$out"; fi
  done
  [[ "$DRY" == 1 ]] && return 0
  if [[ "$bad" == 1 ]]; then print_fixes; fail "the hosts file is written, but rank 0 cannot reach every peer yet: run the commands above, then ./glm53 init --hosts $OPT_HOSTS --force"; fi
  print_fixes
  say "init OK. Next: ./glm53 setup --download-once   (one 319 GB download, copied over the fabric)"
}

cmd_init() {
  if [[ -n "$OPT_HOSTS" ]]; then init_named; else init_discover; fi
}

# ---- sync: this clone onto every Spark --------------------------------------------------------------------------------
reach_all() {                                    # every Spark answers ssh from here (fail fast, with the fix)
  [[ "$DRY" == 1 ]] && return 0
  local r out bad=0
  for r in 0 1 2 3; do
    [[ "${H_SSH[$r]}" == local ]] && continue
    out="$("${SSH[@]}" "${H_SSH[$r]}" true 2>&1)" || { bad=1; echo "rank $r: ssh ${H_SSH[$r]} failed: $out" >&2; ssh_fix "${H_SSH[$r]}" "$out"; }
  done
  [[ "$bad" == 0 ]] || { print_fixes; return 1; }
}
cmd_sync() {
  local mode="${RECIPE_SYNC:-rsync}" r t dest sha url what bad=0
  sha="$(git -C "$RECIPE_ROOT" rev-parse HEAD 2>/dev/null || true)"
  what="${sha:0:12}"; [[ -n "$sha" ]] || what="this tree (not a git clone)"
  say "recipe $what -> $REMOTE_REPO on every Spark ($mode)"
  if [[ "$mode" == rsync && "$DRY" != 1 ]]; then
    command -v rsync >/dev/null || fail "rsync not found here: install it (sudo apt install -y rsync / brew install rsync), or RECIPE_SYNC=git ./glm53 sync"
  fi
  for r in 0 1 2 3; do
    t="${H_SSH[$r]}"
    if [[ "$t" == local && "$(local_path "$REMOTE_REPO")" == "$RECIPE_ROOT" ]]; then echo "  rank $r: this clone" >&2; continue; fi
    case "$mode" in
      rsync)
        dest="${REMOTE_REPO#\~/}"
        local a=(rsync -a --delete "--exclude=/$TP4/runs/" --exclude=__pycache__/)
        if [[ "$t" == local ]]; then
          a+=("$RECIPE_ROOT/" "$(local_path "$REMOTE_REPO")/")
          if [[ "$DRY" == 1 ]]; then echo "DRY [local]: mkdir -p $(local_path "$REMOTE_REPO") && ${a[*]}"; continue; fi
          mkdir -p "$(local_path "$REMOTE_REPO")" && "${a[@]}" || { bad=1; fix "rsync -a $RECIPE_ROOT/ $(local_path "$REMOTE_REPO")/"; }
        else
          a+=(-e "$(ssh_e)" "--rsync-path=mkdir -p $(printf %q "$dest") && rsync" "$RECIPE_ROOT/" "$t:$dest/")
          if [[ "$DRY" == 1 ]]; then echo "DRY [$t]: ${a[*]}"; continue; fi
          "${a[@]}" || { bad=1; echo "  rank $r: rsync to $t failed" >&2; fix "$(on "$t" 'sudo apt install -y rsync')   # if rsync is missing there"; }
        fi ;;
      git)
        url="$(git -C "$RECIPE_ROOT" remote get-url origin 2>/dev/null)" || fail "RECIPE_SYNC=git: this clone has no origin remote"
        [[ -z "$(git -C "$RECIPE_ROOT" status --porcelain --untracked-files=no 2>/dev/null)" ]] \
          || echo "  warning: local edits here are not copied by RECIPE_SYNC=git (commit and push them, or use rsync)" >&2
        local q; q="$(qdir "$REMOTE_REPO")"
        hrun "$t" "if [ -d $q/.git ]; then git -C $q fetch -q origin; else git clone -q $(printf %q "$url") $q; fi && git -C $q checkout -q --detach $sha" \
          || { bad=1; fix "push commit ${sha:0:12} to $url (the Sparks fetch it from there), or RECIPE_SYNC=rsync ./glm53 sync"; } ;;
      *) fail "RECIPE_SYNC=$mode: rsync or git" ;;
    esac
    [[ "$DRY" == 1 ]] || echo "  rank $r: $t:$REMOTE_REPO <- $what" >&2
  done
  print_fixes
  return $bad
}

# ---- setup ------------------------------------------------------------------------------------------------------------
install_token() {                                # HF_TOKEN from the driver -> the token file on the Sparks that download
  [[ -n "${HF_TOKEN:-}" ]] || return 0
  local r t
  for r in "$@"; do
    t="${H_SSH[$r]}"
    if [[ "$DRY" == 1 ]]; then echo "DRY [$t]: write \$HF_TOKEN (from stdin, never argv) to \${HF_HOME:-~/.cache/huggingface}/token unless one exists"; continue; fi
    printf '%s' "$HF_TOKEN" | hrun "$t" 'umask 077; d="${HF_HOME:-$HOME/.cache/huggingface}"; mkdir -p "$d"; if [ -s "$d/token" ]; then cat >/dev/null; echo "token already present"; else cat > "$d/token"; echo "token installed"; fi' \
      | sed "s/^/  rank $r ($t): Hugging Face /" >&2
  done
}
model_paths() {                                  # model_paths RANK: PACK_DIR DRAFTER_DIR HEAD_DIR as that Spark resolves them
  if [[ "$DRY" == 1 ]]; then printf '<PACK_DIR on rank %s>\n<DRAFTER_DIR on rank %s>\n<HEAD_DIR on rank %s>\n' "$1" "$1" "$1"; return 0; fi
  rcmd "$1" "bash $TP4/rank.sh paths --rank $1"
}
copy_models() {                                  # rank 0's pack, drafter and lm_head -> ranks 1-3, rsync over the fabric
  local dir=$1 r p i n=0 alive bad=0
  local src=() dst=() pids=() ranks=() logs=()
  local IFS_=$IFS
  IFS=$'\n'
  src=($(model_paths 0))
  IFS=$IFS_
  [[ ${#src[@]} == 3 ]] || fail "could not read the model paths on rank 0 (./glm53 sync first?)"
  if [[ "$DRY" != 1 ]]; then
    hrun "${H_SSH[0]}" 'command -v rsync >/dev/null' || fail "rsync missing on rank 0: $(on "${H_SSH[0]}" 'sudo apt install -y rsync')"
  fi
  say "copying the pack, drafter and lm_head from rank 0 to ranks 1-3 over the fabric (rsync; resumes where it stopped)"
  for r in 1 2 3; do
    IFS=$'\n'
    dst=($(model_paths "$r"))
    IFS=$IFS_
    [[ ${#dst[@]} == 3 ]] || fail "could not read the model paths on rank $r"
    p="${H_PEER[$r]}"
    local cmd="" k
    for k in 0 1 2; do
      cmd+="${cmd:+ && }rsync -a --partial-dir=.rsync-partial --exclude=.cache/ --info=progress2 -e 'ssh -o BatchMode=yes'"
      cmd+=" $(printf %q "--rsync-path=mkdir -p $(printf %q "${dst[$k]}") && rsync") $(printf %q "${src[$k]}/") $(printf %q "$p:${dst[$k]}/")"
    done
    if [[ "$DRY" == 1 ]]; then hrun "${H_SSH[0]}" "$cmd"; continue; fi
    mkdir -p "$dir"; logs[$n]="$dir/copy-rank$r.log"; ranks[$n]=$r
    hrun "${H_SSH[0]}" "$cmd" > "${logs[$n]}" 2>&1 < /dev/null &
    pids[$n]=$!; n=$((n + 1))
    say "rank 0 -> rank $r ($p): started, log $dir/copy-rank$r.log"
  done
  [[ "$DRY" == 1 ]] && return 0
  local t0; t0=$(date +%s)
  while :; do
    alive=0
    for ((i = 0; i < n; i++)); do kill -0 "${pids[$i]}" 2>/dev/null && alive=1; done
    [[ "$alive" == 0 ]] && break
    sleep 5
    if (( ($(date +%s) - t0) % ${PROGRESS_EVERY_S:-60} < 5 )); then
      for ((i = 0; i < n; i++)); do
        kill -0 "${pids[$i]}" 2>/dev/null && echo "  [copy +$(( $(date +%s) - t0 ))s] rank ${ranks[$i]}: $(last_line "${logs[$i]}")" >&2
      done
    fi
  done
  for ((i = 0; i < n; i++)); do
    wait "${pids[$i]}" || { bad=1; echo "FAIL  copy to rank ${ranks[$i]}: $(last_line "${logs[$i]}") (log ${logs[$i]})" >&2; hint_from "${logs[$i]}"; }
  done
  if [[ "$bad" == 1 ]]; then
    fix "rank 0 must ssh to ranks 1-3 without a password: ./glm53 init --force prints how; then ./glm53 setup --download-once again (rsync resumes)"
    print_fixes; return 1
  fi
  say "copy done"
}
cmd_setup() {
  need_hosts
  if [[ -n "$OPT_MODEL_DIR" ]]; then set_cluster MODEL_DIR "$MODEL_DIR"; fi   # every later step needs the same path
  reach_all || exit 1
  cmd_sync || fail "could not copy the recipe to every Spark (above)"
  local dir; dir="$HERE/runs/setup-$(date +%Y%m%d_%H%M%S)"
  local S="bash $TP4/setup.sh"
  if [[ "$OPT_ONCE" == 1 ]]; then
    install_token 0
    say "setup: rank 0 in full (downloads), ranks 1-3 runtime only, in parallel"
    par_run "$dir" setup "0:$S" "1:$S --runtime-only" "2:$S --runtime-only" "3:$S --runtime-only" \
      || fail "setup failed (above); fix it and run ./glm53 setup --download-once again (finished steps are kept)"
    copy_models "$dir" || exit 1
    say "setup: ranks 1-3 serve view + checks on the copied models"
    par_run "$dir" view "1:SKIP_DOWNLOADS=1 $S" "2:SKIP_DOWNLOADS=1 $S" "3:SKIP_DOWNLOADS=1 $S" \
      || fail "setup failed on a copied rank (above)"
  else
    install_token 0 1 2 3
    say "setup on all four in parallel (each downloads; --download-once downloads only on rank 0)"
    par_run "$dir" setup "0:$S" "1:$S" "2:$S" "3:$S" \
      || fail "setup failed (above); fix it and run ./glm53 setup again (finished steps and downloads are kept)"
  fi
  [[ "$DRY" == 1 ]] && return 0
  say "setup OK on all four. Next: ./glm53 up"
}

# ---- serving ----------------------------------------------------------------------------------------------------------
remember_profile() {
  [[ "$DRY" == 1 ]] && return 0
  mkdir -p "$HERE/runs"
  { printf 'PROFILE=%q\n' "$PROFILE"; [[ "${ALLOW_UNVALIDATED:-0}" == 1 ]] && echo "ALLOW_UNVALIDATED=1"; } > "$SERVING_ENV"
}
cmd_up() {
  need_hosts
  gate_profile
  if [[ "$OPT_SKIP_PRE" != 1 ]]; then
    say "preflight on all four (profile $PROFILE)"
    run_serve preflight || fail "preflight failed: fix the above, then ./glm53 up again"
  fi
  remember_profile
  run_serve up || fail "the start failed: ./glm53 logs, then ./glm53 down before the next up"
  [[ "$DRY" == 1 ]] && return 0
  cat >&2 <<EOF

GLM-5.3 is up (profile $PROFILE): http://127.0.0.1:$TFS_HTTP_PORT/v1 on rank 0 (${H_SSH[0]}), model id $TFS_NAME
  ./glm53 chat "Explain RoCE in two sentences."     one request, streamed, with the engine's stats
  ./glm53 smoke                                     /v1 checks;  ./glm53 bench: the 6 reference prompts
  ./glm53 tunnel                                    how to reach it from another machine (ssh -L)
  ./glm53 down                                      stop all four
EOF
}
cmd_chat() {
  need_hosts
  [[ ${#ARGS[@]} -gt 0 ]] || fail "chat needs a prompt: ./glm53 chat \"Explain RoCE in two sentences.\""
  local envs="" k a q=""
  for k in THINKING DRAFT TEMPERATURE MAX_TOKENS; do [[ -n "${!k+x}" ]] && envs+="$k=$(printf %q "${!k}") "; done
  for a in "${ARGS[@]}"; do q+=" $(printf %q "$a")"; done
  rcmd 0 "${envs}bash $TP4/chat.sh$q"
}
cmd_tunnel() {
  need_hosts
  if [[ "${H_SSH[0]}" == local ]]; then
    local lan; lan="$(ip -4 route get 1.1.1.1 2>/dev/null | sed -n 's/.* src \([0-9.]*\).*/\1/p' | head -n 1)"
    [[ -n "$lan" ]] || lan="<this Spark's IP>"
    echo "rank 0 is this machine: the API is http://127.0.0.1:$TFS_HTTP_PORT/v1 here. From a laptop on the same network:"
    echo "  ssh -N -L $TFS_HTTP_PORT:127.0.0.1:$TFS_HTTP_PORT $(id -un)@$(hostname).local    # mDNS name; or $(id -un)@$lan"
    echo "  then http://127.0.0.1:$TFS_HTTP_PORT/v1 on the laptop (Ctrl-C closes the tunnel)"
    return 0
  fi
  local c=("${SSH[@]}" -N -o ExitOnForwardFailure=yes -L "$TFS_HTTP_PORT:127.0.0.1:$TFS_HTTP_PORT" "${H_SSH[0]}")
  if [[ "$OPT_OPEN" == 1 && "$DRY" != 1 ]]; then
    say "tunnel open: http://127.0.0.1:$TFS_HTTP_PORT/v1 here -> rank 0 (${H_SSH[0]}); Ctrl-C closes it"
    exec "${c[@]}"
  fi
  echo "${c[*]}    # then http://127.0.0.1:$TFS_HTTP_PORT/v1 here"
}

[[ -n "$PROFILE_FROM" && "$PROFILE_FROM" == last-up ]] && say "profile $PROFILE (from the last ./glm53 up)"
case "$CMD" in
  init) cmd_init ;;
  setup) cmd_setup ;;
  sync) need_hosts; cmd_sync ;;
  check) need_hosts; par_run "$HERE/runs/check-$(date +%Y%m%d_%H%M%S)" check \
           "0:bash $TP4/setup.sh --check" "1:bash $TP4/setup.sh --check" "2:bash $TP4/setup.sh --check" "3:bash $TP4/setup.sh --check" ;;
  preflight) need_hosts; gate_profile; run_serve preflight ;;
  up) cmd_up ;;
  status|logs|smoke|bench|fadvise) need_hosts; run_serve "$CMD" ;;
  down) need_hosts; run_serve down && { [[ "$DRY" == 1 ]] || rm -f "$SERVING_ENV"; } ;;
  chat) cmd_chat ;;
  tunnel) cmd_tunnel ;;
esac
