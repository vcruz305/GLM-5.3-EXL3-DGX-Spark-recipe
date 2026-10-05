#!/usr/bin/env bash
# Find the four DGX Sparks of this recipe on the ConnectX-7 fabric, from one of them. `./glm53 init` runs it on rank 0
# (this Spark, or over ssh on the Spark that --via names) and decides from its output; it also probes the Sparks that
# `./glm53 init --hosts` names. It changes nothing on any machine, except that the first ssh to a peer records the
# peer's host key (StrictHostKeyChecking=accept-new, as NVIDIA's discover-sparks script does).
#
#   bash tools/discover.sh discover [IFNAME]   this Spark's fabric port and address, the other Sparks on that subnet,
#                                              and each one's facts over ssh -o BatchMode=yes
#   bash tools/discover.sh probe IFNAME        facts about this machine, KEY=VALUE lines
#
# discover, step by step (the conventions of NVIDIA's DGX Spark clustering playbooks):
#   1. this machine's GPU must be a GB10 (nvidia-smi); otherwise it is not a Spark (ERROR code=NOT_SPARK)
#   2. the ConnectX-7 ports come from ibdev2netdev (else /sys/class/infiniband/*/device/net/*). The fabric port is one
#      that is Up and has an IPv4 address: IFNAME (or FABRIC_IFNAME) if given, else enp1s0f* before enP2p1s0f*, and a
#      static address before a 169.254 link-local one
#   3. candidate peers on that port's subnet: `ip -4 neigh`, avahi-browse _ssh._tcp (what discover-sparks uses; only
#      when avahi-utils is installed) and one ping per address when the subnet has at most DISCOVER_SWEEP_MAX (1024)
#      addresses; at most DISCOVER_MAX_CANDIDATES (64)
#   4. every candidate over ssh -o BatchMode=yes (5 s connect timeout, all at once): /etc/machine-id, GPU name, its
#      address on the same port, the RDMA device behind it. A candidate that refuses the key is retried through any
#      alias in ~/.ssh/config whose HostName is that address. One record per machine-id.
#
# Output (stdout), one record per line: the record type, then TAB-separated KEY=VALUE fields:
#   SELF   if= ip= prefix= hca= host= user= mid= gpu= ports=     this Spark (rank 0)
#   PEER   ip= ssh= host= user= mid= gpu= hca= rsync= arch=      a GB10 that answered ssh; ip = its address on the port
#   OTHER  ip= host= gpu=                                         answered ssh, not a GB10: not used
#   FAIL   ip= why=                                               on the subnet, but ssh -o BatchMode=yes failed
#   NOTE   msg=
#   ERROR  code= msg=                                             discovery could not run (NOT_SPARK, NO_CX7, NO_UP,
#                                                                 NO_IPV4, TOO_MANY, SAME_MID)
# Progress goes to stderr. DRY_RUN=1 prints the commands instead of running anything.

probe_fn() {                                     # probe_fn IFNAME: facts about this machine
  local ifn=$1 d hca=""
  echo "HOST=$(hostname)"
  echo "USER=$(id -un)"
  echo "MID=$(cat /etc/machine-id 2>/dev/null)"
  echo "GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n 1)"
  # the address on IFNAME: a static one before a 169.254 link-local one, else the kernel's order
  echo "IP=$(ip -4 -o addr show dev "$ifn" 2>/dev/null | awk '{split($4, a, "/"); print (a[1] ~ /^169\.254\./ ? 1 : 0), a[1]}' | sort -s -k1,1n | head -n 1 | cut -d' ' -f2)"
  for d in /sys/class/infiniband/*; do [ -e "$d/device/net/$ifn" ] && { hca="${d##*/}"; break; }; done
  [ -n "$hca" ] || hca="$(cx7_ports | awk -v i="$ifn" '$2 == i {print $1; exit}')"
  echo "HCA=$hca"
  echo "IFACES=$(ip -4 -o addr show 2>/dev/null | awk '$2 != "lo" {split($4, a, "/"); printf "%s=%s ", $2, a[1]}')"
  # every ConnectX-7 netdev as port:Up|Down:rdma_device:ip/prefix
  echo "CX7=$(cx7_ports | while read -r d i s; do printf '%s:%s:%s:%s ' "$i" "$s" "$d" "$(first_addr "$i")"; done)"
  echo "RSYNC=$(command -v rsync >/dev/null 2>&1 && echo 1 || echo 0)"
  echo "ARCH=$(uname -m)"
}

clean() { printf '%s' "$1" | tr '\t\n\r' '   '; }
rec() { local t=$1 f out=""; shift; for f in "$@"; do out+=$'\t'"$(clean "$f")"; done; printf '%s%s\n' "$t" "$out"; }
kv() { printf '%s\n' "$1" | sed -n "s/^$2=//p" | head -n 1; }
log() { echo "  [discover] $*" >&2; }
ip2int() { local IFS=.; set -- $1; echo $(( ($1 << 24) + ($2 << 16) + ($3 << 8) + $4 )); }
int2ip() { echo "$(( ($1 >> 24) & 255 )).$(( ($1 >> 16) & 255 )).$(( ($1 >> 8) & 255 )).$(( $1 & 255 ))"; }
is_ipv4() { [[ "$1" =~ ^[0-9]{1,3}(\.[0-9]{1,3}){3}$ ]]; }

cx7_ports() {                                    # "hca ifname Up|Down" per ConnectX-7 netdev
  if command -v ibdev2netdev >/dev/null 2>&1; then
    ibdev2netdev 2>/dev/null | awk '$4 == "==>" {s = $6; gsub(/[()]/, "", s); print $1, $5, s}'
  else
    local d n i st
    for d in /sys/class/infiniband/*; do
      for n in "$d"/device/net/*; do
        [ -e "$n" ] || continue
        i="${n##*/}"; st="$(cat "/sys/class/net/$i/operstate" 2>/dev/null)"
        [ "$st" = up ] && st=Up || st=Down
        echo "${d##*/} $i $st"
      done
    done
  fi
}
first_addr() {                                   # first_addr IFNAME: "ip/prefix", static before 169.254
  ip -4 -o addr show dev "$1" 2>/dev/null | awk '{print ($4 ~ /^169\.254\./ ? 1 : 0), $4}' | sort -s -k1,1n | head -n 1 | cut -d' ' -f2
}

SSHV=(ssh -o BatchMode=yes -o ConnectTimeout=5 -o StrictHostKeyChecking=accept-new)
TIMEOUT_BIN="$(command -v timeout || true)"
ssh_probe() {                                    # ssh_probe TARGET IFNAME: probe_fn over ssh (script on stdin)
  local to=(); [ -n "$TIMEOUT_BIN" ] && to=("$TIMEOUT_BIN" 25)
  ${to[@]+"${to[@]}"} "${SSHV[@]}" "$1" bash -s -- "$2" <<< "$(declare -f probe_fn cx7_ports first_addr); probe_fn \"\$1\""
}
ssh_aliases_for() {                              # ssh_aliases_for IP: ~/.ssh/config aliases whose HostName is IP
  local cfg="$HOME/.ssh/config" files=() inc f a
  [ -f "$cfg" ] || return 0
  files=("$cfg")
  for inc in $(awk 'tolower($1) == "include" {for (i = 2; i <= NF; i++) print $i}' "$cfg"); do
    case "$inc" in "~/"*) inc="$HOME/${inc#\~/}" ;; /*) ;; *) inc="$HOME/.ssh/$inc" ;; esac
    for f in $inc; do [ -f "$f" ] && files+=("$f"); done
  done
  for a in $(awk 'tolower($1) == "host" {for (i = 2; i <= NF; i++) if ($i !~ /[*?!]/) print $i}' "${files[@]}" | sort -u); do
    [ "$(ssh -G "$a" 2>/dev/null | awk '$1 == "hostname" {print $2; exit}')" = "$1" ] && echo "$a"
  done
}

dry_plan() {
  local p="${1:-<port>}"
  cat <<EOF
DRY [rank 0]: nvidia-smi --query-gpu=name --format=csv,noheader     must name a GB10; anything else is not a Spark
DRY [rank 0]: ibdev2netdev                                          the ConnectX-7 ports, Up or Down (else /sys/class/infiniband/*/device/net/*)
DRY [rank 0]: ip -4 -o addr show dev $p                    the fabric port: Up, with an IPv4 address${1:+ (FABRIC_IFNAME=$1)}
DRY [rank 0]: ip -4 neigh show dev $p
DRY [rank 0]: timeout 15 avahi-browse -p -r -f -t _ssh._tcp         only if installed: the IPv4 rows on $p
DRY [rank 0]: ping -c 1 -W 1 -I $p <every address of its subnet>    only if the subnet has <= ${DISCOVER_SWEEP_MAX:-1024} addresses, 128 at a time
DRY [rank 0]: ${SSHV[*]} <candidate> bash -s -- $p  < probe   every candidate at once: /etc/machine-id, GPU, its address on $p, RDMA device
EOF
}

discover_fn() {
  local want="${1:-${FABRIC_IFNAME:-}}"
  local sweep_max="${DISCOVER_SWEEP_MAX:-1024}" max_cand="${DISCOVER_MAX_CANDIDATES:-64}"
  if [ "${DRY_RUN:-0}" != 0 ]; then dry_plan "$want"; return 0; fi

  # 1. this machine
  local self; self="$(probe_fn "${want:-lo}")"
  local gpu; gpu="$(kv "$self" GPU)"
  case "$gpu" in
    *GB10*) ;;
    *) local what="nvidia-smi reports no GPU"; [ -n "$gpu" ] && what="nvidia-smi reports '$gpu', not a GB10"
       rec ERROR code=NOT_SPARK "host=$(kv "$self" HOST)" "gpu=$gpu" "msg=$(kv "$self" HOST) is not a DGX Spark: $what"; return 1 ;;
  esac

  # 2. the fabric port
  local ports; ports="$(cx7_ports)"
  if [ -z "$ports" ]; then
    rec ERROR code=NO_CX7 "msg=no ConnectX-7 port found (ibdev2netdev and /sys/class/infiniband list nothing)"; return 1
  fi
  local hca ifn st addr pick="" summary="" klass
  while read -r hca ifn st; do
    [ -n "$ifn" ] || continue
    addr="$(first_addr "$ifn")"
    summary+="$ifn($st${addr:+ $addr}) "
    [ -n "$want" ] && [ "$ifn" != "$want" ] && continue
    [ "$st" = Up ] && [ -n "$addr" ] || continue
    klass=1; case "$ifn" in enP*) klass=2 ;; esac
    pick+="$klass $ifn $hca $addr"$'\n'
  done <<< "$ports"
  summary="${summary% }"
  pick="$(printf '%s' "$pick" | sort -k1,1n -k2,2 | head -n 1)"
  if [ -z "$pick" ]; then
    if [ -n "$want" ]; then
      rec ERROR code=NO_IPV4 "ports=$summary" "msg=FABRIC_IFNAME=$want is not an Up ConnectX-7 port with an IPv4 address here. Ports: $summary"
    elif ! printf '%s\n' "$ports" | awk '$3 == "Up"' | grep -q .; then
      rec ERROR code=NO_UP "ports=$summary" "msg=no ConnectX-7 port is Up (no QSFP link). Ports: $summary"
    else
      rec ERROR code=NO_IPV4 "ports=$summary" "msg=the ConnectX-7 port that is Up has no IPv4 address yet. Ports: $summary"
    fi
    return 1
  fi
  local IF HCA ADDR SIP PFX
  read -r _ IF HCA ADDR <<< "$pick"
  SIP="${ADDR%/*}"; PFX="${ADDR#*/}"
  [[ "$PFX" =~ ^[0-9]+$ ]] && (( PFX >= 1 && PFX <= 32 )) || PFX=32
  local sint mask net bc size
  sint=$(ip2int "$SIP"); mask=$(( (0xFFFFFFFF << (32 - PFX)) & 0xFFFFFFFF ))
  net=$(( sint & mask )); bc=$(( net | (~mask & 0xFFFFFFFF) )); size=$(( bc - net + 1 ))
  self="$(probe_fn "$IF")"
  rec SELF "if=$IF" "ip=$SIP" "prefix=$PFX" "hca=$(kv "$self" HCA)" "host=$(kv "$self" HOST)" "user=$(kv "$self" USER)" \
    "mid=$(kv "$self" MID)" "gpu=$gpu" "ports=$summary"
  log "this Spark: $(kv "$self" HOST), fabric port $IF ($HCA), $SIP/$PFX; ports: $summary"

  # 3. candidates on the subnet
  local mine; mine=" $(ip -4 -o addr show 2>/dev/null | awk '{split($4, a, "/"); printf "%s ", a[1]}')"
  local raw="" c n=0
  raw+="$(ip -4 neigh show dev "$IF" 2>/dev/null | awk '$0 !~ /FAILED|INCOMPLETE/ {print $1}')"$'\n'
  log "ip neigh on $IF: $(printf '%s' "$raw" | grep -c .) entries"
  if command -v avahi-browse >/dev/null 2>&1; then
    local av; av="$(${TIMEOUT_BIN:+$TIMEOUT_BIN 15} avahi-browse -p -r -f -t _ssh._tcp 2>/dev/null \
      | awk -F';' -v i="$IF" '$1 == "=" && $2 == i && $3 == "IPv4" {print $8}')"
    log "avahi _ssh._tcp on $IF: $(printf '%s' "$av" | grep -c .) addresses"
    raw+="$av"$'\n'
  else
    rec NOTE "msg=avahi-browse not installed (avahi-utils): mDNS not used"
  fi
  if (( size - 2 <= sweep_max && PFX <= 30 )); then
    log "ping sweep of $(int2ip "$net")/$PFX on $IF ($(( size - 2 )) addresses)"
    local a pings
    pings="$(
      for (( a = net + 1; a < bc; a++ )); do
        (( a == sint )) && continue
        { ping -c 1 -W 1 -I "$IF" "$(int2ip "$a")" >/dev/null 2>&1 && int2ip "$a"; } &
        n=$(( n + 1 )); (( n % 128 == 0 )) && wait
      done
      wait
    )"
    raw+="$pings"$'\n'
    raw+="$(ip -4 neigh show dev "$IF" 2>/dev/null | awk '$0 !~ /FAILED|INCOMPLETE/ {print $1}')"$'\n'
  else
    rec NOTE "msg=$(int2ip "$net")/$PFX has $(( size - 2 )) addresses: no ping sweep (DISCOVER_SWEEP_MAX=$sweep_max); candidates come from ip neigh and avahi only"
  fi
  local cands=() ci
  while read -r c; do
    is_ipv4 "$c" || continue
    ci=$(ip2int "$c")
    (( (ci & mask) == net && ci != net && ci != bc )) || continue
    case "$mine" in *" $c "*) continue ;; esac
    cands+=("$c")
  done < <(printf '%s\n' "$raw" | sort -u -t. -k1,1n -k2,2n -k3,3n -k4,4n)
  log "${#cands[@]} candidate(s) on $(int2ip "$net")/$PFX: ${cands[*]:-none}"
  if (( ${#cands[@]} > max_cand )); then
    rec ERROR code=TOO_MANY "msg=${#cands[@]} hosts answer on $(int2ip "$net")/$PFX ($IF), more than DISCOVER_MAX_CANDIDATES=$max_cand"; return 1
  fi
  (( ${#cands[@]} > 0 )) || return 0

  # 4. every candidate over ssh, all at once
  local tmp; tmp="$(mktemp -d)" || { rec ERROR code=TMP "msg=mktemp failed"; return 1; }
  local i
  for i in "${!cands[@]}"; do
    ( ssh_probe "${cands[$i]}" "$IF" > "$tmp/$i.out" 2> "$tmp/$i.err"; echo $? > "$tmp/$i.rc" ) &
  done
  wait
  local -A MID_IPS=() MID_OUT=() MID_SSH=() OK_TARGET=()
  local out err why alias got mid self_mid; self_mid="$(kv "$self" MID)"
  for i in "${!cands[@]}"; do
    c="${cands[$i]}"; out="$(cat "$tmp/$i.out" 2>/dev/null)"; got="$c"
    if [ "$(cat "$tmp/$i.rc" 2>/dev/null)" != 0 ] || [ -z "$(kv "$out" MID)" ]; then
      err="$(grep -v '^[[:space:]]*$' "$tmp/$i.err" 2>/dev/null | tail -n 1)"
      out=""
      case "$err" in
        *"Permission denied"*)
          for alias in $(ssh_aliases_for "$c"); do
            out="$(ssh_probe "$alias" "$IF" 2>/dev/null)" && [ -n "$(kv "$out" MID)" ] && { got="$alias"; break; }
            out=""
          done ;;
      esac
      if [ -z "$out" ]; then rec FAIL "ip=$c" "why=${err:-ssh exited $(cat "$tmp/$i.rc" 2>/dev/null)}"; continue; fi
    fi
    mid="$(kv "$out" MID)"
    [ "$mid" = "$self_mid" ] && continue                      # this Spark through another address
    case "$(kv "$out" GPU)" in
      *GB10*) ;;
      *) [ -n "${MID_OUT[$mid]+x}" ] || rec OTHER "ip=$c" "host=$(kv "$out" HOST)" "gpu=$(kv "$out" GPU)"
         MID_OUT[$mid]="$out"; continue ;;
    esac
    OK_TARGET[$c]="$got"
    if [ -n "${MID_OUT[$mid]+x}" ]; then
      if [ "$(kv "${MID_OUT[$mid]}" IP)" != "$(kv "$out" IP)" ]; then
        rec ERROR code=SAME_MID "msg=$(kv "${MID_OUT[$mid]}" HOST) and $(kv "$out" HOST) report the same /etc/machine-id with different addresses on $IF (two Sparks from one cloned image?)"
        rm -rf "$tmp"; return 1
      fi
      MID_IPS[$mid]+=" $c"
    else
      MID_OUT[$mid]="$out"; MID_IPS[$mid]="$c"; MID_SSH[$mid]="$got"
    fi
  done
  rm -rf "$tmp"
  local fip ssh_t pi
  for mid in "${!MID_IPS[@]}"; do
    out="${MID_OUT[$mid]}"; fip="$(kv "$out" IP)"
    if [ -z "$fip" ]; then
      rec FAIL "ip=${MID_IPS[$mid]%% *}" "why=$(kv "$out" HOST) has no IPv4 address on $IF (cable the same ConnectX-7 port on every Spark)"; continue
    fi
    pi=$(ip2int "$fip")
    if (( (pi & mask) != net )); then
      rec FAIL "ip=$fip" "why=$(kv "$out" HOST)'s $IF address $fip is outside $(int2ip "$net")/$PFX"; continue
    fi
    if [ -n "${OK_TARGET[$fip]+x}" ]; then
      ssh_t="${OK_TARGET[$fip]}"
    elif "${SSHV[@]}" "$fip" true < /dev/null > /dev/null 2>&1; then
      ssh_t="$fip"
    else
      ssh_t="${MID_SSH[$mid]}"
    fi
    rec PEER "ip=$fip" "ssh=$ssh_t" "host=$(kv "$out" HOST)" "user=$(kv "$out" USER)" "mid=$mid" "gpu=$(kv "$out" GPU)" \
      "hca=$(kv "$out" HCA)" "rsync=$(kv "$out" RSYNC)" "arch=$(kv "$out" ARCH)"
  done
  return 0
}

main() {
  case "${1:-}" in
    probe) [ -n "${2:-}" ] || { echo "usage: discover.sh probe IFNAME" >&2; return 2; }; probe_fn "$2" ;;
    discover) shift; discover_fn "$@" ;;
    *) sed -n '2,32p' "${BASH_SOURCE[0]:-/dev/null}" 2>/dev/null; echo "usage: discover.sh discover [IFNAME] | probe IFNAME" >&2; return 2 ;;
  esac
}
# stdin is the script itself under `ssh host bash -s`: nothing below may read it
main "$@" < /dev/null
