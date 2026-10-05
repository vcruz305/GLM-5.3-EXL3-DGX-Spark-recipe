# Source on a Spark (bash) before starting its rank to enable the b12x RoCE one-shot decode reductions. Only sets
# environment variables. rank.sh sources it with FABRIC_IP set to this rank's address from the hosts file.
# Needs: B12X_STAGE (setup.sh staged b12x @ b58f34e there), ROCE_HCA, FABRIC_IFNAME, FABRIC_IP.
# Port of the measured stack's env.sh: same variables and values; the fabric address comes from the hosts file
# instead of a fixed subnet.
: "${B12X_STAGE:?B12X_STAGE}" "${ROCE_HCA:=rocep1s0f0}" "${FABRIC_IFNAME:=enp1s0f0np0}"
# 1. import path (pip --target does not process .pth files, so the cutlass DSL package dir is listed explicitly)
case ":${PYTHONPATH:-}:" in
  *":$B12X_STAGE/site:"*) ;;
  *) export PYTHONPATH="$B12X_STAGE/site:$B12X_STAGE/site/nvidia_cutlass_dsl/dsl_packages${PYTHONPATH:+:$PYTHONPATH}" ;;
esac
# 2. caches inside the stage dir (setup.sh prebuilds the RDMA proxy .so there; CuTe kernels compile on first use)
export B12X_ROCE_CACHE_DIR="$B12X_STAGE/roce_cache"
export B12X_COMPILE_CACHE_DIR="$B12X_STAGE/compile_cache"
# 3. a wait long enough that a rank still JIT-compiling cannot time out a collective. Must be equal on all ranks; a
#    timeout poisons the runtime (the per-round health check then stops every rank).
export B12X_ROCE_SPIN_LIMIT="${B12X_ROCE_SPIN_LIMIT:-300000000}"
export TF_GLM53_ROCE=1
# 4. RDMA device. TensorFold's roce.py hands NCCL_IB_HCA verbatim to b12x, whose proxy matches names with strcmp, so
#    NCCL's exact-match prefix '=' must go ('=rocep1s0f0' silently disables RoCE). The plain name still selects only
#    that device for NCCL (prefix match).
case "${NCCL_IB_HCA:-}" in
  "") export NCCL_IB_HCA="$ROCE_HCA" ;;
  "="*) export NCCL_IB_HCA="${NCCL_IB_HCA#=}" ;;
esac
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-=$FABRIC_IFNAME}"
# roce.py copies NCCL_SOCKET_IFNAME into GLOO_SOCKET_IFNAME only when unset; gloo needs the bare name
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-$FABRIC_IFNAME}"
# 5. GID: the RoCE v2 GID of this Spark's fabric address on that device. The index differs per host (the measured
#    cluster had 5 / 3 / 5 / 9) and can move on reboot or address changes, so it is resolved at every start. The
#    default index 3 can be the IPv6 link-local GID, which mixes GID families across ranks. Only B12X_ROCE_GID_INDEX
#    is set: NCCL keeps its own GID choice (do not set a uniform NCCL_IB_GID_INDEX).
_b12x_ip="${FABRIC_IP:-}"
[ -n "$_b12x_ip" ] || _b12x_ip=$(ip -4 -o addr show dev "$FABRIC_IFNAME" 2>/dev/null | awk '{print $4}' | cut -d/ -f1 | head -1)
unset B12X_ROCE_GID_INDEX
if [ -n "$_b12x_ip" ]; then
  _b12x_hex=$(printf '%02x%02x:%02x%02x' $(echo "$_b12x_ip" | tr . ' '))
  _b12x_p=/sys/class/infiniband/$NCCL_IB_HCA/ports/1
  for _b12x_i in $(ls "$_b12x_p/gids" 2>/dev/null | sort -n); do
    if [ "$(cat "$_b12x_p/gid_attrs/types/$_b12x_i" 2>/dev/null)" = "RoCE v2" ] && grep -q "ffff:$_b12x_hex\$" "$_b12x_p/gids/$_b12x_i"; then
      export B12X_ROCE_GID_INDEX=$_b12x_i; break
    fi
  done
fi
if [ -z "${B12X_ROCE_GID_INDEX:-}" ]; then
  echo "[b12x env] WARNING: no RoCE v2 GID for ${_b12x_ip:-<no address>} on $NCCL_IB_HCA; the rank will refuse to start (TFS_ROCE=0 for NCCL)" >&2
elif [ -n "${NCCL_IB_GID_INDEX:-}" ] && [ "$NCCL_IB_GID_INDEX" != "$B12X_ROCE_GID_INDEX" ]; then
  echo "[b12x env] WARNING: NCCL_IB_GID_INDEX=$NCCL_IB_GID_INDEX overrides B12X_ROCE_GID_INDEX=$B12X_ROCE_GID_INDEX in roce.py" >&2
fi
[ -n "${B12X_ENV_QUIET:-}" ] || echo "[b12x env] $(hostname): HCA=$NCCL_IB_HCA ip=${_b12x_ip:-?} B12X_ROCE_GID_INDEX=${B12X_ROCE_GID_INDEX:-unset} SPIN=$B12X_ROCE_SPIN_LIMIT GLOO=$GLOO_SOCKET_IFNAME" >&2
unset _b12x_ip _b12x_hex _b12x_p _b12x_i
