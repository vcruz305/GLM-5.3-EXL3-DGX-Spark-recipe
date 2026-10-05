#!/usr/bin/env python3
"""Memory / liveness watchdog for the GLM-5.3 TP=4 TensorFold server (one per node; the tf_serve stack's watchdog,
adapted from the sweep's tf_speed_memwatch.py - same rules, serving process pattern, no time limit by default).

Every INTERVAL s: MemAvailable >= TFS_MIN_AVAIL_GIB (recipe default 2) and swap used <= its value at watcher start
+ TFS_MAX_SWAP_GROWTH_KB (default 0: any growth trips). On a breach: kill -9 the local server rank, write the VIOLATION
flag and (--central, rank 0) kill every peer's rank too.
--central also polls the peers (TFS_PEERS: how rank 0 reaches ranks 1-3 over ssh) every PEER_EVERY s and stops all
four when
  * a peer reports a violation, or
  * some ranks are gone while others still run for more than TFS_WATCH_GRACE_S (45) s: a dead rank leaves the others
    blocked in a collective forever (rank 0 would accept requests it can never answer).
usage: tf_serve_memwatch.py [--central] [--max-hours H] [--dry-run]     (H 0 = until the STOP file appears)
Stop it with: touch /tmp/tf_serve_memwatch.STOP (rank.sh unwatch does this after the ranks exit)
Recipe changes vs the measured copy: peers come from TFS_PEERS (no built-in addresses), the floor default is 2 GiB, and
the swap rule takes an allowance (TFS_MAX_SWAP_GROWTH_KB, default 0 = the measured rule).
"""
import json
import os
import subprocess
import sys
import time

INTERVAL = 1.0
PEER_EVERY = 5.0
GRACE = float(os.environ.get("TFS_WATCH_GRACE_S", "45"))
MIN_AVAIL_KB = int(float(os.environ.get("TFS_MIN_AVAIL_GIB", "2")) * 1024 * 1024)
MAX_SWAP_KB = int(os.environ.get("TFS_MAX_SWAP_GROWTH_KB", "0"))
LOG, FLAG, STOP = "/tmp/tf_serve_memwatch.log", "/tmp/tf_serve_memwatch.VIOLATION", "/tmp/tf_serve_memwatch.STOP"
PAT = "[t]f_serve_rank.py"          # bracket: never matches its own pkill/pgrep/ssh command line


def peers_from_env() -> list[str]:
    """ssh destinations of ranks 1-3 as rank 0 reaches them (TFS_PEERS, space separated; rank.sh sets it)."""
    return os.environ.get("TFS_PEERS", "").split()


PEERS = peers_from_env()


def meminfo():
    d = {}
    with open("/proc/meminfo") as f:
        for ln in f:
            k, v = ln.split(":", 1)
            d[k] = int(v.split()[0])
    return d


def procs():
    return subprocess.run(["pgrep", "-f", PAT], capture_output=True, text=True).stdout.split()


def kill_local():
    subprocess.run(["pkill", "-9", "-f", PAT])


def ssh(ip, cmd):
    return subprocess.run(["timeout", "20", "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", ip, cmd],
                          capture_output=True, text=True)


def kill_peers():
    for ip in PEERS:
        ssh(ip, f"pkill -9 -f '{PAT}'")


def log(obj):
    with open(LOG, "a") as f:
        f.write(json.dumps(obj) + "\n")


def trip(why, rec, everywhere):
    kill_local()
    if everywhere:
        kill_peers()
    with open(FLAG, "w") as f:
        f.write(json.dumps({"why": why, **rec}) + "\n")
    rec["VIOLATION"] = why


def violations(avail_kb: int, swap_delta_kb: int) -> list[str]:
    viol = []
    if avail_kb < MIN_AVAIL_KB:
        viol.append(f"MemAvailable<{MIN_AVAIL_KB // 1048576}GiB")
    if swap_delta_kb > MAX_SWAP_KB:
        viol.append(f"swap grew {swap_delta_kb} kB")
    return viol


def split_state(seen: dict, counts: dict):
    """(dead, live) among the ranks ever seen running; both non-empty = a split cluster."""
    started = [k for k, v in seen.items() if v]
    dead = [k for k in started if counts.get(k) == 0]
    live = [k for k in started if counts.get(k, 0) > 0]
    return dead, live


def main(argv):
    central = "--central" in argv
    max_s = 0.0
    if "--max-hours" in argv:
        max_s = float(argv[argv.index("--max-hours") + 1]) * 3600
    if "--dry-run" in argv:
        print(json.dumps({"central": central, "peers": PEERS if central else [], "min_avail_gib": MIN_AVAIL_KB / 2**20,
                          "max_swap_growth_kb": MAX_SWAP_KB, "grace_s": GRACE, "pattern": PAT, "log": LOG,
                          "flag": FLAG, "stop": STOP, "max_hours": max_s / 3600 or "unlimited"}))
        return 0
    if central and len(PEERS) != 3:
        print(f"tf_serve_memwatch: --central needs TFS_PEERS = the 3 peers' ssh destinations, got {PEERS}",
              file=sys.stderr)
        return 2
    for p in (FLAG, STOP):
        try:
            os.remove(p)
        except FileNotFoundError:
            pass
    m = meminfo()
    base_swap = m["SwapTotal"] - m["SwapFree"]
    min_avail, max_delta = m["MemAvailable"], 0
    log({"event": "start", "t": time.time(), "host": os.uname().nodename, "central": central, "peers": PEERS,
         "baseline_swap_used_kb": base_swap, "mem_available_kb": m["MemAvailable"], "pid": os.getpid()})
    t_start = time.time()
    last_peer = 0.0
    tripped = False
    seen = {"local": False, **{ip: False for ip in PEERS}}
    split_since = None
    while (not max_s or time.time() - t_start < max_s) and not os.path.exists(STOP):
        m = meminfo()
        su = m["SwapTotal"] - m["SwapFree"]
        delta = su - base_swap
        min_avail = min(min_avail, m["MemAvailable"])
        max_delta = max(max_delta, delta)
        alive = procs()
        seen["local"] |= bool(alive)
        rec = {"t": round(time.time(), 1), "avail_gib": round(m["MemAvailable"] / 2**20, 2),
               "swap_used_mib": round(su / 1024, 1), "swap_delta_kb": delta,
               "min_avail_gib": round(min_avail / 2**20, 2), "max_swap_delta_kb": max_delta, "tf_procs": len(alive)}
        viol = violations(m["MemAvailable"], delta)
        if viol and not tripped:
            tripped = True
            trip(viol, rec, central)
        if central and time.time() - last_peer >= PEER_EVERY:
            last_peer = time.time()
            counts = {"local": len(alive)}
            for ip in PEERS:
                r = ssh(ip, f"cat {FLAG} 2>/dev/null; tail -n 1 {LOG}")
                out = r.stdout.strip()
                try:
                    counts[ip] = int(json.loads(out.splitlines()[-1]).get("tf_procs", -1))
                except (ValueError, IndexError, AttributeError):
                    counts[ip] = -1
                if counts[ip] > 0:
                    seen[ip] = True
                if ("VIOLATION" in out or '"why"' in out) and not tripped:
                    tripped = True
                    trip([f"peer {ip} violation"], rec, True)
                    rec["PEER_VIOLATION"] = ip
            dead, live = split_state(seen, counts)
            if dead and live:
                split_since = split_since or time.time()
                rec["split"] = {"dead": dead, "live": live, "for_s": round(time.time() - split_since, 1)}
                if time.time() - split_since > GRACE and not tripped:
                    tripped = True
                    trip([f"ranks gone on {dead} while {live} still run (> {GRACE:.0f} s)"], rec, True)
            else:
                split_since = None
            if not live and all(v for v in seen.values()):
                seen = {k: False for k in seen}     # a clean stop of all four: arm again for the next start
            rec["peers"] = counts
        log(rec)
        time.sleep(INTERVAL)
    log({"event": "stop", "t": time.time(), "min_avail_gib": round(min_avail / 2**20, 2),
         "max_swap_delta_kb": max_delta, "tripped": tripped})
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
