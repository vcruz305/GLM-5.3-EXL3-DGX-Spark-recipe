# AGENTS.md

Instructions for an AI agent setting this recipe up for a user. The user does not need to know the names or addresses
of their Sparks: `./glm53 init`, run on one of them, finds the other three. Read the [README](README.md) for the numbers
and the reasons; run only what is here, in this order.

## What you are setting up

GLM-5.3 (the SAGE MixedK EXL3 pack, 3.38 bpw) on four NVIDIA DGX Sparks as one OpenAI-compatible `/v1` server:
TensorFold, tensor parallel over the four (one rank per Spark), talking over the ConnectX-7 fabric. Rank 0 serves
`http://127.0.0.1:8890/v1`, model id `GLM-5.3-EXL3-3.38bpw`. Everything is driven by `./glm53` from one Spark, which
becomes rank 0.

## Prerequisites

Confirm these with the user first. You cannot fix 1-3 and 5 with commands; tell the user what is missing.

1. **Four DGX Sparks (GB10)**, powered on, each cabled to **one QSFP switch** on the **same ConnectX-7 port** (NVIDIA:
   four Sparks need a switch; use the same port on every Spark).
2. **An IPv4 address on that port on all four, in one subnet.** NVIDIA Sync's
   [Cluster Assistant](https://docs.nvidia.com/sync/latest/cluster-assistant.html) sets it up, or step 4 of NVIDIA's
   [Multi Sparks Through a Switch](https://build.nvidia.com/spark/multi-sparks-through-switch) playbook (static,
   DHCP from the switch, or link-local `169.254.x.x`). Check on a Spark: `ibdev2netdev` shows the port `(Up)` and
   `ip -br addr` shows an address on it.
3. **The same user account (user name) on all four.** NVIDIA's playbooks require it; `init` connects to the other
   Sparks as the current user.
4. **Passwordless ssh from the Spark you work on to the other three.** `init` checks it and prints the fix when it is
   missing (step 1 below).
5. **Hugging Face:** access to the gated pack
   [vcruz305/GLM-5.3-EXL3-3.38bpw](https://huggingface.co/vcruz305/GLM-5.3-EXL3-3.38bpw) (the user requests it on the
   model page and waits for approval) and a read token from <https://huggingface.co/settings/tokens>.
6. **On every Spark:** DGX OS (Ubuntu 24.04, aarch64), CUDA 13 with `nvcc`, `libpython3.12-dev`, `gcc`,
   `libibverbs-dev`, `rsync`, about 330 GB of free disk, and no other job on the GPU. `setup` checks each and prints the
   `sudo apt install` line for what is missing.

## Step 0: get a shell on any one Spark

That Spark becomes rank 0 and serves the API. Run every later command on it. If you already run on a Spark
(`nvidia-smi --query-gpu=name --format=csv,noheader` prints a name containing `GB10`), skip to step 1.

From the user's laptop, the ways NVIDIA documents
([Set Up Local Network Access](https://build.nvidia.com/spark/connect-to-your-spark)):

- **mDNS name:** `ssh <user>@<hostname>.local`. Every Spark announces its hostname over mDNS (NVIDIA's example is
  `spark-abcd.local`; the hostname is printed on the Quick Start Guide in the box) and announces ssh as the
  `_ssh._tcp` service. On a Linux laptop with `avahi-utils`, `avahi-browse -rt _ssh._tcp` lists the machines that
  announce ssh.
- **IP address:** `ssh <user>@<ip>`, with the Spark's address on the home or office network from the router's admin
  page, when `.local` names do not resolve (NVIDIA: some networks, such as complex corporate ones, block mDNS).
- **NVIDIA Sync:** if the user added the Spark to [NVIDIA Sync](https://docs.nvidia.com/sync/latest/direct-connections.html),
  Sync can open a terminal on it (Sync set up key-based ssh when the Spark was added). Ask the user to open that
  terminal for you.
- **At the Spark:** a display, keyboard and mouse.

Ask the user for the user name and, if `.local` does not resolve, the IP. Never ask for a password in chat: when ssh
asks for one, the user types it in their own terminal (or runs `ssh-copy-id <user>@<host>` once).

## Step 1: clone and find the other three Sparks

```bash
git clone https://github.com/vcruz305/GLM-5.3-EXL3-DGX-Spark-recipe.git && cd GLM-5.3-EXL3-DGX-Spark-recipe
./glm53 init
```

What `init` does, on this Spark (`tensorfold-four-spark-tp4/tools/discover.sh`):

1. checks that this machine is a Spark (`nvidia-smi` reports a GB10);
2. picks the fabric port: a ConnectX-7 port (`ibdev2netdev`) that is Up with an IPv4 address;
3. lists the machines on that port's subnet: `ip -4 neigh`, `avahi-browse _ssh._tcp` if installed, and a ping of every
   address when the subnet has at most 1,024 (a `169.254.0.0/16` link-local subnet is not swept);
4. connects to each with `ssh -o BatchMode=yes` (no password; the first contact records the host key) and reads its
   `/etc/machine-id`, GPU and address on the same port; keeps GB10 machines, one per machine-id;
5. requires exactly three others, orders them by fabric IP (this Spark is rank 0), and writes
   `tensorfold-four-spark-tp4/hosts` (fabric IPs as ssh targets) and `cluster.env` (`FABRIC_IFNAME`, `ROCE_HCA`).

Success ends like this (addresses and names differ):

```text
  found:  spark-1b2c 192.168.1.10, NVIDIA GB10, RDMA device rocep1s0f1, ssh 192.168.1.10 as nvidia: ok
  found:  spark-3d4e 192.168.1.12, NVIDIA GB10, RDMA device rocep1s0f1, ssh 192.168.1.12 as nvidia: ok
  found:  spark-5f6a 192.168.1.16, NVIDIA GB10, RDMA device rocep1s0f1, ssh 192.168.1.16 as nvidia: ok
==> 2/3 hosts file .../tensorfold-four-spark-tp4/hosts (ranks 1-3 in fabric-IP order)
==> 3/3 rank 0 is this Spark: discovery already checked ssh -o BatchMode=yes to ranks 1-3
==> init OK. Run every ./glm53 command on this Spark. Next: ./glm53 setup --download-once
```

If it prints `To fix:` instead, act on it (table at the end), then run `./glm53 init --force` (`--force` replaces an
existing hosts file). The common one is **missing passwordless ssh** (`Permission denied`, `found 0 of the other
three`). The fix is NVIDIA's `discover-sparks` script from its Connect Two Sparks / Multi Sparks playbooks. The user
runs it once, on this Spark, in their own terminal, as their normal user (not sudo). It asks for the account password
of each Spark:

```bash
curl -fsSLO https://raw.githubusercontent.com/NVIDIA/dgx-spark-playbooks/refs/heads/main/nvidia/connect-two-sparks/assets/discover-sparks && bash ./discover-sparks
```

It finds the Sparks over avahi on the ConnectX-7 ports (it needs `avahi-utils`: `sudo apt install -y avahi-utils`) and
installs one shared key, `~/.ssh/id_ed25519_shared`, on all of them
([source](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/connect-two-sparks/assets/discover-sparks)).
`init` also prints the by-hand alternative: `ssh-copy-id <ip>` for each Spark it could not log in to.

- `./glm53 init --dry-run` prints every discovery command and runs none.
- Overrides, only when discovery cannot work: `FABRIC_IFNAME=<port> ./glm53 init` (use that port);
  `./glm53 init --hosts local,<ip>,<ip>,<ip>` (name the other three yourself, from `ip -br addr` on each Spark; never
  guess); from a machine that is not a Spark, `./glm53 init --via <one Spark>` (your key must then be on all four).

## Step 2: install on all four

Check for a Hugging Face token on this Spark: `test -s ~/.cache/huggingface/token && echo token-present`. If there is
none, the user provides it themselves, in their own terminal on this Spark (never in a command you log or in chat):

```bash
HF_TOKEN=hf_... ./glm53 setup --download-once     # the user types this, with their token
```

With a token already present, you run:

```bash
./glm53 setup --download-once
```

It copies this clone to the other three, builds the runtime on all four in parallel (one log each in
`tensorfold-four-spark-tp4/runs/setup-<time>/`), downloads the 319 GB pack once on this Spark and copies it to the
others over the fabric. The download dominates (hours on a slow link); progress lines print every minute. Success:

```text
OK    rank 0 (local): ...
OK    rank 1 (192.168.1.10): ...
...
==> setup OK on all four. Next: ./glm53 up
```

A failure prints `FAIL  rank N ...` plus a `To fix:` block; act on it and run the same command again (finished steps
and downloads are kept). If it stops at `no Hugging Face token`, the runtime is already built: the user runs
`~/glm53-tensorfold/venv/bin/hf auth login` on this Spark (or the `HF_TOKEN=` line above), then you rerun setup. A pack
already on disk: `./glm53 setup --download-once --model-dir /absolute/path/to/GLM-5.3-EXL3-3.38bpw`. Afterwards
`./glm53 check` prints `OK` for each rank.

## Step 3: start the server

```bash
./glm53 up
```

Expect `preflight OK on all four`, then a status block every 30 s (`rank N: LOADING ...`) for about 8 minutes (the
first start is longer: it builds CUDA kernels), then:

```text
READY: rank 0 serves /v1 on 127.0.0.1:8890 (model GLM-5.3-EXL3-3.38bpw)
GLM-5.3 is up (profile fast-160k): http://127.0.0.1:8890/v1 on rank 0 (local), model id GLM-5.3-EXL3-3.38bpw
```

Then `./glm53 smoke` must end with `ALL CHECKS PASSED`.

## Step 4: talk to it

- One request with the engine's stats: `./glm53 chat "Explain RoCE in two sentences."` streams the answer and ends with
  `[finish stop; ... server decode N tok/s; ... mode dflash d7 c0.6]`.
- OpenAI-compatible API on rank 0: base URL `http://127.0.0.1:8890/v1`, model `GLM-5.3-EXL3-3.38bpw` (alias
  `glm-5.3`), `/v1/chat/completions` and `/v1/models`. The default bind is loopback with no API key (a client that
  insists on one can send any string).

```bash
curl -s http://127.0.0.1:8890/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "GLM-5.3-EXL3-3.38bpw",
  "messages": [{"role": "user", "content": "Explain RoCE in two sentences."}],
  "max_tokens": 1024}'
```

- One request at a time; others queue. Context 163,840 tokens (prompt plus reply); longer requests get HTTP 400.
- Thinking is on by default: the reasoning arrives in `reasoning_content`, the answer in `content` (read both fields of
  every streamed chunk). `"chat_template_kwargs": {"enable_thinking": false}` turns it off. Send a sensible
  `max_tokens` (default 4096): a client disconnect does not stop the generation.
- `"draft": false` decodes without the drafter (same tokens, slower). Every reply carries a `tensorfold` stats block.
- **From the user's laptop:** `./glm53 tunnel` prints `ssh -N -L 8890:127.0.0.1:8890 <user>@<this Spark>.local`; the
  user runs it on the laptop, then uses `http://127.0.0.1:8890/v1` there. Expose the port on the network only if the
  user asks: `TFS_HTTP_HOST=0.0.0.0 TFS_API_KEY_FILE=<file with a key> ./glm53 up` (refused without a key file).

## Status, logs, stop

- `./glm53 status`: every rank's state and rank 0's `/health`.
- `./glm53 logs`: copies rank logs, watchdog logs and bench JSON into `tensorfold-four-spark-tp4/runs/<time>/`.
- `./glm53 down`: stops all four ranks and their watchdogs. Restart: `./glm53 down`, then `./glm53 up`.
- `./glm53 bench`: the 6 reference prompts (token ids and tok/s against the README tables).
- After `git pull`: `./glm53 sync`, then `./glm53 check`. After the Sparks reboot: `./glm53 up` (if `preflight`
  reports `no RoCE v2 GID`, a fabric address changed: `./glm53 init --force` first).
- `--dry-run` on any command prints what it would run (ssh, rsync, discovery) and runs nothing.

Profiles: the default (`fast-160k`, 163,840 tokens) is the measured one. `./glm53 up --profile dcp4-262k` (262,144
tokens) is measured and slower. `--profile int4-262k --allow-unvalidated` only when the user asks for it.

## When something fails

Every `./glm53` command that fails prints a `To fix:` block; its output stays in `tensorfold-four-spark-tp4/runs/`.
Read it, the README's [Troubleshooting](README.md#troubleshooting) table and `./glm53 logs` before changing any setting.

| Message | What to do |
|---|---|
| `init`: `This machine ... is not a DGX Spark` | you are not on a Spark: step 0, then run `init` there |
| `init`: `Permission denied`, `found 0 of the other three` | passwordless ssh is missing: the user runs NVIDIA's `discover-sparks` line (step 1) on this Spark, then `./glm53 init --force` |
| `init`: `found 1 (or 2) of the other three`, `only N ... answered` | a Spark is off, cabled on another port, or has no address on the fabric port (prerequisites 1-2). On a `169.254.x.x` fabric: `sudo apt install -y avahi-utils` on this Spark (mDNS), then `./glm53 init --force` |
| `init`: `exactly four Sparks and 5 answered` | more Sparks on the fabric: ask the user which four, then `./glm53 init --hosts local,<ip>,<ip>,<ip> --force` |
| `init`: `ssh-keygen -R <ip>` in the fix | a Spark's host key changed (reinstalled): the user confirms, run that line, then `./glm53 init --force` |
| `init`: `no ConnectX-7 port is Up` / `has no IPv4 address yet` | prerequisites 1-2: cabling and fabric addresses (Cluster Assistant or the switch playbook) |
| `init`: `same /etc/machine-id` | two Sparks were cloned from one image: name the four with `--hosts`, and tell the user |
| `setup`: `no Hugging Face token` / gated repo errors | step 2: the user's token and pack access |
| `setup`: `no Python.h`, `gcc + libibverbs`, `nvcc not found`, `rsync not found` | the printed `sudo apt install` line (the user runs sudo), then `./glm53 setup --download-once` again |
| `setup`: pack incomplete / truncated / download failed | `./glm53 setup --download-once` again (resumes); stalls: `HF_HUB_DISABLE_XET=1 ./glm53 setup --download-once` |
| `recipe clone at <sha>, here <sha>` | `./glm53 sync` |
| `GPU BUSY`, `PORT ... in use`, `a server rank already runs` | `./glm53 status`; `./glm53 down` if it is this server, otherwise ask the user about the other job |
| `LOW MEMORY`, `GiB is free` | `./glm53 fadvise` and stop other jobs; never lower `TF_GLM53_CACHE_RESERVE_GB` |
| `REFUSE: no RoCE v2 GID` | a fabric address changed: `./glm53 init --force` |
| `START FAILED`, `TIMEOUT`, `FATAL` | `./glm53 logs`, read `rank*.log`, then `./glm53 down` and `./glm53 preflight` before the next `up` |
| `watchdog flagged`, `VIOLATION` | a memory guard stopped all four: `./glm53 logs`, `./glm53 down`, report to the user |
| replies end with `finish_reason: length` and only `reasoning_content` | the reply spent `max_tokens` thinking: raise it or turn thinking off |

## Do not

- **Do not guess host names or addresses**, and do not hand-write the hosts file. `./glm53 init` on a Spark finds the
  others; use `--hosts` only with addresses read from the Sparks themselves.
- **Do not ask for passwords or tokens in chat, and do not type them for the user.** `discover-sparks`,
  `ssh-copy-id`, `sudo` and `hf auth login` prompt in the user's terminal.
- **Do not install TensorFold from PyPI, a release tag or `ashhart/TensorFold` main.** None of them has the
  `glm_moe_dsa` family yet (PR #159 is open), and PR #159 alone does not load this pack (host memory and fp16
  tensors, see the folder README). `setup.sh` checks out the pinned `vcruz305/TensorFold` commits; every launcher
  refuses another tree.
- **Do not repoint the pins at the upstream PR branches** (`glm53-gb10-loading` and the rest, open as
  [drowzeys/TensorFold #1 to #5](https://github.com/drowzeys/TensorFold/pulls)). They rename functions
  the runtime check looks for, and none of them has been measured through these scripts.
- **Do not `pip install b12x`.** PyPI 1.3.0 has no `comm.roce` module. `setup.sh` stages commit `b58f34e`.
- **Do not serve the pack directory.** TensorFold needs the BF16 `lm_head.weight` and the fixed chat template; both
  live in the serve view `setup.sh` builds. The pack is never edited.
- **Do not run bf16 at 262,144 tokens with `TF_GLM53_DCP=1`.** It swapped and tripped the watchdog. Use the default
  profile, `--profile int4-262k --allow-unvalidated` or `--profile dcp4-262k`.
- **Do not lower `TF_GLM53_CACHE_RESERVE_GB`, disable the watchdog or allow swap growth** to make a context fit. A
  Spark that swaps under this load can hang until its watchdog reboots it.
- **Do not pass `--parallel`.** The server answers one request at a time; `--parallel N>1` switches TensorFold to a
  different scheduler that was never measured with this recipe.
- **Do not use `--profile int4-262k` / `--allow-unvalidated`** unless asked to: its quality gate is not finished.
- **Do not edit the TensorFold checkouts** under `~/glm53-tensorfold/`. `verify_runtime` refuses a dirty or moved tree.
- **Do not start while another job holds a GPU** on any of the four Sparks. `preflight` and `rank.sh start` refuse.
- **Do not put a Hugging Face token on a command line** that is logged or shared; `HF_TOKEN=... ./glm53 setup` passes
  it to the Sparks over ssh stdin only.
- **Do not expose port 8890 beyond loopback** unless the user asks, and then only with `TFS_API_KEY_FILE`.
- **Do not quote the SixCat decode figure as user-facing speed**, and do not redistribute the DFlash2 weights
  (CC BY-NC-ND 4.0: non-commercial use only).

## NVIDIA references

- Multi-Spark networking and ssh: [Multi Sparks Through a Switch](https://build.nvidia.com/spark/multi-sparks-through-switch)
  ([source](https://github.com/NVIDIA/dgx-spark-playbooks/tree/main/nvidia/multi-sparks-through-switch)),
  [Connect Two Sparks](https://build.nvidia.com/spark/connect-two-sparks),
  [`discover-sparks`](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/connect-two-sparks/assets/discover-sparks).
- ConnectX-7 port and interface names (`enp1s0f0np0` / `enp1s0f1np1`, `rocep1s0f*`):
  [DGX Spark User Guide, ConnectX-7 Networking](https://docs.nvidia.com/dgx/dgx-spark/spark-clustering.html).
- NVIDIA Sync: [Cluster Assistant](https://docs.nvidia.com/sync/latest/cluster-assistant.html) (sets up the fabric and
  ssh between up to four Sparks; four need a switch),
  [inspect the cluster network](https://docs.nvidia.com/sync/latest/cluster-network-inspection.html),
  [Direct Connections](https://docs.nvidia.com/sync/latest/direct-connections.html).
- Reaching a Spark: [Set Up Local Network Access](https://build.nvidia.com/spark/connect-to-your-spark).
