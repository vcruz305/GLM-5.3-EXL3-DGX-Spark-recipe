### A_roce (RoCE, tiles save:/workspace/tf_speed/tiles/tiles.json, status done)

Cells: median-of-3 tok/s (tokens/round, verify ms/round, draft ms/round); serial = median of the serial + bookend runs.

| config | code | prose | math | chat_explain | chat_multiturn | long_doc | mean core3 | mean all | exact |
|---|---|---|---|---|---|---|---|---|---|
| serial | 27.14 | 27.14 | 27.09 | 27.22 | 27.01 | 25.09 | 27.13 | 26.78 | yes |
| d7c0.30 | 39.21 (3.43, 79.0, 7.8) | 33.83 (2.63, 69.7, 7.4) | 55.65 (5.74, 94.9, 7.5) | 36.74 (3.06, 75.1, 7.5) | 31.21 (2.28, 64.7, 7.6) | 40.49 (3.84, 86.0, 8.1) | 42.90 | 39.52 | yes |
| d7c0.50 | 40.20 (3.19, 71.1, 7.7) | 35.55 (2.44, 60.3, 7.6) | 57.15 (5.50, 88.0, 7.5) | 38.67 (2.84, 65.3, 7.4) | 32.86 (2.21, 59.0, 7.6) | 41.47 (3.62, 78.9, 7.7) | 44.30 | 40.98 | yes |
| d7c0.40 | 39.81 (3.30, 74.5, 7.5) | 34.31 (2.52, 64.9, 7.8) | 56.68 (5.68, 91.8, 7.6) | 38.30 (2.97, 69.3, 7.5) | 32.21 (2.26, 61.8, 7.7) | 41.57 (3.76, 82.1, 7.6) | 43.60 | 40.48 | yes |
| d7c0.60 | 40.62 (3.08, 67.6, 7.4) | 35.58 (2.35, 58.2, 7.3) | 57.39 (5.38, 85.6, 7.4) | 38.69 (2.73, 62.6, 7.3) | 32.83 (2.16, 57.3, 7.7) | 43.44 (3.60, 74.5, 7.7) | 44.53 | 41.42 | yes |
| d7c0.75 | 39.94 (2.84, 62.8, 7.5) | 35.28 (2.25, 55.7, 7.4) | 57.42 (5.01, 79.2, 7.4) | 37.98 (2.56, 59.1, 7.5) | 32.77 (2.05, 54.6, 7.3) | 42.87 (3.34, 69.7, 7.5) | 44.21 | 41.04 | yes |
| d5c0.50 | 39.72 (3.06, 68.6, 7.6) | 35.37 (2.40, 59.5, 7.6) | 54.25 (4.78, 80.0, 7.5) | 37.82 (2.76, 64.7, 7.6) | 32.91 (2.18, 58.3, 7.3) | 41.74 (3.48, 74.9, 7.6) | 43.11 | 40.30 | yes |

exactness: 125 runs identical to serial, 0 differ; failures: 0

fixed-row verify window, rank-0 wall ms (graph replay + host read), median of 15: raw / through the per-round health guard / guard cost
| ctx | P0 | R1 | R2 | R3 | R4 | R5 | R6 | R7 | R8 |
|---|---|---|---|---|---|---|---|---|---|
| short | 116 | 36.2 / 36.5 / 0.28 | 46.0 / 46.8 / 0.78 | 55.5 / 55.9 / 0.40 | 62.9 / 63.5 / 0.63 | 70.6 / 71.3 / 0.70 | 78.0 / 78.3 / 0.30 | 86.4 / 87.4 / 1.01 | 93.2 / 93.2 / -0.02 |
| long | 2906 | 38.8 / 39.4 / 0.66 | 51.6 / 51.8 / 0.20 | 61.5 / 62.1 / 0.59 | 69.0 / 70.2 / 1.21 | 77.4 / 78.0 / 0.64 | 84.7 / 85.3 / 0.57 | 93.1 / 93.6 / 0.47 | 103.2 / 104.0 / 0.83 |

health-guard cost (guarded - raw) over 16 window sizes: median 0.61 ms, mean 0.58 ms, range -0.02..1.21 ms
least-squares fit (short): verify_ms(R) = 29.9 + 8.05 R
least-squares fit (long): verify_ms(R) = 32.9 + 8.77 R
drafter block pass (rowtime step, median of 15): code 7.61 ms, long_doc 7.83 ms

distinct routed experts per MoE layer vs window rows (eager probe, 48 windows of 8 serial-reply tokens, 75 MoE layers); i.i.d. = 256(1-(1-8/256)^R)
| R | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| measured mean | 8.0 | 13.8 | 19.0 | 23.6 | 28.0 | 32.3 | 36.5 | 40.3 |
| min / max layer | 8 / 8 | 8 / 16 | 11 / 24 | 13 / 32 | 14 / 40 | 15 / 48 | 17 / 54 | 21 / 62 |
| i.i.d. | 8.0 | 15.8 | 23.3 | 30.5 | 37.6 | 44.4 | 51.0 | 57.4 |
R=8 by prompt: code 41.1, prose 39.0, math 42.4, chat_explain 39.0; teacher-forced windows equal to serial: 48/48

profile: code serial (tokens/round 1.0) - top kernels, GPU ms a round over 8 rounds (rank 0, profiler on: wall is inflated):
```
rounds 8  wall/round 40.91 ms  gpu kernel time/round 40.86 ms  (gaps/overlap +0.05 ms)
    9.107 ms/round    124x  void tf_exl3x::grouped_kernel<2, 8, 4, 1, 2, 10>(__half const*, __half const*, long const*, long const*, int const*, int c
    5.865 ms/round    156x  kernel_cutlass_kernel_b12xcommroce_oneshot_cute_RoceOneshotLaunch_object_at__ptri32gmemalign16_ptri32gmemalign16__________
    5.557 ms/round    198x  void (anonymous namespace)::linear_kernel<6, 2, 4, 1, 1>((anonymous namespace)::Jobs, int, int)
    3.578 ms/round     97x  _router_part
    3.017 ms/round     25x  void tf_exl3x::grouped_kernel<2, 8, 4, 1, 8, 8>(__half const*, __half const*, long const*, long const*, int const*, int co
    2.312 ms/round    107x  void (anonymous namespace)::linear_kernel<6, 2, 8, 1, 1>((anonymous namespace)::Jobs, int, int)
    1.930 ms/round    435x  (anonymous namespace)::rot_in_kernel(void const*, int, (anonymous namespace)::RotJobs, int)
    1.532 ms/round     78x  _expand
    1.161 ms/round     78x  _absorb
    1.135 ms/round     78x  _attn_chunks
    1.026 ms/round     67x  void (anonymous namespace)::linear_kernel<8, 2, 8, 1, 1>((anonymous namespace)::Jobs, int, int)
    0.833 ms/round     34x  void (anonymous namespace)::linear_kernel<10, 2, 8, 1, 1>((anonymous namespace)::Jobs, int, int)
```

profile: code prof-d7c0.30 (tokens/round 3.43) - top kernels, GPU ms a round over 8 rounds (rank 0, profiler on: wall is inflated):
```
rounds 8  wall/round 108.84 ms  gpu kernel time/round 108.15 ms  (gaps/overlap +0.70 ms)
   43.460 ms/round    124x  void tf_exl3x::grouped_kernel<2, 8, 4, 1, 2, 10>(__half const*, __half const*, long const*, long const*, int const*, int c
   14.297 ms/round     25x  void tf_exl3x::grouped_kernel<2, 8, 4, 1, 8, 8>(__half const*, __half const*, long const*, long const*, int const*, int co
   14.132 ms/round    156x  kernel_cutlass_kernel_b12xcommroce_oneshot_cute_RoceOneshotLaunch_object_at__ptri32gmemalign16_ptri32gmemalign16__________
    6.196 ms/round    198x  void (anonymous namespace)::linear_kernel<6, 2, 4, 1, 1>((anonymous namespace)::Jobs, int, int)
    5.939 ms/round     98x  _router_part
    2.597 ms/round    107x  void (anonymous namespace)::linear_kernel<6, 2, 8, 1, 1>((anonymous namespace)::Jobs, int, int)
    2.537 ms/round     14x  ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<4096ul>)
    2.261 ms/round     31x  void (anonymous namespace)::group_kernel<64, 16, 64, 1, 4, 8, false, false, false, false>(__nv_bfloat16 const*, float cons
    2.152 ms/round     78x  _expand
    2.085 ms/round    435x  (anonymous namespace)::rot_in_kernel(void const*, int, (anonymous namespace)::RotJobs, int)
    1.804 ms/round     78x  _absorb
    1.226 ms/round     75x  (anonymous namespace)::group_kernel(int const*, int*, int*, int*, int, int, int, int)
```

profile: prose prof-d7c0.30 (tokens/round 2.634) - top kernels, GPU ms a round over 8 rounds (rank 0, profiler on: wall is inflated):
```
rounds 8  wall/round 101.90 ms  gpu kernel time/round 101.24 ms  (gaps/overlap +0.66 ms)
   38.900 ms/round    124x  void tf_exl3x::grouped_kernel<2, 8, 4, 1, 2, 10>(__half const*, __half const*, long const*, long const*, int const*, int c
   13.659 ms/round     25x  void tf_exl3x::grouped_kernel<2, 8, 4, 1, 8, 8>(__half const*, __half const*, long const*, long const*, int const*, int co
   12.881 ms/round    156x  kernel_cutlass_kernel_b12xcommroce_oneshot_cute_RoceOneshotLaunch_object_at__ptri32gmemalign16_ptri32gmemalign16__________
    6.150 ms/round    198x  void (anonymous namespace)::linear_kernel<6, 2, 4, 1, 1>((anonymous namespace)::Jobs, int, int)
    5.920 ms/round     98x  _router_part
    2.579 ms/round    107x  void (anonymous namespace)::linear_kernel<6, 2, 8, 1, 1>((anonymous namespace)::Jobs, int, int)
    2.349 ms/round     14x  ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<4096ul>)
    2.294 ms/round     31x  void (anonymous namespace)::group_kernel<64, 16, 64, 1, 4, 8, false, false, false, false>(__nv_bfloat16 const*, float cons
    2.126 ms/round     78x  _expand
    2.119 ms/round    435x  (anonymous namespace)::rot_in_kernel(void const*, int, (anonymous namespace)::RotJobs, int)
    1.755 ms/round     78x  _absorb
    1.206 ms/round     78x  _attn_chunks
```

memory (1 s watchdog, whole run incl. load): rank 0: min MemAvailable 19.78 GiB, max swap growth 0 kB (baseline swap used 238.3 MiB); rank 1: min MemAvailable 16.92 GiB, max swap growth 0 kB (baseline swap used 253.6 MiB); rank 2: min MemAvailable 18.54 GiB, max swap growth 0 kB (baseline swap used 227.4 MiB); rank 3: min MemAvailable 19.14 GiB, max swap growth 0 kB (baseline swap used 2142.8 MiB)

### B_nccl (NCCL, tiles load:/workspace/tf_speed/tiles/tiles.json, status done)

Cells: median-of-3 tok/s (tokens/round, verify ms/round, draft ms/round); serial = median of the serial + bookend runs.

| config | code | prose | math | chat_explain | chat_multiturn | long_doc | mean core3 | mean all | exact |
|---|---|---|---|---|---|---|---|---|---|
| serial | 24.12 | 24.13 | 24.20 | 24.22 | 24.23 | 23.05 | 24.15 | 23.99 | yes |
| d7c0.30 | 35.63 (3.43, 87.9, 7.6) | 30.49 (2.63, 78.0, 7.7) | 49.55 (5.74, 107.5, 7.7) | 33.05 (3.06, 84.2, 7.7) | 28.19 (2.28, 72.6, 7.6) | 36.65 (3.84, 96.2, 7.9) | 38.56 | 35.59 | yes |
| d7c0.50 | 36.23 (3.19, 79.7, 7.6) | 31.87 (2.44, 68.3, 7.6) | 51.01 (5.50, 99.3, 7.6) | 34.85 (2.84, 73.1, 7.6) | 29.57 (2.21, 66.6, 7.5) | 37.48 (3.62, 88.1, 7.8) | 39.70 | 36.84 | yes |
| d7c0.40 | 35.85 (3.30, 83.6, 7.6) | 31.05 (2.52, 72.7, 7.7) | 50.69 (5.68, 103.6, 7.7) | 34.57 (2.97, 77.5, 7.6) | 29.04 (2.26, 69.5, 7.6) | 37.46 (3.76, 91.7, 7.8) | 39.20 | 36.44 | yes |
| d7c0.60 | 36.44 (3.08, 76.1, 7.6) | 31.48 (2.35, 66.4, 7.6) | 51.30 (5.38, 96.5, 7.6) | 34.71 (2.73, 70.4, 7.5) | 29.37 (2.16, 65.1, 7.5) | 39.10 (3.60, 83.3, 7.9) | 39.74 | 37.07 | yes |

exactness: 86 runs identical to serial, 0 differ; failures: 0

fixed-row verify window, rank-0 wall ms (graph replay + host read), median of 15: raw / through the per-round health guard / guard cost
| ctx | P0 | R1 | R2 | R3 | R4 | R5 | R6 | R7 | R8 |
|---|---|---|---|---|---|---|---|---|---|
| short | 116 | 40.7 / 41.1 / 0.34 | 55.1 / 55.3 / 0.13 | 62.7 / 62.5 / -0.12 | 70.0 / 69.8 / -0.27 | 77.5 / 77.8 / 0.36 | 85.1 / 85.7 / 0.64 | 96.8 / 97.3 / 0.58 | 106.5 / 106.5 / 0.01 |
| long | 2906 | 41.9 / 42.4 / 0.47 | 59.6 / 61.1 / 1.49 | 67.6 / 67.9 / 0.32 | 76.3 / 76.5 / 0.21 | 84.1 / 84.9 / 0.84 | 93.7 / 93.3 / -0.36 | 104.8 / 104.1 / -0.70 | 115.6 / 116.5 / 0.86 |

health-guard cost (guarded - raw) over 16 window sizes: median 0.33 ms, mean 0.30 ms, range -0.70..1.49 ms
least-squares fit (short): verify_ms(R) = 34.5 + 8.84 R
least-squares fit (long): verify_ms(R) = 36.1 + 9.86 R
drafter block pass (rowtime step, median of 15): code 7.43 ms, long_doc 7.89 ms

memory (1 s watchdog, whole run incl. load): rank 0: min MemAvailable 18.68 GiB, max swap growth 0 kB (baseline swap used 238.2 MiB); rank 1: min MemAvailable 16.45 GiB, max swap growth 0 kB (baseline swap used 253.4 MiB); rank 2: min MemAvailable 18.26 GiB, max swap growth 0 kB (baseline swap used 227.3 MiB); rank 3: min MemAvailable 18.19 GiB, max swap growth 0 kB (baseline swap used 2142.5 MiB)

### C_roce (RoCE, tiles load:/workspace/tf_speed/tiles/tiles.json, status done)

Cells: median-of-3 tok/s (tokens/round, verify ms/round, draft ms/round); serial = median of the serial + bookend runs.

| config | code | prose | math | chat_explain | chat_multiturn | long_doc | mean core3 | mean all | exact |
|---|---|---|---|---|---|---|---|---|---|
| serial | 26.98 | 26.92 | 26.95 | 26.98 | 26.98 | 25.16 | 26.95 | 26.66 | yes |
| d7c0.60 | 40.54 (3.08, 67.8, 7.4) | 35.16 (2.35, 58.6, 7.7) | 57.18 (5.38, 85.8, 7.6) | 38.54 (2.73, 62.7, 7.4) | 32.84 (2.16, 57.4, 7.6) | 43.19 (3.60, 74.7, 7.8) | 44.29 | 41.24 | yes |
| d7c0.60e0.40 | 40.10 (3.04, 67.5, 7.6) | 35.01 (2.33, 58.3, 7.7) | 57.56 (5.38, 85.3, 7.4) | 37.81 (2.67, 62.4, 7.6) | 32.74 (2.15, 57.4, 7.5) | 42.79 (3.52, 73.7, 7.9) | 44.22 | 41.00 | yes |
| d7c0.60e0.90 | 40.49 (3.12, 68.4, 7.7) | 34.75 (2.35, 59.4, 7.6) | 57.52 (5.44, 86.2, 7.6) | 38.01 (2.73, 63.7, 7.5) | 32.83 (2.16, 57.5, 7.5) | 42.31 (3.55, 75.3, 7.8) | 44.25 | 40.98 | yes |
| d7c0.70 | 39.84 (2.92, 64.7, 7.8) | 34.82 (2.26, 56.6, 7.6) | 56.08 (5.06, 81.9, 7.5) | 38.37 (2.65, 60.7, 7.5) | 32.84 (2.10, 55.8, 7.5) | 43.11 (3.43, 71.1, 7.7) | 43.58 | 40.84 | yes |
| d7c0.55 | 40.59 (3.13, 69.1, 7.4) | 35.11 (2.38, 59.6, 7.4) | 56.70 (5.38, 86.6, 7.6) | 38.47 (2.78, 64.1, 7.5) | 32.65 (2.17, 58.1, 7.4) | 41.71 (3.60, 77.7, 7.9) | 44.13 | 40.87 | yes |

exactness: 104 runs identical to serial, 0 differ; failures: 0

memory (1 s watchdog, whole run incl. load): rank 0: min MemAvailable 19.7 GiB, max swap growth 0 kB (baseline swap used 238.1 MiB); rank 1: min MemAvailable 16.88 GiB, max swap growth 0 kB (baseline swap used 253.3 MiB); rank 2: min MemAvailable 18.75 GiB, max swap growth 0 kB (baseline swap used 227.0 MiB); rank 3: min MemAvailable 19.04 GiB, max swap growth 0 kB (baseline swap used 2142.4 MiB)

### Cross-boot serial reply hashes (pinned tiles)

| prompt | A_roce (RoCE) | B_nccl (NCCL) | C_roce (RoCE) | equal |
|---|---|---|---|---|
| code | 4bdccad4d7b29264 | 4bdccad4d7b29264 | 4bdccad4d7b29264 | EQUAL |
| prose | ad50c3af6ffc1f7b | ad50c3af6ffc1f7b | ad50c3af6ffc1f7b | EQUAL |
| math | ab7ef1414c11ac53 | ab7ef1414c11ac53 | ab7ef1414c11ac53 | EQUAL |
| chat_explain | 991dae523a8675ee | 991dae523a8675ee | 991dae523a8675ee | EQUAL |
| chat_multiturn | 776da81c50f6f960 | 776da81c50f6f960 | 776da81c50f6f960 | EQUAL |
| long_doc | c085e3f6f231f7d1 | c085e3f6f231f7d1 | c085e3f6f231f7d1 | EQUAL |

### Ranking by mean tok/s over all prompts (exact configs only)

| rank | arm | config | mean all | mean core3 | exact |
|---|---|---|---|---|---|
| 1 | A_roce (RoCE) | d7c0.60 | 41.42 | 44.53 | yes |
| 2 | C_roce (RoCE) | d7c0.60 | 41.24 | 44.29 | yes |
| 3 | A_roce (RoCE) | d7c0.75 | 41.04 | 44.21 | yes |
| 4 | C_roce (RoCE) | d7c0.60e0.40 | 41.00 | 44.22 | yes |
| 5 | C_roce (RoCE) | d7c0.60e0.90 | 40.98 | 44.25 | yes |
| 6 | A_roce (RoCE) | d7c0.50 | 40.98 | 44.30 | yes |
| 7 | C_roce (RoCE) | d7c0.55 | 40.87 | 44.13 | yes |
| 8 | C_roce (RoCE) | d7c0.70 | 40.84 | 43.58 | yes |
| 9 | A_roce (RoCE) | d7c0.40 | 40.48 | 43.60 | yes |
| 10 | A_roce (RoCE) | d5c0.50 | 40.30 | 43.11 | yes |
| 11 | A_roce (RoCE) | d7c0.30 | 39.52 | 42.90 | yes |
| 12 | B_nccl (NCCL) | d7c0.60 | 37.07 | 39.74 | yes |
| 13 | B_nccl (NCCL) | d7c0.50 | 36.84 | 39.70 | yes |
| 14 | B_nccl (NCCL) | d7c0.40 | 36.44 | 39.20 | yes |
| 15 | B_nccl (NCCL) | d7c0.30 | 35.59 | 38.56 | yes |
