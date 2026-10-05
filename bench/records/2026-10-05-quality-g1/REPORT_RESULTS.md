# TensorFold vs exllamav3 teacher-forced gate - PASS (with REVIEW flags)

| block | set | n | NLL ref | NLL cand | dNLL [95% CI] | T_nll | NLL | acc ref | acc cand | dacc [95% CI] | T_acc | acc | agree all | agree m>=0.5 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| prompt_path | sixcat | 2975 | 4.91739 | 4.77946 | -0.13793 [-0.39416, +0.10817] | 0.07332 | **PASS** | 0.5224 | 0.5217 | -0.0007 [-0.0199, +0.0184] | 0.0024 | **PASS** | 0.5116 | 0.5815 |
| prompt_path | natural | 40928 | 1.83218 | 1.83895 | +0.00678 [-0.00331, +0.02409] | 0.01306 | **PASS** | 0.6110 | 0.6103 | -0.0006 [-0.0036, +0.0016] | 0.0011 | **PASS** | 0.9195 | 0.9846 |
| prompt_path | all | 43903 | 2.04124 | 2.03821 | -0.00303 [-0.02425, +0.02152] | 0.00721 | **PASS** | 0.6050 | 0.6043 | -0.0006 [-0.0037, +0.0017] | 0.0011 | **PASS** | 0.8918 | 0.9561 |
| decode_path | sixcat | 376 | 4.01814 | 3.77224 | -0.24590 [-0.80656, +0.30891] | 0.31752 | **PASS** | 0.5213 | 0.5452 | +0.0239 [-0.0343, +0.0532] | 0.0106 | **PASS** | 0.6383 | 0.6893 |
| decode_path | natural | 2048 | 1.73223 | 1.73767 | +0.00544 [-0.00383, +0.01558] | 0.01122 | **PASS** | 0.6138 | 0.6089 | -0.0049 [-0.0107, +0.0010] | 0.0063 | **PASS** | 0.9146 | 0.9818 |
| decode_path | all | 2424 | 2.08681 | 2.05326 | -0.03355 [-0.14123, +0.04222] | 0.04444 | **PASS** | 0.5994 | 0.5990 | -0.0004 [-0.0091, +0.0079] | 0.0037 | **PASS** | 0.8717 | 0.9343 |

## Greedy (SixCat prompts)

| arm/row | first 4 | baseline | ok | exllamav3 first 4 ok | first divergence vs exllamav3 |
|---|---|---|---|---|---|
| tf_greedy/sixcat_123 | [15209, 74702, 33666, 3145] | [15209, 74702, 33666, 3145] | True | True | None |
| tf_greedy/sixcat_383 | [15209, 74702, 33666, 3145] | [15209, 74702, 33666, 3145] | True | True | None |
| tf_greedy/sixcat_2475 | [15209, 74702, 33666, 3145] | [15209, 74702, 33666, 3145] | True | True | None |

## Self-controls and secondary comparisons (not gated)

| pair | n | dNLL | agree all | agree m>=0.5 |
|---|---|---|---|---|
| ex_p512_vs_ex_p8192 | 43903 | -0.00742 | 0.8962 | 0.9607 |
| ex_p2048_vs_ex_p8192 | 43903 | -0.00046 | 0.8968 | 0.9602 |
| ex_p8192_oneshot_vs_ex_p8192 | 43903 | +0.00721 | 0.8957 | 0.9582 |
| ex_d_vs_ex_p8192 | 2424 | -0.03044 | 0.8779 | 0.9475 |
| ex_d_p512_vs_ex_d | 2424 | +0.02774 | 0.9146 | 0.97 |
| tf_p_vs_ex_p512 | 43903 | +0.00439 | 0.8927 | 0.9576 |
| tf_p_vs_ex_p2048 | 43903 | -0.00257 | 0.8917 | 0.9564 |
| tf_d_vs_ex_d_p512 | 2424 | -0.06129 | 0.8771 | 0.9349 |
| tf_p_rs_vs_tf_p | 43903 | -0.00091 | 0.9055 | 0.9612 |
| tf_p_nosp_vs_tf_p | 43903 | +0.01854 | 0.9081 | 0.9622 |
| tf_p_c2048_vs_tf_p | 43903 | +0.00000 | 1.0000 | 1.0 |
| tf_p_c1024_vs_tf_p | 43903 | -0.00728 | 0.9111 | 0.9653 |
| tf_d_vs_tf_p | 2424 | +0.00959 | 0.9043 | 0.9451 |

## Flags

- DIAGNOSTIC: ex_d top-1 on copyable (repeated-context) positions is 0.828 (< 0.90): the engine is not using its context the way a healthy model does
- DIAGNOSTIC: ex_d_oneshot top-1 on copyable (repeated-context) positions is 0.825 (< 0.90): the engine is not using its context the way a healthy model does
- DIAGNOSTIC: ex_d_p512 top-1 on copyable (repeated-context) positions is 0.822 (< 0.90): the engine is not using its context the way a healthy model does
- DIAGNOSTIC: ex_d_p512_oneshot top-1 on copyable (repeated-context) positions is 0.823 (< 0.90): the engine is not using its context the way a healthy model does
- DIAGNOSTIC: ex_p2048 top-1 on copyable (repeated-context) positions is 0.840 (< 0.90): the engine is not using its context the way a healthy model does
- DIAGNOSTIC: ex_p2048_oneshot top-1 on copyable (repeated-context) positions is 0.838 (< 0.90): the engine is not using its context the way a healthy model does
- DIAGNOSTIC: ex_p512 top-1 on copyable (repeated-context) positions is 0.842 (< 0.90): the engine is not using its context the way a healthy model does
- DIAGNOSTIC: ex_p8192 top-1 on copyable (repeated-context) positions is 0.843 (< 0.90): the engine is not using its context the way a healthy model does
- DIAGNOSTIC: ex_p8192_oneshot top-1 on copyable (repeated-context) positions is 0.837 (< 0.90): the engine is not using its context the way a healthy model does
- DIAGNOSTIC: tf_d top-1 on copyable (repeated-context) positions is 0.832 (< 0.90): the engine is not using its context the way a healthy model does
- DIAGNOSTIC: tf_p top-1 on copyable (repeated-context) positions is 0.839 (< 0.90): the engine is not using its context the way a healthy model does
- DIAGNOSTIC: tf_p_c1024 top-1 on copyable (repeated-context) positions is 0.841 (< 0.90): the engine is not using its context the way a healthy model does
- DIAGNOSTIC: tf_p_c2048 top-1 on copyable (repeated-context) positions is 0.839 (< 0.90): the engine is not using its context the way a healthy model does
- DIAGNOSTIC: tf_p_nosp top-1 on copyable (repeated-context) positions is 0.835 (< 0.90): the engine is not using its context the way a healthy model does
- DIAGNOSTIC: tf_p_rs top-1 on copyable (repeated-context) positions is 0.839 (< 0.90): the engine is not using its context the way a healthy model does
- SERVING: TensorFold's chat template renders sixcat_123 to different ids than the baseline's strict template (serving parity, not a teacher-forced result)
- SERVING: TensorFold's chat template renders sixcat_383 to different ids than the baseline's strict template (serving parity, not a teacher-forced result)
- SERVING: TensorFold's chat template renders sixcat_2475 to different ids than the baseline's strict template (serving parity, not a teacher-forced result)
- DIAGNOSTIC: prompt_path/sixcat: shape control ex_p2048 vs ex_p8192 is NOT benign (agree 0.5681, dNLL -0.08690 on NLL 4.9174); excluded from the threshold
- REVIEW: prompt_path/sixcat: agreement at margin >= 0.5 is 0.5815 vs exllamav3 self-control 0.6113
- DIAGNOSTIC: prompt_path/all: shape control ex_p2048 vs ex_p8192 is NOT benign (agree 0.8968, dNLL -0.00046 on NLL 2.0412); excluded from the threshold
- DIAGNOSTIC: decode_path/sixcat: shape control ex_d_p512 vs ex_d is NOT benign (agree 0.9016, dNLL +0.11773 on NLL 4.0181); excluded from the threshold
- REVIEW: decode_path/sixcat: agreement at margin >= 0.5 is 0.6893 vs exllamav3 self-control 0.7314
- DIAGNOSTIC: decode_path/all: shape control ex_d_p512 vs ex_d is NOT benign (agree 0.9146, dNLL +0.02774 on NLL 2.0868); excluded from the threshold
- NOTE: optional block prompt_path_dcp not run (tf_dcp_p missing)
- NOTE: optional block decode_path_dcp not run (tf_dcp_d missing)

## Arms

| arm | set | n | mean NLL | top-1 acc |
|---|---|---|---|---|
| ex_d | sixcat | 376 | 4.01814 | 0.5213 |
| ex_d | natural | 2048 | 1.73223 | 0.6138 |
| ex_d_oneshot | sixcat | 376 | 3.70062 | 0.5319 |
| ex_d_oneshot | natural | 2048 | 1.73793 | 0.6074 |
| ex_d_p512 | sixcat | 376 | 4.13587 | 0.5160 |
| ex_d_p512 | natural | 2048 | 1.74345 | 0.6074 |
| ex_d_p512_oneshot | sixcat | 376 | 3.75474 | 0.5346 |
| ex_d_p512_oneshot | natural | 2048 | 1.74287 | 0.6104 |
| ex_greedy | sixcat | 192 | 0.00758 | 1.0000 |
| ex_p2048 | sixcat | 2975 | 4.83049 | 0.5176 |
| ex_p2048 | natural | 40928 | 1.83800 | 0.6115 |
| ex_p2048_oneshot | sixcat | 2975 | 4.82311 | 0.5166 |
| ex_p2048_oneshot | natural | 40928 | 1.84007 | 0.6099 |
| ex_p512 | sixcat | 2975 | 4.68416 | 0.5291 |
| ex_p512 | natural | 40928 | 1.84117 | 0.6106 |
| ex_p8192 | sixcat | 2975 | 4.91739 | 0.5224 |
| ex_p8192 | natural | 40928 | 1.83218 | 0.6110 |
| ex_p8192_oneshot | sixcat | 2975 | 4.84407 | 0.5200 |
| ex_p8192_oneshot | natural | 40928 | 1.84524 | 0.6099 |
| tf_d | sixcat | 376 | 3.77224 | 0.5452 |
| tf_d | natural | 2048 | 1.73767 | 0.6089 |
| tf_greedy | sixcat | 192 | 0.00594 | 1.0000 |
| tf_p | sixcat | 2975 | 4.77946 | 0.5217 |
| tf_p | natural | 40928 | 1.83895 | 0.6103 |
| tf_p_c1024 | sixcat | 2975 | 4.64000 | 0.5311 |
| tf_p_c1024 | natural | 40928 | 1.84128 | 0.6100 |
| tf_p_c2048 | sixcat | 2975 | 4.77946 | 0.5217 |
| tf_p_c2048 | natural | 40928 | 1.83895 | 0.6103 |
| tf_p_nosp | sixcat | 2975 | 5.01881 | 0.5082 |
| tf_p_nosp | natural | 40928 | 1.84145 | 0.6108 |
| tf_p_rs | sixcat | 2975 | 4.72947 | 0.5230 |
| tf_p_rs | natural | 40928 | 1.84162 | 0.6102 |

## Recorded PP baseline vs exllamav3 TP reference arm

| row | n | PP NLL sum | TP NLL sum | rel | PP top1 | TP top1 |
|---|---|---|---|---|---|---|
| sixcat_123 | 121 | 560.084 | 566.216 | +1.0949% | 40 | 39 |
| sixcat_383 | 381 | 1688.607 | 1569.824 | -7.0344% | 170 | 190 |
| sixcat_2475 | 2473 | 11258.820 | 12493.199 | +10.9637% | 1380 | 1325 |
