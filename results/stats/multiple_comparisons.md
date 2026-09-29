# Multiple-comparison check

All 35 paired tests behind the writeup's claims form one family. Bonferroni threshold: 0.05 / 35 = 0.0014. Holm-adjusted p-values control the same family-wise error rate and are never larger than Bonferroni's.

A and B are conditions on the same seeds; `only A` / `only B` count the seeds where only one of them succeeds; the difference is B − A in success rate, with a 95% interval that is consistent with the exact test (Clopper–Pearson on the discordant seeds).

| id | claim | model | A | B | successes A / B | only A / only B | p | Holm p | Bonferroni | B − A [95% CI] |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | Retrained vs official checkpoint, full chunk, nominal | pair | `official` | `ours` | 163 / 149 | 30 / 16 | 0.054 | 0.59 | n/a (no-difference claim) | -7.0 [-13.2, +0.1] |
| F1a | Temporal ensembling (TE) is insensitive to latency: 0 vs 400 ms | official | `te_d00_m0` | `te_d20_m0` | 142 / 136 | 11 / 5 | 0.21 | 1 | n/a (no-difference claim) | -3.0 [-6.2, +1.4] |
| F1a | Temporal ensembling (TE) is insensitive to latency: 0 vs 400 ms | ours | `te_d00_m0` | `te_d20_m0` | 149 / 154 | 8 / 13 | 0.38 | 1 | n/a (no-difference claim) | +2.5 [-2.4, +6.7] |
| F1b | TE is no better than full chunk after a 6 cm move | official | `full_d00_m6` | `te_d00_m6` | 40 / 53 | 20 / 33 | 0.098 | 0.89 | n/a (no-difference claim) | +6.5 [-1.1, +13.4] |
| F1b | TE is no better than full chunk after a 6 cm move | ours | `full_d00_m6` | `te_d00_m6` | 37 / 42 | 18 / 23 | 0.53 | 1 | n/a (no-difference claim) | +2.5 [-4.2, +8.8] |
| F2a | Replan 25 beats full chunk after a 6 cm move | official | `full_d00_m6` | `replan25_d00_m6` | 40 / 136 | 10 / 106 | 2.2e-21 | 6.9e-20 | yes | +48.0 [+40.3, +53.1] |
| F2a | Replan 25 beats full chunk after a 6 cm move | ours | `full_d00_m6` | `replan25_d00_m6` | 37 / 123 | 11 / 97 | 2.4e-18 | 6.7e-17 | yes | +43.0 [+35.1, +48.4] |
| F2b | Replan 25 loses success with latency: 0 vs 400 ms | official | `replan25_d00_m0` | `replan25_d20_m0` | 150 / 116 | 53 / 19 | 7.6e-05 | 0.0014 | yes | -17.0 [-24.0, -8.6] |
| F2b | Replan 25 loses success with latency: 0 vs 400 ms | ours | `replan25_d00_m0` | `replan25_d20_m0` | 147 / 108 | 56 / 17 | 5.3e-06 | 0.00012 | yes | -19.5 [-26.1, -11.2] |
| F2c | Replan 25 is worse than full chunk at 400 ms | official | `full_d20_m0` | `replan25_d20_m0` | 138 / 116 | 51 / 29 | 0.018 | 0.24 | no | -11.0 [-19.4, -1.8] |
| F2c | Replan 25 is worse than full chunk at 400 ms | ours | `full_d20_m0` | `replan25_d20_m0` | 143 / 108 | 55 / 20 | 6.5e-05 | 0.0013 | yes | -17.5 [-24.7, -8.9] |
| F3 | Replan 10 improves with latency: 0 vs 200 ms | official | `replan10_d00_m0` | `replan10_d10_m0` | 13 / 41 | 9 / 37 | 4.1e-05 | 0.00085 | yes | +14.0 [+7.4, +18.7] |
| F3 | Replan 10 improves with latency: 0 vs 200 ms | ours | `replan10_d00_m0` | `replan10_d10_m0` | 12 / 53 | 8 / 49 | 2.7e-08 | 6.5e-07 | yes | +20.5 [+13.8, +24.9] |
| C1 | Full chunk loses success with latency: 0 vs 400 ms (checkpoint-dependent) | official | `full_d00_m0` | `full_d20_m0` | 162 / 138 | 34 / 10 | 0.00039 | 0.007 | yes | -12.0 [-17.0, -5.3] |
| C1 | Full chunk loses success with latency: 0 vs 400 ms (checkpoint-dependent) | ours | `full_d00_m0` | `full_d20_m0` | 150 / 143 | 23 / 16 | 0.34 | 1 | no | -3.5 [-9.5, +3.1] |
| C2 | TE vs full chunk, nominal (checkpoint-dependent) | official | `full_d00_m0` | `te_d00_m0` | 162 / 142 | 38 / 18 | 0.01 | 0.16 | no | -10.0 [-16.6, -2.3] |
| C2 | TE vs full chunk, nominal (checkpoint-dependent) | ours | `full_d00_m0` | `te_d00_m0` | 150 / 149 | 35 / 34 | 1 | 1 | no | -0.5 [-9.0, +8.0] |
| F4a | TE at c = -0.05 beats the default after a 6 cm move | official | `te_d00_m6` | `te-0.05_d00_m6` | 53 / 115 | 20 / 82 | 4.3e-10 | 1.1e-08 | yes | +31.0 [+21.8, +38.3] |
| F4a | TE at c = -0.05 beats the default after a 6 cm move | ours | `te_d00_m6` | `te-0.05_d00_m6` | 42 / 136 | 12 / 106 | 5.8e-20 | 1.7e-18 | yes | +47.0 [+38.8, +52.7] |
| F4b | TE at c = -0.05 keeps the latency robustness: default vs -0.05 at 400 ms | official | `te_d20_m0` | `te-0.05_d20_m0` | 136 / 143 | 18 / 25 | 0.36 | 1 | n/a (no-difference claim) | +3.5 [-3.4, +9.9] |
| F4b | TE at c = -0.05 keeps the latency robustness: default vs -0.05 at 400 ms | ours | `te_d20_m0` | `te-0.05_d20_m0` | 154 / 160 | 18 / 24 | 0.44 | 1 | n/a (no-difference claim) | +3.0 [-3.8, +9.4] |
| F4c | TE at c = -0.05 costs nominal success | official | `te_d00_m0` | `te-0.05_d00_m0` | 142 / 120 | 46 / 24 | 0.012 | 0.16 | no | -11.0 [-18.7, -2.4] |
| F4c | TE at c = -0.05 costs nominal success | ours | `te_d00_m0` | `te-0.05_d00_m0` | 149 / 133 | 43 / 27 | 0.072 | 0.72 | no | -8.0 [-16.0, +0.7] |
| F4d | TE at c = -0.05 beats replan 25 at 400 ms | official | `replan25_d20_m0` | `te-0.05_d20_m0` | 116 / 143 | 17 / 44 | 0.00073 | 0.012 | yes | +13.5 [+5.6, +20.0] |
| F4d | TE at c = -0.05 beats replan 25 at 400 ms | ours | `replan25_d20_m0` | `te-0.05_d20_m0` | 108 / 160 | 8 / 60 | 5.7e-11 | 1.5e-09 | yes | +26.0 [+19.1, +30.5] |
| F4e | TE at c = -0.05 vs replan 25 after a 6 cm move | official | `replan25_d00_m6` | `te-0.05_d00_m6` | 136 / 115 | 38 / 17 | 0.0065 | 0.1 | no | -10.5 [-17.0, -2.9] |
| F4e | TE at c = -0.05 vs replan 25 after a 6 cm move | ours | `replan25_d00_m6` | `te-0.05_d00_m6` | 123 / 136 | 22 / 35 | 0.11 | 0.89 | no | +6.5 [-1.4, +13.7] |
| F4f | TE at c = -0.05 vs replan 25, nominal | official | `replan25_d00_m0` | `te-0.05_d00_m0` | 150 / 120 | 40 / 10 | 2.4e-05 | 0.00052 | yes | -15.0 [-20.0, -8.1] |
| F4f | TE at c = -0.05 vs replan 25, nominal | ours | `replan25_d00_m0` | `te-0.05_d00_m0` | 147 / 133 | 26 / 12 | 0.034 | 0.4 | no | -7.0 [-12.3, -0.5] |
| F5a | TE at c = -0.1 loses nominal success | official | `te_d00_m0` | `te-0.1_d00_m0` | 142 / 51 | 102 / 11 | 1.3e-19 | 3.6e-18 | yes | -45.5 [-50.9, -37.6] |
| F5a | TE at c = -0.1 loses nominal success | ours | `te_d00_m0` | `te-0.1_d00_m0` | 149 / 46 | 111 / 8 | 2.5e-24 | 8.4e-23 | yes | -51.5 [-56.0, -44.2] |
| F5b | Reactivity peaks between c = 0 and -0.1: -0.05 vs -0.1 after a 6 cm move | official | `te-0.05_d00_m6` | `te-0.1_d00_m6` | 115 / 41 | 83 / 9 | 3.9e-16 | 1.1e-14 | yes | -37.0 [-41.8, -29.7] |
| F5b | Reactivity peaks between c = 0 and -0.1: -0.05 vs -0.1 after a 6 cm move | ours | `te-0.05_d00_m6` | `te-0.1_d00_m6` | 136 / 47 | 98 / 9 | 4.9e-20 | 1.5e-18 | yes | -44.5 [-49.3, -37.1] |
| F5c | Latency revives TE at c = -0.2: 0 vs 400 ms | official | `te-0.2_d00_m0` | `te-0.2_d20_m0` | 6 / 145 | 0 / 139 | 2.9e-42 | 9.8e-41 | yes | +69.5 [+65.9, +69.5] |
| F5c | Latency revives TE at c = -0.2: 0 vs 400 ms | ours | `te-0.2_d00_m0` | `te-0.2_d20_m0` | 2 / 155 | 0 / 153 | 1.8e-46 | 6.1e-45 | yes | +76.5 [+72.9, +76.5] |

## Verdict per claim

- **baseline** Retrained vs official checkpoint, full chunk, nominal: no-difference claim; paired difference pair -7.0 [-13.2, +0.1] points.
- **F1a** Temporal ensembling (TE) is insensitive to latency: 0 vs 400 ms: no-difference claim; paired difference official -3.0 [-6.2, +1.4] points; ours +2.5 [-2.4, +6.7] points.
- **F1b** TE is no better than full chunk after a 6 cm move: no-difference claim; paired difference official +6.5 [-1.1, +13.4] points; ours +2.5 [-4.2, +8.8] points.
- **F2a** Replan 25 beats full chunk after a 6 cm move: survives on both checkpoints.
- **F2b** Replan 25 loses success with latency: 0 vs 400 ms: survives on both checkpoints.
- **F2c** Replan 25 is worse than full chunk at 400 ms: survives only on ours.
- **F3** Replan 10 improves with latency: 0 vs 200 ms: survives on both checkpoints.
- **C1** Full chunk loses success with latency: 0 vs 400 ms (checkpoint-dependent): survives only on official.
- **C2** TE vs full chunk, nominal (checkpoint-dependent): does not survive.
- **F4a** TE at c = -0.05 beats the default after a 6 cm move: survives on both checkpoints.
- **F4b** TE at c = -0.05 keeps the latency robustness: default vs -0.05 at 400 ms: no-difference claim; paired difference official +3.5 [-3.4, +9.9] points; ours +3.0 [-3.8, +9.4] points.
- **F4c** TE at c = -0.05 costs nominal success: does not survive.
- **F4d** TE at c = -0.05 beats replan 25 at 400 ms: survives on both checkpoints.
- **F4e** TE at c = -0.05 vs replan 25 after a 6 cm move: does not survive.
- **F4f** TE at c = -0.05 vs replan 25, nominal: survives only on official.
- **F5a** TE at c = -0.1 loses nominal success: survives on both checkpoints.
- **F5b** Reactivity peaks between c = 0 and -0.1: -0.05 vs -0.1 after a 6 cm move: survives on both checkpoints.
- **F5c** Latency revives TE at c = -0.2: 0 vs 400 ms: survives on both checkpoints.
