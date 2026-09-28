# Results summary

## official: success rate vs delay (95% Wilson CI)

| mode | 0 ms | 40 ms | 100 ms | 200 ms | 400 ms |
|---|---|---|---|---|---|
| full chunk | 81.0% [75–86] | 79.5% [73–85] | 80.0% [74–85] | 76.0% [70–81] | 69.0% [62–75] |
| replan every 50 | 74.0% [68–80] | 73.0% [66–79] | 74.5% [68–80] | 69.0% [62–75] | 66.5% [60–73] |
| replan every 25 | 75.0% [69–80] | 72.0% [65–78] | 69.5% [63–75] | 65.0% [58–71] | 58.0% [51–65] |
| replan every 10 | 6.5% [4–11] | 7.0% [4–11] | 12.5% [9–18] | 20.5% [15–27] | 15.0% [11–21] |
| temporal ensembling | 71.0% [64–77] | 71.0% [64–77] | 70.0% [63–76] | 70.0% [63–76] | 68.0% [61–74] |

Paired against full chunk on the same seeds (exact McNemar p; wins = only this mode succeeds):

| mode | level | n | wins | losses | p |
|---|---|---|---|---|---|
| replan every 50 | 0 ms | 200 | 17 | 31 | 0.0595 |
| replan every 50 | 40 ms | 200 | 23 | 36 | 0.117 |
| replan every 50 | 100 ms | 200 | 24 | 35 | 0.193 |
| replan every 50 | 200 ms | 200 | 24 | 38 | 0.098 |
| replan every 50 | 400 ms | 200 | 29 | 34 | 0.615 |
| replan every 25 | 0 ms | 200 | 19 | 31 | 0.119 |
| replan every 25 | 40 ms | 200 | 20 | 35 | 0.0581 |
| replan every 25 | 100 ms | 200 | 20 | 41 | 0.00985 |
| replan every 25 | 200 ms | 200 | 25 | 47 | 0.0128 |
| replan every 25 | 400 ms | 200 | 29 | 51 | 0.0183 |
| replan every 10 | 0 ms | 200 | 1 | 150 | 1.06e-43 |
| replan every 10 | 40 ms | 200 | 2 | 147 | 3.13e-41 |
| replan every 10 | 100 ms | 200 | 5 | 140 | 2.32e-35 |
| replan every 10 | 200 ms | 200 | 7 | 118 | 3.98e-27 |
| replan every 10 | 400 ms | 200 | 7 | 115 | 2.68e-26 |
| temporal ensembling | 0 ms | 200 | 18 | 38 | 0.0105 |
| temporal ensembling | 40 ms | 200 | 20 | 37 | 0.0331 |
| temporal ensembling | 100 ms | 200 | 13 | 33 | 0.00453 |
| temporal ensembling | 200 ms | 200 | 19 | 31 | 0.119 |
| temporal ensembling | 400 ms | 200 | 23 | 25 | 0.885 |

Failure stages (delay): episodes that failed, by the highest reward reached (0 no contact · 1 touched · 2 lifted · 3 handover attempted)

| mode | level | 0 | 1 | 2 | 3 |
|---|---|---|---|---|---|
| full chunk | 0 ms | 3 | 15 | 20 | 0 |
| full chunk | 40 ms | 5 | 10 | 26 | 0 |
| full chunk | 100 ms | 4 | 8 | 28 | 0 |
| full chunk | 200 ms | 3 | 11 | 34 | 0 |
| full chunk | 400 ms | 3 | 6 | 53 | 0 |
| replan every 50 | 0 ms | 8 | 5 | 39 | 0 |
| replan every 50 | 40 ms | 4 | 8 | 42 | 0 |
| replan every 50 | 100 ms | 4 | 9 | 38 | 0 |
| replan every 50 | 200 ms | 6 | 7 | 49 | 0 |
| replan every 50 | 400 ms | 8 | 12 | 47 | 0 |
| replan every 25 | 0 ms | 9 | 0 | 41 | 0 |
| replan every 25 | 40 ms | 10 | 1 | 45 | 0 |
| replan every 25 | 100 ms | 10 | 4 | 47 | 0 |
| replan every 25 | 200 ms | 8 | 7 | 55 | 0 |
| replan every 25 | 400 ms | 6 | 14 | 64 | 0 |
| replan every 10 | 0 ms | 148 | 3 | 36 | 0 |
| replan every 10 | 40 ms | 135 | 4 | 47 | 0 |
| replan every 10 | 100 ms | 115 | 1 | 59 | 0 |
| replan every 10 | 200 ms | 10 | 5 | 144 | 0 |
| replan every 10 | 400 ms | 12 | 11 | 147 | 0 |
| temporal ensembling | 0 ms | 23 | 18 | 17 | 0 |
| temporal ensembling | 40 ms | 25 | 13 | 20 | 0 |
| temporal ensembling | 100 ms | 22 | 17 | 21 | 0 |
| temporal ensembling | 200 ms | 24 | 20 | 16 | 0 |
| temporal ensembling | 400 ms | 20 | 17 | 27 | 0 |

## official: success rate vs displacement (95% Wilson CI)

| mode | 0 cm | 2 cm | 4 cm | 6 cm |
|---|---|---|---|---|
| full chunk | 81.0% [75–86] | 65.5% [59–72] | 41.5% [35–48] | 20.0% [15–26] |
| replan every 50 | 74.0% [68–80] | 70.5% [64–76] | 51.0% [44–58] | 35.5% [29–42] |
| replan every 25 | 75.0% [69–80] | 73.5% [67–79] | 76.5% [70–82] | 68.0% [61–74] |
| replan every 10 | 6.5% [4–11] | 8.0% [5–13] | 7.5% [5–12] | 6.5% [4–11] |
| temporal ensembling | 71.0% [64–77] | 68.5% [62–75] | 51.5% [45–58] | 26.5% [21–33] |

Paired against full chunk on the same seeds (exact McNemar p; wins = only this mode succeeds):

| mode | level | n | wins | losses | p |
|---|---|---|---|---|---|
| replan every 50 | 0 cm | 200 | 17 | 31 | 0.0595 |
| replan every 50 | 2 cm | 200 | 31 | 21 | 0.212 |
| replan every 50 | 4 cm | 200 | 37 | 18 | 0.0145 |
| replan every 50 | 6 cm | 200 | 45 | 14 | 6.53e-05 |
| replan every 25 | 0 cm | 200 | 19 | 31 | 0.119 |
| replan every 25 | 2 cm | 200 | 38 | 22 | 0.0519 |
| replan every 25 | 4 cm | 200 | 88 | 18 | 2.97e-12 |
| replan every 25 | 6 cm | 200 | 106 | 10 | 2.16e-21 |
| replan every 10 | 0 cm | 200 | 1 | 150 | 1.06e-43 |
| replan every 10 | 2 cm | 200 | 5 | 120 | 1.15e-29 |
| replan every 10 | 4 cm | 200 | 11 | 79 | 7.77e-14 |
| replan every 10 | 6 cm | 200 | 12 | 39 | 0.000198 |
| temporal ensembling | 0 cm | 200 | 18 | 38 | 0.0105 |
| temporal ensembling | 2 cm | 200 | 28 | 22 | 0.48 |
| temporal ensembling | 4 cm | 200 | 50 | 30 | 0.033 |
| temporal ensembling | 6 cm | 200 | 33 | 20 | 0.0984 |

Failure stages (displacement): episodes that failed, by the highest reward reached (0 no contact · 1 touched · 2 lifted · 3 handover attempted)

| mode | level | 0 | 1 | 2 | 3 |
|---|---|---|---|---|---|
| full chunk | 0 cm | 3 | 15 | 20 | 0 |
| full chunk | 2 cm | 24 | 17 | 28 | 0 |
| full chunk | 4 cm | 45 | 11 | 61 | 0 |
| full chunk | 6 cm | 86 | 7 | 67 | 0 |
| replan every 50 | 0 cm | 8 | 5 | 39 | 0 |
| replan every 50 | 2 cm | 17 | 7 | 35 | 0 |
| replan every 50 | 4 cm | 25 | 15 | 58 | 0 |
| replan every 50 | 6 cm | 44 | 7 | 78 | 0 |
| replan every 25 | 0 cm | 9 | 0 | 41 | 0 |
| replan every 25 | 2 cm | 8 | 5 | 40 | 0 |
| replan every 25 | 4 cm | 5 | 4 | 38 | 0 |
| replan every 25 | 6 cm | 12 | 1 | 51 | 0 |
| replan every 10 | 0 cm | 148 | 3 | 36 | 0 |
| replan every 10 | 2 cm | 155 | 2 | 27 | 0 |
| replan every 10 | 4 cm | 146 | 6 | 33 | 0 |
| replan every 10 | 6 cm | 152 | 4 | 31 | 0 |
| temporal ensembling | 0 cm | 23 | 18 | 17 | 0 |
| temporal ensembling | 2 cm | 25 | 17 | 21 | 0 |
| temporal ensembling | 4 cm | 47 | 16 | 34 | 0 |
| temporal ensembling | 6 cm | 88 | 15 | 44 | 0 |
