# Results summary

## ours_100000: success rate vs delay (95% Wilson CI)

| mode | 0 ms | 40 ms | 100 ms | 200 ms | 400 ms |
|---|---|---|---|---|---|
| full chunk | 75.0% [69–80] | 76.0% [70–81] | 73.5% [67–79] | 72.5% [66–78] | 71.5% [65–77] |
| replan every 50 | 80.5% [74–85] | 83.0% [77–88] | 85.5% [80–90] | 84.0% [78–88] | 74.0% [68–80] |
| replan every 25 | 73.5% [67–79] | 75.0% [69–80] | 73.0% [66–79] | 59.0% [52–66] | 54.0% [47–61] |
| replan every 10 | 6.0% [3–10] | 6.0% [3–10] | 11.0% [7–16] | 26.5% [21–33] | 12.0% [8–17] |
| temporal ensembling | 74.5% [68–80] | 75.0% [69–80] | 76.0% [70–81] | 76.0% [70–81] | 77.0% [71–82] |

Paired against full chunk on the same seeds (exact McNemar p; wins = only this mode succeeds):

| mode | level | n | wins | losses | p |
|---|---|---|---|---|---|
| replan every 50 | 0 ms | 200 | 30 | 19 | 0.152 |
| replan every 50 | 40 ms | 200 | 32 | 18 | 0.0649 |
| replan every 50 | 100 ms | 200 | 40 | 16 | 0.00184 |
| replan every 50 | 200 ms | 200 | 39 | 16 | 0.00267 |
| replan every 50 | 400 ms | 200 | 32 | 27 | 0.603 |
| replan every 25 | 0 ms | 200 | 28 | 31 | 0.795 |
| replan every 25 | 40 ms | 200 | 29 | 31 | 0.897 |
| replan every 25 | 100 ms | 200 | 34 | 35 | 1 |
| replan every 25 | 200 ms | 200 | 31 | 58 | 0.00554 |
| replan every 25 | 400 ms | 200 | 20 | 55 | 6.49e-05 |
| replan every 10 | 0 ms | 200 | 1 | 139 | 2.02e-40 |
| replan every 10 | 40 ms | 200 | 1 | 141 | 5.13e-41 |
| replan every 10 | 100 ms | 200 | 2 | 127 | 2.46e-35 |
| replan every 10 | 200 ms | 200 | 14 | 106 | 1.16e-18 |
| replan every 10 | 400 ms | 200 | 3 | 122 | 1.53e-32 |
| temporal ensembling | 0 ms | 200 | 34 | 35 | 1 |
| temporal ensembling | 40 ms | 200 | 31 | 33 | 0.901 |
| temporal ensembling | 100 ms | 200 | 33 | 28 | 0.609 |
| temporal ensembling | 200 ms | 200 | 35 | 28 | 0.45 |
| temporal ensembling | 400 ms | 200 | 34 | 23 | 0.185 |

Failure stages (delay): episodes that failed, by the highest reward reached (0 no contact · 1 touched · 2 lifted · 3 handover attempted)

| mode | level | 0 | 1 | 2 | 3 |
|---|---|---|---|---|---|
| full chunk | 0 ms | 7 | 15 | 28 | 0 |
| full chunk | 40 ms | 5 | 17 | 26 | 0 |
| full chunk | 100 ms | 9 | 13 | 31 | 0 |
| full chunk | 200 ms | 5 | 13 | 37 | 0 |
| full chunk | 400 ms | 3 | 21 | 33 | 0 |
| replan every 50 | 0 ms | 14 | 5 | 20 | 0 |
| replan every 50 | 40 ms | 11 | 5 | 18 | 0 |
| replan every 50 | 100 ms | 11 | 4 | 14 | 0 |
| replan every 50 | 200 ms | 9 | 8 | 15 | 0 |
| replan every 50 | 400 ms | 10 | 8 | 34 | 0 |
| replan every 25 | 0 ms | 22 | 1 | 30 | 0 |
| replan every 25 | 40 ms | 18 | 4 | 28 | 0 |
| replan every 25 | 100 ms | 19 | 3 | 32 | 0 |
| replan every 25 | 200 ms | 15 | 7 | 60 | 0 |
| replan every 25 | 400 ms | 13 | 6 | 73 | 0 |
| replan every 10 | 0 ms | 163 | 1 | 24 | 0 |
| replan every 10 | 40 ms | 158 | 1 | 29 | 0 |
| replan every 10 | 100 ms | 147 | 1 | 30 | 0 |
| replan every 10 | 200 ms | 22 | 3 | 122 | 0 |
| replan every 10 | 400 ms | 13 | 11 | 152 | 0 |
| temporal ensembling | 0 ms | 17 | 18 | 16 | 0 |
| temporal ensembling | 40 ms | 15 | 18 | 17 | 0 |
| temporal ensembling | 100 ms | 15 | 17 | 16 | 0 |
| temporal ensembling | 200 ms | 13 | 19 | 16 | 0 |
| temporal ensembling | 400 ms | 13 | 20 | 13 | 0 |

## ours_100000: success rate vs displacement (95% Wilson CI)

| mode | 0 cm | 2 cm | 4 cm | 6 cm |
|---|---|---|---|---|
| full chunk | 75.0% [69–80] | 65.0% [58–71] | 42.5% [36–49] | 18.5% [14–24] |
| replan every 50 | 80.5% [74–85] | 70.0% [63–76] | 47.0% [40–54] | 26.5% [21–33] |
| replan every 25 | 73.5% [67–79] | 78.0% [72–83] | 73.0% [66–79] | 61.5% [55–68] |
| replan every 10 | 6.0% [3–10] | 3.0% [1–6] | 3.0% [1–6] | 3.5% [2–7] |
| temporal ensembling | 74.5% [68–80] | 64.5% [58–71] | 47.0% [40–54] | 21.0% [16–27] |

Paired against full chunk on the same seeds (exact McNemar p; wins = only this mode succeeds):

| mode | level | n | wins | losses | p |
|---|---|---|---|---|---|
| replan every 50 | 0 cm | 200 | 30 | 19 | 0.152 |
| replan every 50 | 2 cm | 200 | 28 | 18 | 0.184 |
| replan every 50 | 4 cm | 200 | 28 | 19 | 0.243 |
| replan every 50 | 6 cm | 200 | 29 | 13 | 0.0195 |
| replan every 25 | 0 cm | 200 | 28 | 31 | 0.795 |
| replan every 25 | 2 cm | 200 | 47 | 21 | 0.00219 |
| replan every 25 | 4 cm | 200 | 82 | 21 | 1.07e-09 |
| replan every 25 | 6 cm | 200 | 97 | 11 | 2.39e-18 |
| replan every 10 | 0 cm | 200 | 1 | 139 | 2.02e-40 |
| replan every 10 | 2 cm | 200 | 2 | 126 | 4.85e-35 |
| replan every 10 | 4 cm | 200 | 3 | 82 | 5.3e-21 |
| replan every 10 | 6 cm | 200 | 7 | 37 | 5.3e-06 |
| temporal ensembling | 0 cm | 200 | 34 | 35 | 1 |
| temporal ensembling | 2 cm | 200 | 30 | 31 | 1 |
| temporal ensembling | 4 cm | 200 | 38 | 29 | 0.328 |
| temporal ensembling | 6 cm | 200 | 23 | 18 | 0.533 |

Failure stages (displacement): episodes that failed, by the highest reward reached (0 no contact · 1 touched · 2 lifted · 3 handover attempted)

| mode | level | 0 | 1 | 2 | 3 |
|---|---|---|---|---|---|
| full chunk | 0 cm | 7 | 15 | 28 | 0 |
| full chunk | 2 cm | 22 | 19 | 29 | 0 |
| full chunk | 4 cm | 43 | 12 | 60 | 0 |
| full chunk | 6 cm | 72 | 10 | 81 | 0 |
| replan every 50 | 0 cm | 14 | 5 | 20 | 0 |
| replan every 50 | 2 cm | 23 | 8 | 29 | 0 |
| replan every 50 | 4 cm | 47 | 11 | 48 | 0 |
| replan every 50 | 6 cm | 75 | 6 | 66 | 0 |
| replan every 25 | 0 cm | 22 | 1 | 30 | 0 |
| replan every 25 | 2 cm | 19 | 1 | 24 | 0 |
| replan every 25 | 4 cm | 14 | 1 | 39 | 0 |
| replan every 25 | 6 cm | 23 | 3 | 51 | 0 |
| replan every 10 | 0 cm | 163 | 1 | 24 | 0 |
| replan every 10 | 2 cm | 162 | 3 | 29 | 0 |
| replan every 10 | 4 cm | 166 | 5 | 23 | 0 |
| replan every 10 | 6 cm | 168 | 1 | 24 | 0 |
| temporal ensembling | 0 cm | 17 | 18 | 16 | 0 |
| temporal ensembling | 2 cm | 26 | 23 | 22 | 0 |
| temporal ensembling | 4 cm | 51 | 17 | 38 | 0 |
| temporal ensembling | 6 cm | 90 | 17 | 51 | 0 |

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
