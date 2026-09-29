# Temporal ensembling: weight coefficient

ACT weights the predictions for the current step by exp(−c·i), i = 0 the oldest; c = 0.01 is the default.
Success rate with a 95% Wilson interval, 200 paired seeds; p is an exact McNemar test against c = 0.01 on the same seeds.

## Official checkpoint

| c | nominal | 400 ms latency | 6 cm displacement |
|---|---|---|---|
| 0.01 (default) | 71.0% [64–77] | 68.0% [61–74] | 26.5% [21–33] |
| 0 | 71.5% [65–77], p = 1 | 70.5% [64–76], p = 0.18 | 36.0% [30–43], p = 0.00055 |
| -0.05 | 60.0% [53–67], p = 0.012 | 71.5% [65–77], p = 0.36 | 57.5% [51–64], p = 4.3e-10 |
| -0.1 | 25.5% [20–32], p = 1.3e-19 | 71.5% [65–77], p = 0.41 | 20.5% [15–27], p = 0.21 |
| -0.2 | 3.0% [1–6], p = 1.4e-38 | 72.5% [66–78], p = 0.29 | 1.5% [1–4], p = 8.1e-13 |

## Our checkpoint (100k steps)

| c | nominal | 400 ms latency | 6 cm displacement |
|---|---|---|---|
| 0.01 (default) | 74.5% [68–80] | 77.0% [71–82] | 21.0% [16–27] |
| 0 | 78.5% [72–84], p = 0.096 | 78.0% [72–83], p = 0.69 | 33.5% [27–40], p = 2.2e-05 |
| -0.05 | 66.5% [60–73], p = 0.072 | 80.0% [74–85], p = 0.44 | 68.0% [61–74], p = 5.8e-20 |
| -0.1 | 23.0% [18–29], p = 2.5e-24 | 80.0% [74–85], p = 0.46 | 23.5% [18–30], p = 0.64 |
| -0.2 | 1.0% [0–4], p = 1.1e-44 | 77.5% [71–83], p = 1 | 0.0% [0–2], p = 4.5e-13 |
