
# Phase D — 54-image validation (single protocol, frozen metrics)

Matching: class-level GT **with per-class box counts**, greedy one-to-one per class by confidence (matched_c = min(n_gt_c, n_det_c)). Raw-only recovery: upright-missed target class with a raw box of the correct class at >= T and IoU <= 0.7 vs every upright box. Inference once at conf=0.05; thresholds applied locally on shared pools. CPU runtime. v10b-dual+raw is kept as a documented ablation — the app path is dual-only (raw branch rejected: all audited raw-only boxes were position FPs).

| Threshold | Config | Recall | FP/Extras | 0-box | Same-class dup | Cross-class overlap | Raw-only recovery | Latency ms/img |
|---|---|---|---|---|---|---|---|---|
| 0.50 | v26m-dual | 0.514 (38/74) | 10 | 12/54 | 0 | 2 | n/a | 839 |
| 0.40 | v26m-dual | 0.527 (39/74) | 13 | 8/54 | 0 | 2 | n/a | 839 |
| 0.30 | v26m-dual | 0.541 (40/74) | 19 | 4/54 | 0 | 4 | n/a | 839 |
| 0.50 | v10b-dual | 0.730 (54/74) | 23 | 2/54 | 0 | 5 | n/a | 1270 |
| 0.40 | v10b-dual | 0.743 (55/74) | 27 | 1/54 | 0 | 6 | n/a | 1270 |
| 0.30 | v10b-dual | 0.811 (60/74) | 36 | 0/54 | 0 | 8 | n/a | 1270 |
| 0.50 | v10b-dual+raw | 0.743 (55/74) | 23 | 2/54 | 0 | 5 | 1/1 | 1737 |
| 0.40 | v10b-dual+raw | 0.757 (56/74) | 27 | 1/54 | 0 | 6 | 1/1 | 1737 |
| 0.30 | v10b-dual+raw | 0.824 (61/74) | 38 | 0/54 | 0 | 10 | 1/1 | 1737 |

## Per-class recall @0.50 (hits/targets)

| Class | v26m-dual | v10b-dual | v10b-dual+raw |
|---|---|---|---|
| Banh cuon (Rolled rice pancake) | 0/6 | 0/6 | 0/6 |
| Banh mi (Vietnamese baguette sandwich) | 16/20 | 19/20 | 19/20 |
| Bun bo Hue (Hue beef noodle soup) | 5/5 | 5/5 | 5/5 |
| Bun rieu (Crab noodle soup) | 4/7 | 6/7 | 6/7 |
| Canh (Soup) | 0/3 | 1/3 | 1/3 |
| Com tam (Broken rice) | 6/6 | 6/6 | 6/6 |
| Pho (Vietnamese noodle soup) | 5/5 | 5/5 | 5/5 |
| Rau (Vegetables) | 0/8 | 4/8 | 4/8 |
| Trung (Egg) | 0/4 | 1/4 | 2/4 |
| Sup cua (Crab soup) | 2/10 | 7/10 | 7/10 |

## Per-class recall @0.30 (v10b-dual+raw)

| Class | hits/targets |
|---|---|
| Banh cuon (Rolled rice pancake) | 0/6 |
| Banh mi (Vietnamese baguette sandwich) | 20/20 |
| Bun bo Hue (Hue beef noodle soup) | 5/5 |
| Bun rieu (Crab noodle soup) | 7/7 |
| Canh (Soup) | 1/3 |
| Com tam (Broken rice) | 6/6 |
| Pho (Vietnamese noodle soup) | 5/5 |
| Rau (Vegetables) | 6/8 |
| Trung (Egg) | 3/4 |
| Sup cua (Crab soup) | 8/10 |
