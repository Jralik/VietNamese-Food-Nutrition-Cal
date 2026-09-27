
# Model comparison — v26m (app) vs yolov10b, 54 anh val

| Config | Recall@0.50 | Recall@0.30 | Anh 0 box @0.50 | ms/anh (CPU) |
|---|---|---|---|---|
| v26m-dual (app hien tai) | 0.546 | 0.546 | 12/54 | 1089 |
| v10b-dual | 0.759 | 0.759 | 2/54 | 1574 |
| v10b-stretch (ref pipeline) | 0.611 | 0.611 | 6/54 | 735 |

## Recall@0.50 theo nhom mon

| Nhom | v26m-dual (app hien tai) | v10b-dual | v10b-stretch (ref pipeline) |
|---|---|---|---|
| BanhCuon | 0.00 | 0.00 | 0.00 |
| BanhMi | 0.75 | 0.91 | 0.69 |
| BunBoHue | 0.50 | 0.90 | 0.50 |
| BunRieu-CanhBun | 0.43 | 0.71 | 0.50 |
| ComSuon | 1.00 | 1.00 | 1.00 |
| Pho | 1.00 | 1.00 | 1.00 |
| SupCua | 0.20 | 0.70 | 0.60 |

## Recall@0.50 theo class (tren cac anh co class do)

| Class | So anh | v26m-dual (app hien tai) | v10b-dual | v10b-stretch (ref pipeline) |
|---|---|---|---|---|
| Banh cuon (Rolled rice pancake) | 6 | 0/6 | 0/6 | 0/6 |
| Banh mi (Vietnamese baguette sandwich) | 16 | 14/16 | 16/16 | 12/16 |
| Bun bo Hue (Hue beef noodle soup) | 5 | 5/5 | 5/5 | 5/5 |
| Bun rieu (Crab noodle soup) | 7 | 4/7 | 6/7 | 4/7 |
| Com tam (Broken rice) | 5 | 5/5 | 5/5 | 5/5 |
| Pho (Vietnamese noodle soup) | 5 | 5/5 | 5/5 | 5/5 |
| Rau (Vegetables) | 8 | 0/8 | 4/8 | 0/8 |
| Trung (Egg) | 4 | 0/4 | 1/4 | 1/4 |
| Sup cua (Crab soup) | 10 | 2/10 | 7/10 | 6/10 |

## Near-miss: class bi miss @0.50 nhung co box o conf thap hon

| Config | Class | So anh miss | median max-conf | max max-conf | Sẽ bat duoi slider 0.30? |
|---|---|---|---|---|---|
| v26m-dual (app hien tai) | Rau (Vegetables) | 8 | 0.000 | 0.196 | Khong |
| v26m-dual (app hien tai) | Sup cua (Crab soup) | 8 | 0.000 | 0.094 | Khong |
| v26m-dual (app hien tai) | Banh cuon (Rolled rice pancake) | 6 | 0.000 | 0.000 | Khong |
| v26m-dual (app hien tai) | Trung (Egg) | 4 | 0.000 | 0.090 | Khong |
| v26m-dual (app hien tai) | Bun rieu (Crab noodle soup) | 3 | 0.000 | 0.000 | Khong |
| v26m-dual (app hien tai) | Banh mi (Vietnamese baguette sandwich) | 2 | 0.396 | 0.476 | Co |
| v10b-dual | Banh cuon (Rolled rice pancake) | 6 | 0.000 | 0.105 | Khong |
| v10b-dual | Rau (Vegetables) | 4 | 0.234 | 0.387 | Khong |
| v10b-dual | Trung (Egg) | 3 | 0.215 | 0.408 | Khong |
| v10b-dual | Sup cua (Crab soup) | 3 | 0.290 | 0.338 | Khong |
| v10b-dual | Bun rieu (Crab noodle soup) | 1 | 0.354 | 0.354 | Co |
| v10b-stretch (ref pipeline) | Rau (Vegetables) | 8 | 0.033 | 0.231 | Khong |
| v10b-stretch (ref pipeline) | Banh cuon (Rolled rice pancake) | 6 | 0.000 | 0.000 | Khong |
| v10b-stretch (ref pipeline) | Banh mi (Vietnamese baguette sandwich) | 4 | 0.208 | 0.295 | Khong |
| v10b-stretch (ref pipeline) | Sup cua (Crab soup) | 4 | 0.068 | 0.338 | Khong |
| v10b-stretch (ref pipeline) | Trung (Egg) | 3 | 0.198 | 0.408 | Khong |
| v10b-stretch (ref pipeline) | Bun rieu (Crab noodle soup) | 3 | 0.000 | 0.000 | Khong |
