# Cross-class IoU threshold validation

**Post-freeze correctness evaluation** — không thay thế hay sửa số liệu Chapter 4 frozen.

- Protocol: dual-preresize (stretch + letterbox, PIL BILINEAR) + conf 0.3 + class-aware NMS IoU 0.55 — pool TRƯỚC cross-class suppression.
- Images: 54 dùng, 0 thiếu file.
- Detections sau class-aware NMS: 96.
- Cross-class pairs (IoU > 0): 11 — true_pair 0, candidate 11, noise 0.

## Phân bố IoU của true_pair (cả hai lớp cùng có trong GT class-set)

GT chỉ có lớp + số lượng (KHÔNG có bounding box), nên phân bố này **không phải**
bằng chứng về spatial false-suppression; nó giới hạn trên IoU của các cặp
có khả năng tương ứng hai món thật — ngưỡng phải nằm trên mức này.

- Không có cặp true_pair nào (các món thật trong val set không chồng nhau).

## Số cặp bị suppress theo ngưỡng (pairwise approximation)

| Ngưỡng | true_pair bị suppress (rủi ro) | candidate bị loại (lợi ích) | noise bị loại |
|---|---|---|---|
| 0.60 | 0 | 8 | 0 |
| 0.65 | 0 | 8 | 0 |
| 0.70 | 0 | 8 | 0 |
| 0.75 | 0 | 8 | 0 |
| 0.80 | 0 | 8 | 0 |
| 0.90 | 0 | 8 | 0 |

## Chi tiết candidate (duplicate) tại ngưỡng 0.70 trở lên

| Ảnh | IoU | lớp giữ | conf giữ | lớp bị loại | conf bị loại |
|---|---|---|---|---|---|
| BunRieu-CanhBun/bun-rieu (1).jpg | 0.988 | Bun bo Hue (Hue beef noodle soup) | 0.96 | Bun rieu (Crab noodle soup) | 0.35 |
| BunRieu-CanhBun/bun-rieu1 (1).jpg | 0.986 | Bun rieu (Crab noodle soup) | 0.85 | Bun bo Hue (Hue beef noodle soup) | 0.54 |
| Pho/Pho1 (3).jpg | 0.984 | Pho (Vietnamese noodle soup) | 0.94 | Bun bo Hue (Hue beef noodle soup) | 0.34 |
| BanhMi/banh-mi250g (2).jpg | 0.978 | Banh mi (Vietnamese baguette sandwich) | 0.96 | Hamburger | 0.90 |
| BanhMi/banh-mi250g (3).jpg | 0.972 | Hamburger | 0.96 | Banh mi (Vietnamese baguette sandwich) | 0.89 |
| SupCua/SupCua1 (2).jpg | 0.971 | Sup cua (Crab soup) | 0.92 | Canh (Soup) | 0.46 |
| BunRieu-CanhBun/bun-rieu (3).jpg | 0.967 | Bun bo Hue (Hue beef noodle soup) | 0.94 | Bun rieu (Crab noodle soup) | 0.55 |
| BunRieu-CanhBun/bun-rieu1 (4).jpg | 0.963 | Bun rieu (Crab noodle soup) | 0.96 | Bun bo Hue (Hue beef noodle soup) | 0.94 |

Records: `scratch/cross_class_val_records.jsonl` (11 pairs).
