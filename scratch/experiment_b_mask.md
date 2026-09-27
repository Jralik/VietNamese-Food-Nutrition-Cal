
# Experiment B — Mask ablation results (20 items / 16 ảnh)

Frozen: images/detections/depth/ArUco/density 0.45/volume logic. CHỈ thay mask. Variants: otsu (iteration-1), fixed60 (gray>60), fixed60cc/fixed80cc (+largest CC), cconly (largest CC của mask gốc).

| Variant | area ratio (median) | volume ratio (median) | mass ratio (median) | Bánh mì MAPE (16 ảnh) |
|---|---|---|---|---|
| original | 1.00 | 1.00 | 1.00 | 84.8% |
| otsu | 0.32 | 0.42 | 0.42 | 93.0% |
| fixed60 | 0.45 | 0.61 | 0.61 | 80.9% |
| fixed60cc | 0.20 | 0.43 | 0.43 | 87.7% |
| fixed80cc | 0.20 | 0.36 | 0.36 | 89.2% |
| cconly | 0.89 | 0.91 | 0.92 | 61.6% |

GT totals: 99.3g (BM_Trung, 2 phần thường) · 242.6g (banh-mi1-5) · 255.6g (250g) · 210g (DucTri) — bánh mì thịt.

## Per-image Banh mi mass totals (g) — original vs các variant

| Ảnh | GT | original | otsu | fixed60 | fixed60cc | fixed80cc | cconly |
|---|---|---|---|---|---|---|---|
| banh-mi1.jpg | 243 | 225 | 3 | 4 | 3 | 3 | 225 |
| banh-mi2.jpg | 243 | 54 | 44 | 47 | 28 | 28 | 41 |
| banh-mi3.jpg | 243 | 502 | 197 | 284 | 217 | 150 | 221 |
| banh-mi4.jpg | 243 | 176 | 18 | 39 | 12 | 12 | 152 |
| banh-mi5.jpg | 243 | 298 | 52 | 1 | 0 | 0 | 261 |
| banh-mi250g (1).jpg | 256 | 276 | 74 | 123 | 26 | 26 | 276 |
| banh-mi250g (2).jpg | 256 | 220 | 97 | 114 | 92 | 92 | 164 |
| banh-mi250g (3).jpg | 256 | 294 | 6 | 7 | 5 | 5 | 294 |
| banh-mi250g (4).jpg | 256 | 448 | 37 | 36 | 21 | 22 | 411 |
| BanhMiDucTri (1).jpg | 210 | 257 | 228 | 240 | 244 | 240 | 250 |
| BanhMiDucTri (2).jpg | 210 | 89 | 91 | 89 | 89 | 89 | 89 |
| BanhMiDucTri (3).jpg | 210 | 121 | 112 | 118 | 118 | 113 | 120 |
| BanhMi_Trung (4).jpg | 99 | 465 | 275 | 322 | 282 | 286 | 310 |
| BanhMi_Trung (1).jpg | 99 | 332 | 350 | 296 | 215 | 215 | 230 |
| BanhMi_Trung (2).jpg | 99 | 233 | 319 | 179 | 287 | 279 | 151 |
| BanhMi_Trung (3).jpg | 99 | 241 | 60 | 94 | 40 | 40 | 305 |

