# Cross-class duplicate suppression — post-freeze correctness fix report

**Nhãn: post-freeze correctness evaluation.** Báo cáo này KHÔNG thay thế hay sửa
bất kỳ số liệu nào của Chapter 4 frozen baseline (commit `18cd148`). Mọi số liệu
frozen (recall 0.811, MAPE, ablation) giữ nguyên; các phép đo dưới đây được chạy
riêng, cùng protocol, và được ghi rõ là sau-freeze.

---

## 1. Bằng chứng bug (đã tái hiện)

Ảnh `BunRieu-CanhBun/bun-rieu (1).jpg` (1800×4000), conf 0.30 (mặc định app):

| Nguồn box | Lớp | Conf | Box (ảnh gốc) |
|---|---|---|---|
| Lượt stretch | Bun bo Hue | 0.955 | (12, 988, 1215, 2080) |
| Lượt letterbox | Bun bo Hue | 0.750 | (21, 984, 1214, 2078) |
| Lượt letterbox | **Bun rieu** | 0.354 | (21, 984, 1214, 2078) — **trùng hệt** |

- Sau class-aware NMS (IoU 0.55), cả Bun bo Hue 0.955 lẫn Bun rieu 0.354 sống sót
  với **IoU giữa hai box = 0.988**.
- Hai box cùng match vào **một** ước lượng khối lượng (`_find_volume_estimation`,
  IoU > 0.5) → hai dòng kết quả giống hệt nhau (661 ± 87 g, 711 cm³, 594.9 kcal…)
  và `total_nutrition` **cộng dinh dưỡng của tô này hai lần**.
- Lưu ý: duplicate tồn tại cả ở conf cao — `banh-mi250g (2).jpg` có Hamburger 0.90
  trùng vùng với Banh mi 0.96 — nên chỉ tăng ngưỡng confidence không giải quyết được.

## 2. Nguyên nhân gốc

1. **Bên trong một lượt dự đoán**: NMS nội bộ của ultralytics là class-aware
   (`agnostic=False`) → cùng một tô, hai nghi ngờ lớp khác nhau đều sống sót.
2. **Lớp gộp của app**: `_nms_class_aware` (IoU 0.55) chỉ so sánh box cùng lớp →
   duplicate khác lớp không bao giờ bị loại.
3. **Hậu xử lý không có dedup**: volume pipeline segment riêng từng detection
   (1 box = 1 prompt = 1 estimation); hai box trùng → hai estimation giống hệt nhau.

## 3. Fix: cross-class duplicate suppression (class-agnostic NMS)

Vị trí: ngay sau `_nms_class_aware` trong `predict_with_dual_preresize` (utils.py)
— điểm chặn duy nhất trước volume matching.

```
if IoU(box_i, box_j) >= 0.70 and class_i != class_j and conf_i < conf_j:
    loại box_i (ghi audit record)
```

- Greedy theo thứ tự confidence giảm dần → kết quả độc lập thứ tự input.
- Cặp cùng lớp vẫn do `_nms_class_aware` quản lý (không đổi behavior cũ).
- Tham số: `CROSS_CLASS_SUPPRESS_ENABLED = True`, `CROSS_CLASS_SUPPRESS_IOU = 0.70`
  (pipeline_config.py); `predict_with_dual_preresize` nhận override
  `suppress_cross_class` / `suppress_cross_class_iou` (OFF dùng để tái lập baseline).
- Audit trail: mỗi box bị loại được ghi `{class_name, confidence, bbox, kept_class,
  kept_confidence, iou, reason: "cross_class_duplicate"}` — gắn attribute
  `suppressed_cross_class` lên Results, lưu vào `st.session_state` và key
  `"suppressed_cross_class"` trong JSON export. Không có thay đổi UI.

## 4. Chọn ngưỡng 0.70 — threshold validation

Nguồn: `cross_class_threshold_validation.md` (cùng thư mục), script
`scripts/validate_cross_class_threshold.py`, 54/54 ảnh val, conf 0.30, pool
TRƯỚC suppression: 96 detections, 11 cặp khác lớp có chồng lấn (IoU > 0).

- **true_pair (cả hai lớp cùng có trong GT class-set): 0 cặp** — phân bố rỗng.
- **candidate (đúng một lớp trong GT): 11 cặp; 8 cặp có IoU ≥ 0.70** (thực tế
  tất cả ≥ 0.963), rơi đúng vào các cặp nhầm lẫn của model: Bun bo Hue ↔ Bun rieu,
  Pho ↔ Bun bo Hue, Banh mi ↔ Hamburger, Sup cua ↔ Canh.
- Ở mọi ngưỡng quét 0.60–0.90: **0 true_pair bị suppress, 8 candidate bị loại**.

**Diễn giải cho thesis**: ngưỡng ≥ mức tối đa của IoU giữa các cặp mà cả hai lớp
cùng xuất hiện trong GT class-set, nhằm hạn chế suppress các cặp có khả năng
tương ứng hai món thật; **đây không phải bằng chứng về spatial false-suppression
do GT không có bounding box**. 0.70 được chọn theo kết quả validation và trade-off
duplicate-removal: dải duplicate quan sát nằm tại IoU ≥ 0.96, còn vùng chồng lấn
"liều" của hai món thật là 0.4–0.6, nên 0.70 cách an toàn cả hai phía.

## 5. Kiểm thử

**Unit test** (`tests/test_phase6_cross_class_suppression.py`, deterministic,
không cần weights) — **8/8 PASS**: duplicate IoU 1.0 khác lớp bị loại; IoU 0.65
(< ngưỡng) được giữ; cặp cùng lớp vẫn do class-aware NMS xử lý; chuỗi 3 box cùng
vùng deterministic theo thứ tự input; box rời rạc giữ hết; audit record đúng
giá trị (không chỉ đủ keys); config hợp lệ; pool bug thật (`bun-rieu (1).jpg`)
sụp về đúng 1 box.

**E2E regression** (`tests/test_phase7_e2e_duplicate_regression.py`, slow,
cần weights) — **7/7 ảnh PASS** trên `BunRieu-CanhBun/`, invariant số:

- `bun-rieu (1).jpg`: đúng **1** food box, lớp Bun bo Hue, conf > 0.5;
  1 duplicate vào audit trail.
- Toàn bộ ảnh: không còn cặp food box khác lớp với IoU ≥ 0.70.
  Tổng cộng 4 duplicate bị suppress trong folder (bun-rieu (1), (3), bun-rieu1 (1), (4)).
- Skip-guard môi trường: thiếu thư mục ảnh hoặc weights → SKIPPED có lý do;
  ảnh tồn tại nhưng sai assertion → FAIL.

## 6. Đánh giá tác động ON/OFF — cùng protocol tuyệt đối

Script `scripts/eval_post_freeze_on_off.py`. Hai lượt chạy **giữ nguyên tuyệt đối**
model, weights, conf 0.30, dual-preresize, class-aware NMS 0.55, val set, GT parser,
metric — chỉ flag suppression khác nhau (OFF = `suppress_cross_class=False`).

| Mode | GT instances | Det instances | Matched | Recall | Precision | Suppressed |
|---|---|---|---|---|---|---|
| OFF | 74 | 96 | 60 | **0.8108** | 0.6250 | 0 |
| ON  | 74 | 88 | 57 | **0.7703** | **0.6477** | 8 |

- Lượt OFF tái lập đúng frozen baseline: **60/74 = 0.811** → protocol khớp, phép
  so sánh có giá trị attribution.
- ON loại đúng 8 duplicate (khớp validation), precision +2.3 điểm %.
- **Recall instance-level giảm 3 instances (60 → 57)** — truy vết từng ảnh cho thấy
  cả 3 rơi vào các ảnh **một vùng, hai nhãn nhập nhằng**, nơi bị suppress là lớp
  đúng-GT nhưng có conf thấp hơn lớp sai giữ lại:

| Ảnh | Bị suppress (conf) | Giữ lại | GT | Nhận định |
|---|---|---|---|---|
| bun-rieu (1).jpg | Bun rieu (0.35) | Bun bo Hue 0.96 | Bun rieu | **GT gán sai** — người dùng xác nhận món thật là Bun bo Hue; fix đưa output về đúng thực tế |
| bun-rieu (3).jpg | Bun rieu (0.55) | Bun bo Hue 0.94 | có Bun rieu | model nhầm nhóm Bún (vùng vẫn giữ 1 nhãn) |
| banh-mi250g (3).jpg | Banh mi (0.89) | Hamburger 0.96 | Banh mi | model nhầm Banh mi ↔ Hamburger trên vùng bánh mì |

- Đọc đúng của con số này: metric instance-level **thưởng cho duplicate** — ở 3 ảnh
  này, lượt OFF được tính match nhờ đếm **cả hai nhãn trên cùng một vùng** (một nhãn
  sai là false positive) trong khi vẫn **double-count dinh dưỡng** của vùng đó. Ở
  cấp vùng (thứ nutrition quan tâm): **0 vùng hai món bị gộp** (validation: 0
  true_pair), 8 vùng duplicate được đếm đúng một lần, mỗi vùng giữ đúng một nhãn.
- Kết luận: fix chấp nhận giảm recall "ảo" trên các ảnh nhập nhằng để đổi lấy
  correctness dinh dưỡng (mỗi vùng = một khẩu phần) và precision cao hơn.

## 7. Ghi chú kỹ thuật môi trường (ràng buộc import)

`pipeline_config` gọi `torch.cuda.is_available()` lúc import. Trên tiến trình bare
Python (ngoài Streamlit), **mọi lời gọi CUDA API trước predict YOLO đầu tiên gây
segfault** (đã đo: chỉ `torch.cuda.is_available()` cũng đủ; repro R2/R2b/R2c),
còn import **sau** predict đầu tiên là an toàn (repro R4: predict#2 vẫn chạy).
Vì vậy:

- `predict_with_dual_preresize` import `pipeline_config` **lazily, sau hai lượt
  predict** — các eval script chạy bare không bị ảnh hưởng.
- `tests/test_phase7` đọc threshold **sau** lời gọi predict đầu tiên, không import
  ở đầu module.
- App Streamlit không đổi behavior: nó đã import `pipeline_config` từ startup
  (main.py → `render_sidebar_ai_config`) từ trước khi fix này xuất hiện.

## 8. Baseline guard

- Thay đổi production: `utils.py` (+96/−1: `_suppress_cross_class_duplicates`,
  2 tham số mới, audit trong `detect_image_result`), `pipeline_config.py` (+18:
  2 constants) — additive, không đụng logic frozen khác.
- File mới: 2 script (`scripts/`), 2 test (`tests/`), báo cáo + records
  (`evaluation_final/post_freeze/`, `scratch/*.jsonl`).
- Không amend `18cd148`; không sửa `evaluation_final/` (file cũ),
  `scratch/evaluation_summary.md`, hay bất kỳ số liệu Chapter 4 nào.
