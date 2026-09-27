# Báo Cáo Thực Nghiệm Đánh Giá Khẩu Phần Trên Tập Ảnh Đo Đạc Thực Tế (Phase 4 Real-Image Benchmark)

## 1. Tóm Tắt Kết Quả Toàn Thể (Dataset-Wide Metrics)

- **Tổng số ảnh thực nghiệm có Ground Truth**: 54 ảnh (100% cân đo thực tế, phân bổ trên 7 nhóm món ăn).
- **Mục tiêu thực nghiệm**: Đánh giá khả năng bảo toàn pipeline CV sau khi tích hợp Nutrition Layer (Pha 3), định lượng sai số portion estimation và kiểm toán phân bổ lỗi (Error Attribution) qua hai giao thức đo đạc.
- **Thời gian thực thi trung bình**: 34.63 giây/ảnh (PyTorch CUDA 12.1, SAM2 + Depth Anything V2 + ArUco).

---

## 2. Bảng Thống Kê So Sánh Giữa Hai Giao Thức Đánh Giá (Dual-Protocol Evaluation)

Để phản ánh chính xác năng lực thực tế của hệ thống và tránh thiên kiến do nhiễu đối tượng phụ trong ảnh, benchmark báo cáo song song theo 2 giao thức:
- **Giao thức 1 (Protocol 1 - Full-Image Aggregation)**: Tính tổng khối lượng mọi bounding box được YOLO phát hiện trong ảnh ($M_{\text{pred}} = \sum_i m_i$).
- **Giao thức 2 (Protocol 2 - Target-Matched Estimation)**: Chỉ tính khối lượng của các đối tượng thuộc món ăn mục tiêu cần đối soát với Ground Truth ($M_{\text{pred, target}} = m_{\text{target}}$).

| Danh Mục Món Ăn | Số Lượng Ảnh | Ground Truth TB ($m_{\text{GT}}$) | Protocol 1: Full-Image MAPE (%) | Protocol 2: Target-Matched MAPE (%) | Mức Độ Cải Thiện Sai Số |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Bún riêu (`BunRieu-CanhBun`)** | 7 | 868.9 g | **18.4%** | **17.3%** | +1.1% |
| **Bún bò Huế (`BunBoHue`)** | 5 | 894.5 g | **16.0%** | **19.4%** | -3.4% |
| **Bánh mì (`BanhMi`)** | 16 | 259.2 g | **34.5%** | **43.3%** | -8.8% |
| **Cơm sườn (`ComSuon`)** | 5 | 390.5 g | **127.1%** | **15.4%** | **+111.6% (Loại bỏ chén canh phụ)** |
| **Phở (`Pho`)** | 5 | 819.5 g | **45.8%** | **46.1%** | -0.3% |
| **Bánh cuốn (`BanhCuon`)** | 6 | 315.5 g | **67.5%** | **67.5%** | +0.0% |
| **Súp cua (`SupCua`)** | 10 | 482.6 g | **75.5%** | **69.5%** | +6.1% |
| **TOÀN BỘ BENCHMARK (Overall)** | **54 ảnh** | — | **51.6%** | **42.9%** | **Giảm 8.7 điểm phần trăm (-16.9% tương đối)** |

*Ghi chú số liệu chi tiết theo Protocol 1:*
- MAE tổng thể: **232.5 gam** | RMSE tổng thể: **354.1 gam**.
- Bún bò Huế: Group CV = `0.209` | Bún riêu: Group CV = `0.138`.

---

## 3. Phân Rã & Phân Tích Chuyên Sâu Từng Nhóm Món (Chương 4 Luận Văn)

### 3.1. Phân tích hiện tượng Cơm sườn: Bản chất của sự chênh lệch giữa 127.1% và 15.4%
Khi phân rã chi tiết 5 bức ảnh Cơm sườn:
- **`com-suon1 (4).jpg`**: GT = 390.5 g $\rightarrow$ Dự đoán = 426.0 g (**Sai số chỉ 9.1%**).
- **`com-suon1 (2).jpg`**: GT = 390.5 g $\rightarrow$ Target-matched = 431.2 g (**Sai số chỉ 10.4%**, Full-image = 1334g / 241.6% do chén canh đi kèm).
- **`com-suon1 (3).jpg`**: GT = 390.5 g $\rightarrow$ Dự đoán = 434.3 g (**Sai số chỉ 11.2%**).
- **`com-suon1 (5).jpg`**: GT = 390.5 g $\rightarrow$ Target-matched = 475.5 g (**Sai số chỉ 21.8%**, Full-image = 1752g / 348.8% do chén canh đi kèm).
- **`com-suon1 (1).jpg`**: GT = 390.5 g $\rightarrow$ Dự đoán = 486.9 g (**Sai số chỉ 24.7%**).

**Kết luận khoa học**:
1. Trong các trường hợp đối tượng mục tiêu được phân lập đúng trên đĩa (Protocol 2), toàn bộ 5 ảnh Cơm sườn đều đạt sai số tương đối thấp, dao động từ **9.1% đến 24.7%**, đạt **MAPE trung bình là 15.4%**.
2. Con số MAPE 127.1% ở Protocol 1 không phải là lỗi ước lượng hình học 3D của đĩa cơm sườn, mà là kết quả của **Evaluation Protocol Mismatch & Side-Dish Interference**: trong ảnh có thêm chén canh súp đi kèm được YOLO phát hiện và nhân bản bounding box, khiến việc cộng dồn toàn bộ ảnh bị lệch so với Ground Truth vốn chỉ cân riêng đĩa cơm sườn.

### 3.2. Phân tích Phở: Nút thắt tại Detector thượng tầng (Upstream Object Detection)
Trong 5 ảnh Phở:
- **`Pho1 (1).jpg`**: YOLO nhận diện chính xác class `Pho` $\rightarrow$ Dự đoán đạt **782.9 g** so với Ground Truth **819.5 g** (**Sai số cực thấp: 4.5%**).
- **`Pho1 (2).jpg`**: YOLO phân loại nhầm (`Misclassification`) sang `Bun mam (Fermented fish noodle soup)`. Do cùng là tô nước nên khối lượng ước lượng vẫn đạt $789.4\text{ g}$ (sai số 3.7%), nhưng trên giao diện người dùng tìm kiếm "Phở" sẽ không thấy.
- **`Pho1 (3).jpg`**: YOLO bỏ sót toàn bộ tô phở (`False Negative / Partial Detection`), chỉ bắt được cụm sợi bánh phở bên trong thành nhãn `Bun` ($64.9\text{ g}$, sai số 92.1%).
- **`Pho1 (4).jpg`**: YOLO bị phân mảnh (`Fragmentation`), tách nước dùng thành `Canh` ($1200\text{g}$) và sợi phở thành `Bun` ($64.8\text{g}$), làm tổng khối lượng đội lên $1264.8\text{ g}$ (sai số 54.3%).
- **`Pho1 (5).jpg`**: YOLO bỏ sót tô phở, chỉ nhận diện `Bun` + `Rau` ($211.3\text{ g}$, sai số 74.2%).

**Kết luận khoa học**:
Nút thắt của nhóm Phở nằm ở **tầng phát hiện đối tượng (Upstream YOLO Detection & Semantic Decomposition)** do bề mặt nước dùng phản chiếu và sự tương đồng về mặt thị giác giữa các món nước, chứ không phải do mô hình chiều sâu (Depth Anything V2) hay tích phân thể tích 3D tính sai.

---

## 4. Hệ Thống Phân Loại Nguồn Gốc Sai Số Hai Tầng (Two-Tier Causal Error Attribution Taxonomy)

Dựa trên việc kiểm toán chi tiết 54 bức ảnh, các nguồn sai số trong toàn bộ pipeline được phân rã thành chuỗi lan truyền nhân quả:
$$E_{\text{detection}} \longrightarrow E_{\text{segmentation}} \longrightarrow E_{\text{geometry/volume}} \longrightarrow E_{\text{mass}} \longrightarrow E_{\text{nutrition}}$$

*Quy ước kiểm toán (Attribution Protocol)*:
- **Tầng 1 (Upstream Target Detection Status)**: Phân loại loại trừ tương hỗ (Mutually exclusive: mỗi ảnh có đúng 1 trạng thái, tổng = 54 ảnh = 100%).
- **Tầng 2 (Downstream / Evaluation Interference)**: Phân loại đa nhãn (Multi-label factors: mỗi ảnh có 0 hoặc nhiều yếu tố nhiễu; các tỷ lệ % không cộng thành 100%).

### Bảng Thống Kê Phân Rã Lỗi Hai Tầng:

| Tầng | Mã Lỗi | Tên Trạng Thái / Yếu Tố Nhiễu | Số Ảnh | Tỷ Lệ % | Mô Tả Kỹ Thuật & Tác Động Trong Pipeline |
| :---: | :---: | :--- | :---: | :---: | :--- |
| **Tầng 1** | **`D1`** | **Correct Target Detected** | **34** | **63.0%** | YOLO bắt đúng đối tượng mục tiêu; downstream 3D pipeline ước lượng bình thường |
| **Tầng 1** | **`D2`** | **Misclassified** | **12** | **22.2%** | Nhầm món tương đồng (Phở $\rightarrow$ Bún mắm, Súp cua $\rightarrow$ Canh) |
| **Tầng 1** | **`D3`** | **Missed / Partial Detection** | **5** | **9.3%** | Bỏ sót khung tô/đĩa chính, chỉ bắt mẩu nhỏ thành phần (sợi bún/thịt) |
| **Tầng 1** | **`D4`** | **Fragmented** | **1** | **1.9%** | Tách một món thống nhất thành nhiều nhãn generic (Nước $\rightarrow$ Canh, Sợi $\rightarrow$ Bún) |
| **Tầng 1** | **`D5`** | **Duplicate Detection** | **2** | **3.7%** | Trùng lặp bounding box do ngưỡng NMS chưa triệt tiêu hoàn toàn |
| *Tổng T1* | — | *Tổng trạng thái Tầng 1* | *54* | *100.0%* | *Tổng đúng 54/54 ảnh benchmark* |
| **Tầng 2** | **`E1`** | **Auxiliary Side-Dish Interference** | **10** | **18.5%** | Xuất hiện đĩa rau, chén nước chấm, chén canh súp phụ đi kèm món chính |
| **Tầng 2** | **`E2`** | **Target-Association Mismatch** | **2** | **3.7%** | Lệch phạm vi đối soát giữa Ground Truth (1 đĩa) và Bounding Box (cả bàn ăn) |
| **Tầng 2** | **`E3`** | **Geometric / Depth Error** | **6** | **11.1%** | Suy giảm độ sâu đơn ảnh trên vật thể dạng dẹt phẳng (Thickness loss ở Bánh cuốn) |

---

## 5. Lan Truyền Sai Số Dinh Dưỡng Tuyến Tính (P4-C)

- Về mặt mô hình toán học, với hệ số dinh dưỡng thành phần cố định $c_N = \frac{N_{100g}}{100}$:
  $$|\hat{N} - N_{\text{GT}}| = c_N |\hat{m} - m_{\text{GT}}|$$
- Quan hệ này được xác nhận khớp 100% trên toàn bộ các ảnh thực nghiệm, đảm bảo tính nhất quán nội tại của Nutrition Layer và cho phép truy nguyên nguồn gốc sai số dinh dưỡng về trực tiếp các failure modes của thị giác máy tính đã phân tích ở Mục 4.

---

## 6. Kết Luận Nghiệm Thu Pha 4

1. **Bảo toàn hồi quy (P4-A)**: Zero regression drift trên toàn bộ pipeline CV và bảng tỷ trọng `DENSITY_DB`.
2. **Khảo sát khẩu phần thực tế (P4-B)**: Hoàn thành 100% (54/54 ảnh) với đầy đủ dữ liệu đo đạc thực tế.
3. **Truy nguyên sai số dinh dưỡng (P4-C)**: Bảo đảm tính tuyến tính và tính toán nhất quán.
4. **Kiểm toán phân bổ lỗi & Giao thức kép (P4-D)**: Phân tách rõ ràng Protocol 1 (Full-image: 51.6%) và Protocol 2 (Target-matched: 42.9%, Cơm sườn 15.4%), tạo cơ sở bảo vệ vững chắc trước hội đồng.
5. **ĐÓNG BĂNG BASELINE (P1 → P4)**: Toàn bộ CSDL, ma trận mapping, baseline CV và kết quả thực nghiệm chính thức được đóng băng để làm cơ sở đối sánh cho khóa luận.