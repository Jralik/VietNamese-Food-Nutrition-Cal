import docx
import shutil

# Load docx
doc = docx.Document(r'd:\Document\Year4_Semester1\KhoaLuanTotNghiep\FoodDetector-2 - V3\KLTN_Chuong1-3_bien_tap_ngon_ngu.docx')

# Helper to safely replace text in a run-based paragraph with math
# For paragraphs without math, we can safely set runs or paragraph text while keeping style
def set_para_text(p, text):
    # Preserve paragraph formatting/style
    style = p.style
    p.text = text
    p.style = style

# 1. P[290]: Has 3 math elements (m_i, N_i, sigma_i)
# Children: [pPr], run[0], run[1], oMath[mi], run[2], run[3], oMath[Ni], run[4], run[5], oMath[sigma_i], run[6]
p290 = doc.paragraphs[290]
p290.runs[0].text = (
    "Cho ảnh I (RGB, kích thước H×W) chụp bữa ăn từ góc trên, cùng với tham chiếu kích thước: "
    "dấu mốc tham chiếu ArUco (marker ArUco) có cạnh thực L = 30 mm đặt trong cùng mặt phẳng đế với thức ăn "
    "(giả định mặt phẳng ngang - flat surface assumption). Trong trường hợp không có marker ArUco trong ảnh, "
    "hệ thống không thực hiện ước lượng thể tích hình học (nhằm tránh sai số lớn từ các phương pháp phỏng đoán theo đĩa) "
    "mà tự động chuyển sang cơ chế tính toán dinh dưỡng theo khẩu phần ăn mẫu chuẩn (mục 3.8.2). "
    "Camera được mô tả bằng ma trận nội tại K và hệ số méo D thu từ hiệu chỉnh một lần (mục 3.8.1). "
    "Bài toán được phát biểu: ước lượng cho mỗi món ăn i xuất hiện trong I một khối lượng "
)

# 2. P[291]: Has 7 math elements
# run[0]: 'Chuỗi suy luận được phân rã thành: phát hiện các món '
# oMath: bi,yi,ci
# run[2]: ' (bounding box, nhãn, confidence); phân đoạn mask '
# oMath: Mi
# run[4]: '; bản đồ độ sâu '
# oMath: Z(u,v)
# run[6]: '; khôi phục tham chiếu tỷ lệ '
# oMath: s
# run[8]: ' (mm/pixel); thể tích '
# oMath: Vi=f(Mi,Z,s)
# run[10]: '; khối lượng '
# oMath: mi=ρ(yi)·Vi
# run[12]: '; dinh dưỡng '
# oMath: Ni=(mi/100)·N100(yi)
# run[14]: '. Đầu vào/đầu ra...'
p291 = doc.paragraphs[291]
p291.runs[2].text = " (hộp giới hạn (bounding box), nhãn lớp, độ tin cậy (confidence)); phân đoạn mặt nạ (mask) "
p291.runs[14].text = (
    ". Đầu vào/đầu ra của từng giai đoạn được xác định tường minh trong các mục 3.4–3.11. "
    "Các giả định và giới hạn của mô hình: mặt phẳng ngang; thức ăn nhìn thấy được từ trên "
    "(phần bị che/chìm là sai số hệ thống); marker phẳng; biến thiên khối lượng riêng giữa các quán "
    "được mô hình hóa bằng phân bố theo lớp món. Khi không có marker ArUco, chuỗi suy luận hình học "
    "(khôi phục tỷ lệ, thể tích, khối lượng) được thay thế trực tiếp bằng việc tra cứu khối lượng khẩu phần ăn mẫu chuẩn."
)

# 3. P[293] (No math):
set_para_text(doc.paragraphs[293], (
    "Cách tiếp cận của đề tài kết hợp hai nhóm phương pháp bổ trợ nhau. "
    "Nhóm hình học chiếu - hiệu chỉnh camera, phát hiện marker, giải bài toán Perspective-n-Point (solvePnP), "
    "phép biến đổi phối cảnh (homography) - chịu trách nhiệm khôi phục tỷ lệ theo đơn vị thực (metric); "
    "đặc điểm của nhóm này là tính xác định cao, sai số có thể kiểm soát và kiểm định được "
    "(sai số chiếu lại - reprojection error, sai số định vị marker). "
    "Nhóm học sâu - phát hiện đối tượng, phân đoạn, ước lượng độ sâu từ ảnh đơn - đảm nhiệm các bước "
    "mà hình học thuần không giải được (nhận diện lớp món, ranh giới thức ăn, độ sâu bề mặt); "
    "đặc điểm của nhóm này là sai số mang tính thống kê, phải được lượng hóa và truyền xuyên suốt quy trình xử lý. "
    "Trục kết hợp là bước cân chỉnh tỷ lệ (Scale Alignment): tham chiếu theo phương ngang (mm/pixel) từ nhóm hình học "
    "được dùng để hiệu chuẩn bản đồ độ sâu của nhóm học sâu (hiệu chỉnh độ nổi - relief, mục 3.9.1), "
    "biến hai nguồn thông tin độc lập thành một mô hình 3D nhất quán. "
    "Vấn đề sự mơ hồ về tỷ lệ kích thước được giải quyết chính xác tại bước này, không phải ở từng mô hình riêng lẻ."
))

# 4. P[296] (No math):
set_para_text(doc.paragraphs[296], (
    "Luồng học sâu (GPU): ảnh đầu vào → phát hiện món ăn (3.4) → phân đoạn thực phẩm theo gợi ý (prompt) hộp giới hạn (3.5) → ước lượng độ sâu cho ảnh đơn (3.7)."
))

# 5. P[297] (No math):
set_para_text(doc.paragraphs[297], (
    "Luồng hình học (CPU, song song): hiệu chỉnh camera một lần (3.8.1); trên từng ảnh - phát hiện marker ArUco (3.8.2), suy ra phép biến đổi phối cảnh và tỷ lệ mm/pixel."
))

# 6. P[298] (No math):
set_para_text(doc.paragraphs[298], (
    "Hai luồng hội tụ tại bước cân chỉnh tỷ lệ và thể tích (3.9): mặt nạ + độ sâu + tỷ lệ → "
    "tích phân cột theo mô hình hình học của từng loại món → thể tích. Tiếp theo là chuỗi tuyến tính: "
    "khối lượng theo khối lượng riêng (3.10) → dinh dưỡng theo bảng thành phần thực phẩm (3.11) → "
    "phân tích cá nhân theo bộ máy quy tắc (rule engine, 3.12) → diễn đạt lời khuyên bằng mô hình ngôn ngữ lớn "
    "kèm sinh tăng cường truy xuất (Retrieval-Augmented Generation - RAG, 3.13). "
    "Kết quả từng bước đều gắn nhãn nguồn tham chiếu (aruco_calibrated hoặc reference_serving_fallback) "
    "và mức độ tin cậy (cao/dự phòng), phục vụ phân tích sai số và cảnh báo người dùng. "
    "Khi ảnh không có marker ArUco, hệ thống không sử dụng phương pháp phỏng đoán theo đĩa (plate heuristic) "
    "do tính thiếu chính xác cao, mà tự động chuyển sang tính toán dinh dưỡng dựa trên khẩu phần ăn mẫu chuẩn kèm cảnh báo hiển thị rõ ràng."
))

# 7. P[301] (No math):
set_para_text(doc.paragraphs[301], (
    "Thiết kế hai luồng song song có hai lý do. Một, các bước hình học nhẹ về tính toán nên không cạnh tranh tài nguyên GPU "
    "với chuỗi mô hình học sâu. Hai, tách luồng giúp phân tích sai số theo nguồn (RQ2): sai số tỷ lệ thuộc luồng hình học, "
    "sai số mặt nạ/độ sâu thuộc luồng học sâu - có thể đo lường riêng biệt bằng cách thay thế từng thành phần. "
    "Kiến trúc cho phép cấu hình linh hoạt từng mô-đun (thay bộ mã hóa độ sâu, thay mô hình phân đoạn) "
    "mà không phá vỡ giao diện dữ liệu giữa các bước - điều kiện cần cho các thí nghiệm so sánh tại Chương 4."
))

# 8. P[303]: Has 1 math element (bi = ...)
p303 = doc.paragraphs[303]
p303.runs[2].text = (
    ". Vai trò của mô-đun trong quy trình xử lý là hai mặt: cung cấp nhãn lớp cho tra cứu khối lượng riêng và dinh dưỡng; "
    "cung cấp hộp giới hạn làm gợi ý không gian cho phân đoạn."
)

# 9. P[304] (No math):
set_para_text(doc.paragraphs[304], (
    "Theo thiết kế của đề tài, hai trọng số phát hiện trên cùng bộ dữ liệu huấn luyện VietFood67 được sử dụng, "
    "phân vai theo môi trường chạy: (i) YOLOv10b (PyTorch, điểm kiểm tra huấn luyện riêng của dự án) là trọng số vận hành chính "
    "của ứng dụng giao diện web, chạy trên CPU do tiến trình phân đoạn và độ sâu sở hữu GPU; "
    "(ii) YOLO26 (ONNX) được dùng trong bộ chạy đánh giá theo lô và các tập lệnh kiểm thử - "
    "sự song song này cho phép so sánh hai kiến trúc trên cùng dữ liệu (RQ3, đánh giá tại mục 4.4 và 4.8)."
))

# 10. P[305] (No math):
set_para_text(doc.paragraphs[305], (
    "Đường suy luận vận hành được thiết kế chống lỗi trùng lặp và méo khung hình: mỗi ảnh được tiền xử lý kép "
    "(dual-preresize - một bản kéo dãn toàn khung hình và một bản đệm viền giữ nguyên tỷ lệ letterbox, nội suy tuyến tính song hướng BILINEAR), "
    "hai bản được suy luận riêng rồi gộp qua phép triệt tiêu phi cực đại theo lớp NMS (IoU 0,55) trước khi qua một bước khử trùng lặp liên lớp không phân biệt nhãn: "
    "hai hộp giới hạn khác lớp có IoU ≥ 0,70 thì hộp có độ tin cậy thấp hơn bị loại - ngưỡng được chọn giữa dải trùng lặp thật (IoU ≥ 0,963) "
    "và vùng chồng lấn hai món thật (~0,4–0,6), đã được kiểm định trên hơn 100 ảnh với 0 cặp món thật bị đè ở mọi ngưỡng 0,60–0,90 (trình bày tại mục 4.12). "
    "Ngưỡng tin cậy hiển thị mặc định 0,30 (qua thanh trượt người dùng); đường dẫn thể tích dùng ngưỡng riêng 0,40 để tránh ước lượng trên phát hiện yếu."
))

# 11. P[308]: Has 1 math element (Mi)
p308 = doc.paragraphs[308]
p308.runs[0].text = "Đầu vào: ảnh và các hộp giới hạn từ mục 3.4. Đầu ra: mặt nạ nhị phân "
p308.runs[2].text = (
    " cho từng món. Mô-đun được thiết kế theo nguyên lý phân đoạn theo gợi ý (promptable segmentation, mục 2.2.1): "
    "hộp giới hạn của phát hiện được truyền làm gợi ý hộp cho bộ giải mặt nạ, nhờ đó ranh giới phân đoạn nhất quán "
    "với kết quả phát hiện và không cần nhánh phân loại ngữ nghĩa độc lập."
)

# 12. P[309] (No math):
set_para_text(doc.paragraphs[309], (
    "Hậu xử lý mặt nạ gồm ba bước chuẩn hóa: (i) xói viền (boundary erosion) bằng nhân kernel hình elip 3×3 để tách phần viền đĩa/bát lem vào mặt nạ; "
    "(ii) kiểm tra tỷ lệ mặt nạ trên hộp giới hạn - mặt nạ phải chiếm hơn 2/3 diện tích hộp giới hạn "
    "(phát hiện món trong hộp nhưng mặt nạ quá nhỏ là tín hiệu phân đoạn hỏng); "
    "(iii) cờ cảnh báo mặt nạ quá lớn - mặt nạ chiếm trên 95% diện tích hộp giới hạn cho thấy phân đoạn có thể tràn sang nền hoặc mặt bàn. "
    "Mặt nạ kết quả được ánh xạ ngược về tọa độ ảnh gốc đầy đủ, phục vụ tích phân thể tích."
))

# 13. P[310] (No math):
set_para_text(doc.paragraphs[310], (
    "Thiết kế này giữ giao diện mô-đun trung lập với phương pháp phân đoạn: mọi ứng viên (FoodSAM, MobileSAM, SAM 2) "
    "đều được đóng gói phía sau cùng giao diện đầu vào (ảnh và gợi ý hộp giới hạn) và đầu ra (mặt nạ cùng cảnh báo hậu xử lý), "
    "cho phép hoán đổi và so sánh công bằng (cùng gợi ý, cùng ảnh kiểm thử - điều kiện so sánh công bằng được nêu trong đề cương)."
))

# 14. P[314] (No math):
set_para_text(doc.paragraphs[314], (
    "Vòng phân tích triệt tiêu (ablation) đầy đủ (mở rộng - nếu tiến độ cho phép): bổ sung SAM 2 [19] làm ứng viên thứ ba, "
    "so sánh ba phương pháp trên tập ảnh có giá trị chuẩn đối chiếu (ground truth) mặt nạ thủ công, đo mIoU, thời gian suy luận, bộ nhớ GPU "
    "và - quan trọng nhất - ảnh hưởng của từng phương pháp đến sai số thể tích ở tầng các bước xử lý phía sau "
    "(đo bằng cách chạy cùng một quy trình xử lý với từng mặt nạ, so sánh sai số khối lượng trên các mẫu có cân thực). "
    "Tiêu chí các bước xử lý phía sau là quan trọng nhất vì mục tiêu cuối của quy trình xử lý là khối lượng chính xác, không phải mặt nạ đẹp."
))

# 15. P[319] (No math):
set_para_text(doc.paragraphs[319], (
    "Ghi chú thực trạng cài đặt: tính đến thời điểm viết báo cáo, hai trong ba ứng viên đã được tích hợp hoàn chỉnh ở vị thế hệ thống nền (backend) hoán đổi được: "
    "SAM 2.1 (biến thể hiera-small, chạy trong môi trường chính của dự án, là hệ thống nền mặc định đã xuyên suốt các đợt kiểm định) "
    "và FoodSAM (cầu nối tiến trình con subprocess tới môi trường riêng của bộ FoodSAM: SAM 2.1 AMG kết hợp nhánh ngữ nghĩa SETR-MLA trên FoodSeg103, "
    "kèm chế độ tách thành phần nguyên liệu - trình bày tại mục 4.10). MobileSAM chưa được tích hợp làm hệ thống nền. "
    "Việc xác nhận chính thức phương pháp phân đoạn (so sánh mIoU, tốc độ, ảnh hưởng các bước xử lý phía sau) vẫn chờ vòng kiểm tra và phân tích triệt tiêu tại Chương 4, "
    "đúng theo quy trình của mục này; ghi chú cài đặt hiện tại KHÔNG đồng nghĩa với kết luận lựa chọn."
))

# 16. P[321] (No math):
set_para_text(doc.paragraphs[321], (
    "Đầu vào: ảnh màu RGB. Đầu ra: bản đồ độ sâu Z(u, v) theo milimet (sau bước chuyển đổi đơn vị). "
    "Hệ thống sử dụng Depth Anything V2 [12] theo ưu tiên: biến thể đo lường trong nhà theo đơn vị thực (metric indoor) - "
    "điểm kiểm tra chính thức trên Hugging Face, các kích cỡ bộ mã hóa vits (mặc định, ~25 triệu tham số, phù hợp bộ nhớ GPU 6 GB) / vitb / vitl "
    "có thể hoán đổi qua cấu hình; độ phân giải đầu vào 518 px; khoảng cách hiệu dụng bị chặn tại 20 m (bối cảnh trong nhà). "
    "Việc dùng bản metric indoor dựa trên điều kiện chụp tiêu chuẩn của đề tài (bàn ăn trong nhà, khoảng cách 0,5–1 m). "
    "Phương án dự phòng đã thiết kế: nếu kiểm định cho thấy bản metric không ổn định trên ảnh cận cảnh thực phẩm, "
    "hệ thống chuyển sang bản tương đối (relative) và khớp hai hệ số a, b của quan hệ Z = a/(d + b) (công thức 2.1) "
    "từ các điểm neo (anchor) có độ sâu thực - bốn góc marker ArUco có vector tịnh tiến tvec từ thuật toán solvePnP; "
    "cần tối thiểu hai điểm neo khác nhau để xác định hai hệ số."
))

# 17. P[322] (No math):
set_para_text(doc.paragraphs[322], (
    "Mô-đun độ sâu còn đảm nhiệm việc ước lượng tham số nội tại khi không có tệp hiệu chỉnh: "
    "đọc tiêu cự từ dữ liệu đặc tả ảnh EXIF nếu ảnh lưu trữ đầy đủ, hoặc dùng quy tắc xấp xỉ f ≈ max(W, H) cho camera điện thoại phổ thông; "
    "khi tệp hiệu chỉnh chuẩn (camera_calib.npz từ mục 3.8.1) tồn tại, tham số chuẩn được ưu tiên tuyệt đối."
))

# 18. P[323] (No math):
set_para_text(doc.paragraphs[323], (
    "Hai giới hạn được thiết kế xử lý tường minh: nhiễu độ sâu trên bề mặt chất lỏng hoặc phản chiếu - "
    "được giảm nhẹ bằng khớp mặt phẳng tham chiếu kiên cố (3.9.1) và mô hình hình học theo loại món (3.9.3); "
    "sai số hệ thống nhân của độ sâu đơn vị thực - được hiệu chỉnh bởi cơ chế tái cân chỉnh điểm neo tại 3.9.1. "
    "Phần thức ăn không nhìn thấy được (chìm trong bát) không được mô hình độ sâu khôi phục và thuộc các mô hình hình học bổ sung tại 3.9.3."
))

# 19. P[325] (No math): CORE REWRITE OF 3.8
set_para_text(doc.paragraphs[325], (
    "Mục tiêu của mô-đun: xác định tham chiếu chuyển đổi từ điểm ảnh sang milimet (pixel → mm) cho từng ảnh, "
    "kèm nhãn nguồn và mức độ tin cậy. Mô-đun được thiết kế dựa trên nguyên tắc phương pháp luận bắt buộc: "
    "\"không suy luận thể tích trên dữ liệu tỷ lệ không xác định\". "
    "Để đảm bảo tính chính xác và loại bỏ các nguồn sai số không kiểm soát được, hệ thống kiên quyết không áp dụng "
    "các phương pháp phỏng đoán kích thước dựa trên đĩa/bát (plate heuristic). "
    "Thay vào đó, chuỗi xử lý chỉ sử dụng tham chiếu kích thước thực có kiểm chứng từ marker ArUco; "
    "trong trường hợp không phát hiện được marker, hệ thống không thực hiện ước lượng thể tích mà kích hoạt cơ chế dự phòng "
    "tính toán dinh dưỡng dựa trên khẩu phần ăn mẫu chuẩn (reference serving size). "
    "Kết quả luôn đi kèm nhãn nguồn tham chiếu (aruco_marker hoặc reference_serving_fallback) và mức độ tin cậy tương ứng "
    "để người dùng nắm rõ cơ sở tính toán."
))

# 20. P[327] (No math):
set_para_text(doc.paragraphs[327], (
    "Trước khi đưa hệ thống vào sử dụng, camera được hiệu chỉnh bằng bảng bàn cờ 9×7 ô (kích thước ô 30 mm), "
    "chụp tối thiểu 20 ảnh từ nhiều góc và khoảng cách; thuật toán calibrateCamera giải ma trận nội tại K và hệ số méo D. "
    "Chất lượng hiệu chỉnh được kiểm soát bằng sai số chiếu lại (reprojection error) với ngưỡng tham chiếu < 0,5 pixel; "
    "kết quả được lưu vào tệp chuẩn (npz/JSON) và được nạp tự động ở mọi lần chạy. "
    "Hiệu chỉnh chuẩn giúp bài toán solvePnP và ước lượng tiêu cự đạt độ chính xác cao - "
    "hai thành phần quyết định độ tin cậy của toàn chuỗi hình học."
))

# 21. P[329] (No math):
set_para_text(doc.paragraphs[329], (
    "Cấp 1 - marker ArUco (độ tin cậy cao): thuật toán phát hiện dấu mốc detectMarkers theo từ điển DICT_4X4_50 "
    "(tự động thử thêm các từ điển phổ biến nếu không tìm thấy), marker có kích thước cạnh đen thực tế L = 30 mm "
    "(khớp đúng quy ước góc của OpenCV); hàm estimatePoseSingleMarkers giải tư thế camera thu được vector quay rvec và vector tịnh tiến tvec. "
    "Phép biến đổi phối cảnh (homography) từ bốn cặp góc cho phép tính tỷ lệ chuyển đổi cục bộ và diện tích điểm ảnh theo luật phối cảnh (mục 3.9.2). "
    "Kết quả được gán nhãn aruco_marker với mức độ tin cậy cao (high)."
))

# 22. P[330] (No math):
set_para_text(doc.paragraphs[330], (
    "Cấp 2 - bảng bàn cờ hiệu chuẩn (độ tin cậy cao): khi marker ArUco không có trong khung hình nhưng bảng bàn cờ hiệu chuẩn 30 mm xuất hiện "
    "(trong bối cảnh chụp tại khu vực cố định có bố trí bảng hiệu chuẩn), hệ thống phát hiện các góc bàn cờ và sử dụng kích thước cạnh ô "
    "làm tham chiếu đo lường tương đương với marker."
))

# 23. P[331] (No math): CORE REWRITE OF PLATE HEURISTIC REJECTION
set_para_text(doc.paragraphs[331], (
    "Lý do loại bỏ phương pháp phỏng đoán theo đĩa (plate heuristic): "
    "Trong các nghiên cứu trước đây và đề xuất ban đầu, phương pháp phỏng đoán kích thước dựa trên đĩa/bát (plate heuristic) "
    "từng được xem xét bằng cách giả định kích thước đế chuẩn (bát phở ~180 mm, đĩa cơm ~220 mm...) kết hợp khớp hình elip trên mép đĩa "
    "để suy ra tỷ lệ phối cảnh. Tuy nhiên, qua quá trình kiểm thử thực nghiệm, nhóm tác giả nhận thấy phương pháp này hoàn toàn không chính xác và thiếu tin cậy. "
    "Dụng cụ chứa món ăn trong ẩm thực Việt Nam có độ biến thiên rất lớn: cùng một món phở nhưng mỗi quán ăn hoặc gia đình lại sử dụng kích cỡ tô khác nhau "
    "(đường kính miệng dao động từ 160 mm đến 220 mm, độ sâu từ 60 mm đến 100 mm; đĩa cơm dao động từ 180 mm đến 240 mm). "
    "Do thể tích tỉ lệ với lũy thừa bậc ba của kích thước hình học, một sai số nhỏ từ giả định đường kính đĩa sẽ bị khuếch đại nghiêm trọng, "
    "dẫn đến kết quả thể tích và khối lượng sai lệch gấp nhiều lần. Vì vậy, hệ thống kiên quyết KHÔNG sử dụng phương pháp phỏng đoán theo đĩa (plate heuristic) "
    "cũng như các phỏng đoán thô từ ảnh để ước lượng thể tích."
))

# 24. P[332] (No math): CORE REWRITE OF REFERENCE SERVING FALLBACK
set_para_text(doc.paragraphs[332], (
    "Cơ chế dự phòng khi không có marker - Tính toán dinh dưỡng theo khẩu phần ăn mẫu chuẩn: "
    "Khi ảnh không phát hiện được marker ArUco (hoặc bảng bàn cờ), hệ thống dừng toàn bộ chuỗi tích phân thể tích 3D "
    "và tự động chuyển sang cơ chế tính toán dựa trên khối lượng khẩu phần ăn mẫu chuẩn (reference serving size). "
    "Cụ thể, hệ thống tra cứu khối lượng khẩu phần chuẩn tương ứng với lớp món ăn từ cơ sở dữ liệu dinh dưỡng chuẩn hóa của Viện Dinh dưỡng Quốc gia (NIN) "
    "và Bảng thành phần thực phẩm Việt Nam (ví dụ: khẩu phần tham chiếu chuẩn của Phở là 450 g, Cơm tấm là 350 g, Bánh mì là 150 g...). "
    "Giá trị dinh dưỡng sau đó được tính toán trực tiếp từ khối lượng chuẩn này nhân với hàm lượng trên 100 g. "
    "Kết quả xuất ra được gắn nhãn nguồn là reference_serving_fallback, xếp vào mức tin cậy dự phòng, đồng thời giao diện hiển thị cảnh báo "
    "cho người dùng biết kết quả đang dựa trên khẩu phần chuẩn trung bình và khuyến nghị nên chụp lại kèm marker ArUco để ước lượng thể tích chính xác cho bữa ăn cụ thể."
))

# 25. P[333] (No math): REWRITE OF FALLBACK PRINCIPLE
set_para_text(doc.paragraphs[333], (
    "Nguyên tắc xử lý dự phòng nhất quán của hệ thống: khi không thu được tham chiếu kích thước thực có kiểm chứng (cấp 1–2), "
    "hệ thống KHÔNG thực hiện ước lượng thể tích mang tính phỏng đoán, mà chuyển hoàn toàn sang tính toán theo khẩu phần chuẩn kèm thông báo rõ ràng cho người dùng. "
    "Ngược lại, khi có sự hiện diện của marker ArUco, hệ thống tiến hành bước tái cân chỉnh độ sâu (re-anchor depth) "
    "theo vector tịnh tiến tvec của marker (mục 3.9.1), bảo đảm độ nổi của bản đồ độ sâu hoàn toàn nhất quán về mặt hình học với tỷ lệ kích thước theo phương ngang."
))

# 26. P[336]: Has 4 math elements
p336 = doc.paragraphs[336]
p336.runs[0].text = "Độ sâu theo đơn vị thực từ mô hình đơn ảnh mang sai số hệ thống nhân: "
p336.runs[4].text = " lệch khỏi 1 (quan sát phát triển hệ thống cho thấy cỡ 10–30%). Tỷ lệ theo phương ngang "
p336.runs[6].text = " (mm/pixel) từ marker - nguồn thông tin độc lập với độ sâu - cho phép khôi phục: tại mặt phẳng thức ăn, "

# 27. P[338]: Has 2 math elements
p338 = doc.paragraphs[338]
p338.runs[2].text = (
    ". Hệ số được chặn trong khoảng [0,5; 2,0] - giá trị ngoài khoảng báo hiệu dữ liệu không nhất quán "
    "và hệ thống ưu tiên giữ độ sâu gốc thay vì áp hiệu chỉnh sai. Sau hiệu chỉnh, độ nổi (độ cao tương đối của bề mặt thức ăn so với mặt phẳng) "
    "nhất quán về đơn vị thực với tỷ lệ theo phương ngang. Mặt phẳng tham chiếu "
)
p338.runs[4].text = (
    " được khớp bình phương tối thiểu trên vành khăn quanh mặt nạ (vùng bàn hoặc đĩa) với hai vòng lược nhiễu bằng phương pháp "
    "cắt lọc xích-ma dựa trên độ lệch tuyệt đối trung vị (MAD-based sigma clipping) - chống nhiễu từ vật thể nền phía sau; "
    "mặt phẳng được dịch theo trung vị phần dư để đại diện đúng bề mặt đỡ dưới thức ăn."
)

# 28. P[344]: Has 1 math element
p344 = doc.paragraphs[344]
p344.runs[0].text = "Khi có ma trận tham số nội tại K (f_x, f_y), công thức chuyển đổi hình học vi phân trực tiếp từ mô hình camera lỗ kim (camera pinhole) được áp dụng: "
p344.runs[2].text = (
    ". Cách tính này đảm bảo các điểm ảnh ở gần camera (Z nhỏ) có diện tích vật lý tương ứng nhỏ hơn các điểm ảnh ở xa (Z lớn), "
    "giải quyết triệt để hiệu ứng phối cảnh mà không giả định diện tích điểm ảnh là hằng số phẳng toàn cục. "
    "Chiều cao khoảng bù thức ăn trên mặt đế (5 mm) được cộng vào phần cột nền. Kết quả tính bằng mm³ và đổi ra cm³."
)

# 29. P[346] (No math):
set_para_text(doc.paragraphs[346], (
    "Tích phân cột thuần chỉ đúng khi toàn bộ khối thức ăn nằm trên mặt phẳng và nhìn thấy được từ góc chụp phía trên. "
    "Với món đựng trong bát hoặc tô lõm (phở, bún bò, cơm trong chén), phần thức ăn nằm dưới đường mép - tích phân cột sẽ bị thiếu. "
    "Do đó khi có marker ArUco, hệ thống phân rã ước lượng hình học theo loại dụng cụ chứa và loại món thành bốn mô hình chuyên biệt:"
))

# 30. P[348]: Has 5 math elements (Drim, H, Drim, Drim, DeltaZ)
p348 = doc.paragraphs[348]
p348.runs[4].text = ", bán kính đáy = 0,52·bán kính miệng, mức làm đầy mặc định 0,78 khi bản đồ độ sâu phẳng. Mô hình này chỉ áp dụng khi có marker ArUco: "
p348.runs[6].text = (
    " suy từ hộp giới hạn × mm/pixel (chặn trong [100; 280] mm, kèm ngưỡng dung sai đường kính 10%); "
    "khi không có marker, hệ thống không áp dụng mô hình bát dựa trên đường kính giả định (tránh lặp lại sai số phỏng đoán theo đĩa) "
    "mà chuyển sang tính dinh dưỡng theo khẩu phần ăn mẫu chuẩn (mục 3.8.2). Khi có marker, mức làm đầy thực được đo bằng khoảng trống mép bát (headroom) "
)
p348.runs[10].text = (
    " từ bản đồ độ sâu, chặn bởi tỷ lệ khoảng trống tối đa 30% độ sâu bát để chống phóng đại thể tích khi độ sâu bị nhiễu; "
    "khi tín hiệu khoảng trống không nhất quán (dấu đảo, vượt ngưỡng), hệ thống rơi về mức làm đầy dự phòng BOWL_NOISE_FILL_RATIO "
    "thay vì giả định đầy đến miệng - nhánh dự phòng này được quét tham số trong thực nghiệm (mục 4.9):"
)

# 31. P[356]: Has 1 math element
p356 = doc.paragraphs[356]
p356.runs[2].text = (
    ". Cơ sở dữ liệu khối lượng riêng được xây dựng theo hai lớp nguồn: "
    "(i) nguồn đo thực nghiệm của đề tài - phương pháp đo lượng giãn nước, mỗi lớp 3 mẫu, ghi nhận trung bình ± độ lệch chuẩn (mục tiêu lõi, sản phẩm tái sử dụng được); "
    "(ii) nguồn tham khảo có ghi nhãn - USDA FoodData Central (usda_fdc) cho nguyên liệu cơ bản, Bảng thành phần thực phẩm Việt Nam (vn_nin_2017), "
    "giá trị từ tài liệu khoa học thực phẩm (literature) theo nhóm món, và ước tính kỹ thuật theo loại đĩa (category_estimate) làm lớp dự phòng cuối - "
    "mọi tra cứu trả về (giá trị, nhãn nguồn), kết quả khối lượng luôn kèm nhãn nguồn để phân tích sai số và phát hiện phương án dự phòng ẩn."
)

# 32. P[363]: Has 3 math elements
p363 = doc.paragraphs[363]
p363.runs[0].text = (
    "Lớp dữ liệu dinh dưỡng được tổ chức theo kiến trúc nguồn sự thật duy nhất (single source of truth): "
    "một tệp chuẩn hóa NIN canonical chứa 2.103 bản ghi (1.250 món và 853 nguyên liệu) với giản đồ 12 nhóm chất trên 100 g, "
    "được chuẩn hóa từ Bảng thành phần thực phẩm Việt Nam [20] (nguồn Viện Dinh dưỡng Quốc gia) bổ sung USDA FoodData Central; "
    "mỗi bản ghi gắn nhãn nguồn và trạng thái kiểm chứng. Bản đồ ánh xạ từ 68 lớp VietFood68 sang các bản ghi NIN phủ 68/68 lớp "
    "với ba loại: trực tiếp (direct: 30), tương đương (approximate: 19), phân rã thành phần (component_based: 16), chỉ 3 lớp không ánh xạ được; "
    "khẩu phần tham chiếu của từng lớp cũng ghi rõ nguồn. Bộ dữ liệu khi chạy (class_names) là hình chiếu trong bộ nhớ của tệp chuẩn hóa này - "
    "không phải nguồn độc lập. Tính nhất quán nội tại của dữ liệu được kiểm tra bằng hệ số Atwater [31]: 97,1% bản ghi nhất quán trong ngưỡng ≤15% "
    "(chi tiết kiểm toán tại mục 4.13). Bảy nhóm chất chính được theo dõi: calories (kcal), protein, fat, carbohydrate, saturates, sugar, salt (g). "
    "Tổng dinh dưỡng bữa ăn "
)
p363.runs[4].text = ". Nguyên tắc của mô-đun: không tự tạo giá trị dinh dưỡng; mọi con số truy về "
p363.runs[6].text = (
    " của nguồn đã ghi, và sai số khối lượng truyền nguyên dạng vào sai số dinh dưỡng. "
    "Khi ảnh không có marker ArUco, khối lượng khẩu phần ăn mẫu chuẩn được nạp trực tiếp làm m_i trong công thức (3.6) để suy ra dinh dưỡng bữa ăn."
)

# 33. P[365] (No math):
set_para_text(doc.paragraphs[365], (
    "Bộ máy được cài đặt thành một mô-đun quy tắc tất định (deterministic rule engine) với kiến trúc bốn tầng tách bạch: "
    "(1) tầng dữ liệu - 12 nhóm chất chuẩn hóa từ nguồn Viện Dinh dưỡng Quốc gia (mục 3.11); "
    "(2) tầng tham chiếu - bảng Nhu cầu Dinh dưỡng Khuyến nghị cho người Việt Nam (RNI 2016) [22]: "
    "protein 60/50 g, canxi 800 mg, sắt 11/18 mg, kẽm 10/8 mg (nam/nữ trưởng thành), giới hạn natri ≤ 2.000 mg theo WHO, "
    "cholesterol ≤ 300 mg, đường tự do ≤ 50 g, muối ≤ 5 g; cùng phương trình tiêu hao năng lượng khi nghỉ BMR Mifflin–St Jeor [21]; "
    "(3) tầng quy tắc tất định - mọi phép tính (tổng năng lượng tiêu hao hằng ngày TDEE = BMR × hệ số hoạt động theo bậc 1,2–1,9 [33]; "
    "điều chỉnh mục tiêu ±500 kcal/ngày theo quy tắc cân bằng năng lượng kinh điển [34], sàn 1.200 kcal; cân bằng năng lượng Atwater 4/4/9 [31]; "
    "độ phủ dinh dưỡng so với RNI; cảnh báo hệ thống theo ngưỡng) đều chạy bằng mã nguồn xác định, không qua mô hình sinh; "
    "(4) tầng giao diện mô hình ngôn ngữ lớn - xuất tệp dữ liệu có cấu trúc JSON (structured facts JSON). "
    "Tỷ lệ phân bổ các chất đa lượng theo khuyến nghị quốc gia (protein 13–20%, chất béo 20–25%, carbohydrate 55–65%) "
    "cũng được nạp từ tầng tham chiếu."
))

# 34. P[368] (No math):
set_para_text(doc.paragraphs[368], (
    "Tệp facts JSON gồm: mục tiêu người dùng (user_goal), lượng thiếu hụt deficits (kèm lượng nạp, mục tiêu và tỷ lệ phần trăm cho từng chất), "
    "lượng vượt mức excesses, đạt mục tiêu on_target, danh sách món ăn phát hiện (kèm khối lượng ước lượng hoặc khối lượng khẩu phần chuẩn "
    "và nhánh truy nguồn từng phần: khối lượng → giá trị trên 100 g → định danh bản ghi → nguồn NIN/USDA). "
    "Thiết kế cốt lõi: facts JSON là TOÀN BỘ dữ liệu mà lớp mô hình ngôn ngữ lớn được phép nhìn thấy cho bữa ăn này "
    "(nguyên tắc \"nếu mô hình ngôn ngữ lớn cần một thông tin, đưa nó tường minh vào facts; không để mô hình tự suy từ dữ liệu một phần\"). "
    "Bộ máy chạy ngoại tuyến (offline), không phụ thuộc dịch vụ ngoài, luôn tạo được lời khuyên ở mức bản mẫu template (mục 3.13, phương án dự phòng), "
    "và hành vi của nó được khóa bằng bộ kiểm thử chấp nhận riêng (mục 4.13)."
))

# 35. P[370] (No math):
set_para_text(doc.paragraphs[370], (
    "Lớp tư vấn được thiết kế theo kiến trúc kết hợp phân lập trách nhiệm: bộ máy quy tắc (3.12) tính toán toàn bộ giá trị; "
    "mô hình ngôn ngữ lớn chỉ đảm nhận diễn đạt. Lời nhắc hệ thống (system prompt) ràng buộc mô hình: nhận facts JSON; "
    "nhiệm vụ giới hạn ở nhận xét món ăn, đánh giá bữa ăn theo các trường thiếu/thừa/đạt chuẩn, và khuyến nghị hành động theo mục tiêu người dùng; "
    "cấm tự tạo giá trị số mới; trả lời tiếng Việt, giọng khoa học, ngắn gọn (3–4 đoạn)."
))

# 36. P[371] (No math):
set_para_text(doc.paragraphs[371], (
    "Giải pháp giảm thiểu hiện tượng tạo thông tin không có căn cứ (hallucination) được triển khai theo ba lớp: "
    "(i) cấu trúc dữ liệu đầu vào (facts JSON tường minh - 3.12); (ii) ràng buộc lời nhắc hệ thống nêu trên; "
    "(iii) truy vấn tăng cường qua cơ chế sinh tăng cường truy xuất (Retrieval-Augmented Generation - RAG) [37]: "
    "câu hỏi người dùng (làm giàu bằng tên món phát hiện) được nhúng bằng mô hình text-embedding-004 (768 chiều), "
    "truy vấn 3 đoạn kiến thức có độ tương đồng cao nhất trong cơ sở dữ liệu vector Qdrant được xây từ 7 nguồn tài liệu chuẩn hóa "
    "(khuyến nghị WHO/FAO, hướng dẫn Bộ Y tế, RDA, nguyên tắc quản lý cân nặng…), lọc theo ngưỡng tương đồng 0,45, "
    "và các đoạn được chèn vào ngữ cảnh với chỉ dẫn ưu tiên sử dụng tài liệu trước kiến thức nội tại."
))

# 37. P[372] (No math):
set_para_text(doc.paragraphs[372], (
    "Về nhà cung cấp mô hình: thiết kế của đề cương ưu tiên Qwen2.5-7B-Instruct [16] suy luận cục bộ qua Ollama "
    "(lượng tử hóa GGUF, phù hợp bộ nhớ GPU 6 GB) - phương án không phụ thuộc dịch vụ ngoài. "
    "Trong cài đặt phát triển hiện tại, lớp tư vấn được triển khai đa nhà cung cấp qua giao diện lập trình ứng dụng (API) "
    "để tiện thử nghiệm và so sánh: Gemini [36] (mặc định), Cerebras, OpenRouter và Cloudflare Workers AI; "
    "người dùng cấu hình nhà cung cấp và khóa xác thực ở thanh bên, facts JSON và ngữ cảnh RAG giữ nguyên giữa các nhà cung cấp. "
    "Việc so sánh định lượng hiện tượng tạo thông tin không có căn cứ giữa kiến trúc kết hợp và mô hình ngôn ngữ lớn thuần túy (RQ5) "
    "thuộc hạng mục mở rộng tại Chương 4; hệ thống không tuyên bố lời khuyên luôn chính xác tuyệt đối - "
    "phản hồi của mô hình luôn được hiển thị kèm ngữ cảnh là \"diễn giải từ số liệu đã tính toán\"."
))

# 38. P[374] (No math):
set_para_text(doc.paragraphs[374], (
    "Lược đồ dữ liệu gồm ba nhóm: hồ sơ người dùng (thể trạng, mục tiêu - lưu trong phiên làm việc); "
    "dữ liệu bữa ăn (kết quả quy trình xử lý dạng JSON: phát hiện, mặt nạ, trực quan hóa độ sâu, thể tích/khối lượng/dinh dưỡng từng món "
    "kèm nhãn nguồn và độ lệch chuẩn, khả năng xuất tệp CSV/JSON cho người dùng); "
    "tài nguyên kiến thức (cơ sở dữ liệu khối lượng riêng dạng bảng có nhãn nguồn; lớp dữ liệu NIN chuẩn hóa; "
    "tài liệu kiến thức cho RAG dạng văn bản được cắt đoạn và nhúng vector). "
    "Một lớp máy chủ giao diện lập trình ứng dụng (FastAPI) và lưu trữ bền vững (PostgreSQL) được xác định là mục tiêu kiến trúc mở rộng "
    "của đề cương - chưa thuộc phạm vi triển khai hiện tại (trình bày tại mục 5.8); "
    "trong cài đặt hiện hành, toàn bộ trạng thái phiên nằm trong phiên làm việc của ứng dụng và kết quả trao đổi qua tệp JSON/CSV tường minh."
))

# 39. P[376] (No math):
set_para_text(doc.paragraphs[376], (
    "Kiến trúc triển khai hiện hành gồm: giao diện Streamlit đa thẻ (ảnh tải lên/ảnh mẫu, video/YouTube, webcam thời gian thực, "
    "camera IP RTSP, hồ sơ sức khỏe, hội thoại tư vấn) và một tiến trình xử lý độc lập chạy quy trình tính toán thể tích trên GPU. "
    "Điểm cốt lõi của kiến trúc là sự cô lập hai tiến trình: onnxruntime (phát hiện) và PyTorch CUDA (phân đoạn và độ sâu) "
    "chạy chung một tiến trình gây lỗi hệ thống cục bộ trên một số cấu hình Windows (xung đột trạng thái thư viện liên kết động DLL và CUDA), "
    "nên quy trình xử lý thể tích được đặt trong tiến trình xử lý riêng biệt. Tiến trình này vận hành theo hai chế độ: "
    "tiến trình nền bền vững (daemon, khởi động một lần, nạp sẵn SAM 2.1 và Depth Anything V2 vào bộ nhớ GPU - dung lượng đỉnh ~623 MiB - "
    "giúp suy luận ấm chỉ mất ~1,8–2,0 giây/yêu cầu thay vì phải nạp lại mô hình mỗi lần) và chế độ chạy một lần (one-shot) dự phòng "
    "(ghi tệp công việc job.json, chạy tiến trình con với thời hạn chờ, đọc output.json). "
    "Giao tiếp giữa hai tiến trình là dòng dữ liệu JSON phân tách theo dòng trên luồng xuất chuẩn stdout "
    "(mọi nhật ký ghi lỗi ép về luồng stderr), kèm cơ chế bắt tay sẵn sàng và tín hiệu duy trì kết nối (heartbeat); "
    "khi tiến trình lỗi hoặc vượt quá thời hạn chờ (300 giây), hệ thống tự động chuyển sang chế độ chạy một lần "
    "rồi kích hoạt phương án dự phòng theo khẩu phần ăn mẫu chuẩn với cảnh báo hiển thị trên giao diện - "
    "đối với kết quả nghiên cứu theo lô thì ngược lại coi phương án dự phòng là lỗi nghiêm trọng (mục 4.3). "
    "Cùng cơ chế cô lập tiến trình, bộ FoodSAM chạy trong môi trường Python hoàn toàn riêng biệt và giao tiếp qua tiến trình con subprocess. "
    "Đồ thị luồng dữ liệu tổng thể: ảnh → (phát hiện - CPU) → JSON phát hiện → "
    "(tiến trình tính toán thể tích: phân đoạn → độ sâu → khôi phục tỷ lệ → thể tích → khối lượng → dinh dưỡng) → JSON kết quả → "
    "(tổng hợp bữa ăn) → bộ máy quy tắc → facts JSON → [mô hình ngôn ngữ lớn] → giao diện người dùng. "
    "Chuỗi này minh họa tính tách rời của các mô-đun: mọi giao tiếp qua JSON tường minh, cho phép chạy riêng từng mô-đun "
    "phục vụ thí nghiệm và tái lập kết quả."
))

# 40. P[378] (No math):
set_para_text(doc.paragraphs[378], (
    "Chương 4 đánh giá hệ thống theo bảng ánh xạ yêu cầu chức năng sang mô-đun đảm nhận (Bảng 3.2) và các chỉ số tại mục 2.8. "
    "Kỳ vọng sai số được trình bày dạng khoảng có căn cứ tài liệu (Bảng 2.2), không phải ngưỡng cam kết cứng: "
    "MAPE khối lượng kỳ vọng nằm trong dải 10–68% quan sát được ở các nghiên cứu tương tự, mục tiêu của đề tài là mức nửa dưới của dải, "
    "đồng thời ưu tiên phân tích rõ nguồn gốc sai số (RQ2) hơn là theo đuổi một con số tuyệt đối. "
    "Chất lượng phát hiện kỳ vọng tiệm cận mốc cơ sở công bố của VietFood67 (mAP50 ≈ 0,92 trên tập kiểm thử). "
    "Về hiệu năng: đo độ trễ từng bước và toàn bộ quy trình trên GPU RTX 3060 6 GB; "
    "chiến lược giảm tải đã thiết kế gồm sử dụng biến thể mô hình nhẹ (SAM 2.1 hiera-small hoặc MobileSAM, độ sâu vits), "
    "suy luận tuần tự và chuyển đổi sang định dạng ONNX - mục tiêu tham chiếu khoảng 15–20 giây/ảnh cho luồng ước lượng khối lượng đầy đủ "
    "(khoảng kỳ vọng, không phải ngưỡng cứng, theo góp ý của giảng viên hướng dẫn và kết quả kiểm thử nhanh giai đoạn 1). "
    "Các tiêu chí cụ thể từng thí nghiệm (mức độ đo được, điều kiện dữ liệu) được trình bày tại bảng ánh xạ thực nghiệm ở Chương 4."
))

# 41. P[381] (No math):
set_para_text(doc.paragraphs[381], (
    "Kết thúc chương: chương này đã thiết kế đầy đủ phương pháp của đề tài từ phát biểu toán học đến kiến trúc triển khai, "
    "với các điểm neo phương pháp luận then chốt: giao diện mô-đun trung lập cho phép so sánh khách quan các phương pháp phân đoạn (3.5–3.6); "
    "nguyên tắc khôi phục tỷ lệ chặt chẽ - kiên quyết loại bỏ phương pháp phỏng đoán theo đĩa (plate heuristic) "
    "để chuyển sang tính toán theo khẩu phần ăn mẫu chuẩn khi thiếu marker ArUco nhằm triệt tiêu sai số suy đoán (3.8); "
    "hiệu chuẩn độ sâu bằng tham chiếu theo phương ngang và các mô hình hình học theo loại món khi có marker (3.9); "
    "cùng kiến trúc phân lập rạch ròi giữa tính toán số liệu và diễn đạt ngôn ngữ tự nhiên trong lớp tư vấn dinh dưỡng (3.12–3.13). "
    "Toàn bộ thiết kế chưa gắn số liệu kết quả; chương tiếp theo trình bày khung thực nghiệm và kết quả đo đạc theo đúng các tiêu chí kỳ vọng đã đặt ra."
))

# Save to target
output_path = r'd:\Document\Year4_Semester1\KhoaLuanTotNghiep\FoodDetector-2 - V3\KLTN_Chuong1-3_bien_tap_ngon_ngu.docx'
doc.save(output_path)
print("Successfully updated Chapter 3 in KLTN_Chuong1-3_bien_tap_ngon_ngu.docx!")
