# 🇻🇳 Vietnamese Food Nutrition & Calorie Detector (VietNamese-Food-Nutrition-Cal)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![YOLOv26](https://img.shields.io/badge/Model-YOLOv26-green.svg)](https://github.com/Jralik/VietNamese-Food-Nutrition-Cal)
[![mAP50](https://img.shields.io/badge/mAP50-0.95-brightgreen.svg)](https://github.com/Jralik/VietNamese-Food-Nutrition-Cal)
[![Streamlit](https://img.shields.io/badge/Framework-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**VietNamese-Food-Nutrition-Cal** là hệ thống phát hiện món ăn Việt Nam thời gian thực và tự động tính toán giá trị dinh dưỡng (Calo, Protein, Chất béo, Đường, Muối), tích hợp trợ lý AI đa mô hình (Gemini, Cerebras, OpenRouter) tư vấn chế độ ăn uống khoa học dựa trên chỉ số BMI & TDEE của người dùng.

Hệ thống được huấn luyện trên mô hình **YOLOv26** tiên tiến với bộ dữ liệu **VietFood67** gồm 67-68 món ăn đặc trưng của ẩm thực Việt Nam, đạt độ chính xác **mAP50 = 0.95**.

---

## ✨ Tính Năng Nổi Bật (Key Features)

1. **Nhận Diện Món Ăn Đa Nguồn (Multi-Source Detection with YOLOv26)**:
   - Nhận diện trực tiếp từ **Hình ảnh (Image)**.
   - Nhận diện qua **Video clip / Youtube Video**.
   - Nhận diện trực tiếp qua **Webcam / IP Camera (RTSP)**.

2. **Tính Toán Dinh Dưỡng Chi Tiết & Tự Động**:
   - Tự động tính toán tổng lượng **Calories, Protein, Fat, Saturates, Sugar, Salt**.
   - Đánh giá chỉ số theo hệ thống đèn giao thông dinh dưỡng (Đỏ - Vàng - Xanh).
   - Cho phép xuất kết quả ra file **CSV** hoặc tải ảnh đã vẽ Bounding Box.

3. **Công Cụ Tính Toán Sinh Học (Rule-Based Health Engine)**:
   - Tự động tính toán **BMI** và **TDEE** (Mifflin-St Jeor) theo thông số sức khỏe (Tuổi, Giới tính, Cân nặng, Chiều cao, Mức độ vận động, Mục tiêu giảm/tăng/giữ cân).
   - Xác định lượng **RDA** mục tiêu cho người dùng.

4. **Trợ Lý Tư Vấn AI Đa Mô Hình (Multi-LLM Advisory System)**:
   - Tích hợp **Google Gemini**, **Cerebras** (Llama/Gemma siêu tốc), và **OpenRouter** (hỗ trợ Reasoning / Suy nghĩ chi tiết).
   - Truyền dữ liệu dinh dưỡng thực tế dưới dạng **Structured Facts JSON** để AI tư vấn chính xác, không ảo tưởng số liệu.
   - Giao diện khung Chatbot riêng biệt với thanh cuộn độc lập và các câu hỏi mẫu tiện lợi.

---

## 🛠️ Hướng Dẫn Cài Đặt & Chạy Ứng Dụng (Getting Started)

### 1. Yêu cầu môi trường
- Python >= 3.10
- Git

### 2. Cài đặt thư viện
```bash
# Tạo môi trường ảo
python -m venv .venv

# Kích hoạt môi trường ảo (Windows)
.\.venv\Scripts\activate

# Cài đặt thư viện phụ thuộc
pip install -r requirements.txt
```

### 3. Chạy ứng dụng Streamlit
```bash
streamlit run main.py
```

Truy cập giao diện tại: `http://localhost:8501`

---

## 📁 Cấu Trúc Thư Mục (Directory Structure)

```
VietNamese-Food-Nutrition-Cal/
├── assets/                  # CSS styling, hình ảnh minh họa giao diện
│   └── css/general-style.css
├── model/                   # Trọng số mô hình nhận diện
│   └── yolov26/
│       └── best.onnx        # Mô hình YOLOv26 ONNX
├── pages/                   # Các trang phụ (About, Dataset)
│   ├── about.py
│   └── dataset.py
├── class_names.py           # Cơ sở dữ liệu 68 món ăn Việt Nam & dinh dưỡng
├── main.py                  # Giao diện chính Streamlit & Chatbot AI
├── utils.py                 # Xử lý YOLOv26, tính toán dinh dưỡng & Rule Engine
├── requirements.txt         # Thư viện phụ thuộc
└── README.md
```

---

## 📜 Giấy Phép (License)
Dự án được phát hành theo giấy phép **MIT License**.