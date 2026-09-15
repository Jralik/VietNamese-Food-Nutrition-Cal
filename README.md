# 🇻🇳 Vietnamese Food Nutrition & Calorie Detector (VietNamese-Food-Nutrition-Cal)

[![Live Demo](https://img.shields.io/badge/Live_App-Streamlit_Cloud-FF4B4B?style=for-the-badge&logo=streamlit)](https://vietnamese-food-nutrition-cal.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![YOLOv26](https://img.shields.io/badge/Model-YOLOv26-green.svg)](https://github.com/Jralik/VietNamese-Food-Nutrition-Cal)
[![mAP50](https://img.shields.io/badge/mAP50-0.95-brightgreen.svg)](https://github.com/Jralik/VietNamese-Food-Nutrition-Cal)

> 🌐 **Live Web Application**: [https://vietnamese-food-nutrition-cal.streamlit.app/](https://vietnamese-food-nutrition-cal.streamlit.app/)

**VietNamese-Food-Nutrition-Cal** là hệ thống phát hiện món ăn Việt Nam thời gian thực và tự động tính toán giá trị dinh dưỡng (Calo, Protein, Chất béo, Đường, Muối), tích hợp trợ lý AI đa mô hình (Gemini, Cerebras, OpenRouter) tư vấn chế độ ăn uống khoa học dựa trên chỉ số BMI & TDEE của người dùng.

Hệ thống được huấn luyện trên mô hình **YOLOv26** tiên tiến với bộ dữ liệu **VietFood67** gồm 67-68 món ăn đặc trưng của ẩm thực Việt Nam, đạt độ chính xác **mAP50 = 0.95**.

---

## 🌐 Trải Nghiệm Trực Tuyến (Live Demo)

Bạn có thể truy cập và trải nghiệm trực tiếp hệ thống tại địa chỉ:  
👉 **[https://vietnamese-food-nutrition-cal.streamlit.app/](https://vietnamese-food-nutrition-cal.streamlit.app/)**
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

## 🛠️ Hướng Dẫn Cài Đặt & Chạy Cục Bộ (Local Setup)

### 1. Yêu cầu môi trường
- Python >= 3.10
- Git

### 2. Cài đặt thư viện
```bash
# Clone repository
git clone https://github.com/Jralik/VietNamese-Food-Nutrition-Cal.git
cd VietNamese-Food-Nutrition-Cal

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
```## Installation

We have tested the code with `python==3.7` and `pytorch=1.8`, other late versions may also work well. 
<br>
Welcome to provide feedback or suggestion for the version list!
<!-- Please follow the instructions [here](https://pytorch.org/get-started/locally/) to install both PyTorch dependencies. 
Installing PyTorch and TorchVision with CUDA support is strongly recommended. -->

Install FoodSAM with the following steps:

a. Clone the repository locally:

```
git clone https://github.com/jamesjg/FoodSAM.git
```
b. Create a conda virtual environment and activate it
```
conda create -n FoodSAM python=3.7 -y
conda activate FoodSAM
```
c. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/). Here we use PyTorch 1.8.1 and CUDA 11.1. You may also switch to another version by specifying the version number.
```
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
```
d. Install MMCV following the [official instructions](https://mmcv.readthedocs.io/en/latest/#installation). 
```
pip install mmcv-full==1.3.0 -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.8.0/index.html
```
e. Install SAM following official [SAM installation](https://github.com/facebookresearch/segment-anything).
```
pip install git+https://github.com/facebookresearch/segment-anything.git@6fdee8f
```
f. other requirements
```
pip install -r requirement.txt
```

e. Finally download three checkpoints, and move them to "ckpts/" folder as described.

[SAM-vit-h](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)

[FoodSeg103-SETR-MLA](https://smu-my.sharepoint.com/personal/xwwu_smu_edu_sg/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fxwwu%5Fsmu%5Fedu%5Fsg%2FDocuments%2Fcheckpoints%2Ezip&parent=%2Fpersonal%2Fxwwu%5Fsmu%5Fedu%5Fsg%2FDocuments&ga=1)

[UNIDET-Unified_learned_OCIM_RS200_6x+2x](https://drive.google.com/file/d/1HvUv399Vie69dIOQX0gnjkCM0JUI9dqI/edit)

If the above links are not working, you also can download them in [Baidu Disk](https://pan.baidu.com/s/1o1w_Vejrtd7rvWVorSQZfg?pwd=pyyk) (code:`pyyk`).

## Dataset and configs
For UNIDET and FoodSeg103, the configs are already put into the [configs](configs/) folder. 
You can also download other ckpt and configs from their official links.

The default dataset we use is [FoodSeg103](https://github.com/LARC-CMU-SMU/FoodSeg103-Benchmark-v1), other semantic segmentation food datasets like [UECFOODPIXCOMPLETE](https://mm.cs.uec.ac.jp/uecfoodpix/) can also be used. But you should change the  `args.category_txt and args.num_class`. The dataset should be put in the "dataset/"folder.

Your data, configs, and ckpt path should look like this:
````
FoodSAM
-- ckpts
   |-- SETR_MLA
   |   |-- iter_80000.pth
   |-- sam_vit_h_4b8939.pth
   |-- Unified_learned_OCIM_RS200_6x+2x.pth
-- configs
   |-- Base-CRCNN-COCO.yaml
   |-- Unified_learned_OCIM_RS200_6x+2x.yaml
   |-- SETR_MLA_768x768_80k_base.py
-- dataset
   |-- FoodSeg103
   |   |-- Images
   |   |   |-- ann_dir
   |   |   |-- img_dir  
-- FoodSAM
-- mmseg
-- UNIDET
   ...

````


