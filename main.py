import streamlit as st
import warnings
import logging
import os
# google.generativeai is deprecated in favor of google.genai; the advisory flow
# still uses the old SDK, so silence its startup FutureWarning until migrated.
warnings.filterwarnings(
    "ignore",
    message=r"\s*All support for the `google\.generativeai` package has ended",
    category=FutureWarning,
)
# Make the volume/segmentation pipeline observable in the Streamlit console
# (backend actually used, FoodSAM fallbacks, worker failures).
for _name in ("volume_integration", "food_volume_pipeline", "foodsam_segmenter",
              "food_segmentation", "scale_recovery", "depth_estimation"):
    _lg = logging.getLogger(_name)
    _lg.setLevel(logging.INFO)
    if not _lg.handlers:
        _h = logging.StreamHandler()
        _h.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
        _lg.addHandler(_h)
import google.generativeai as genai
import time
import json
from pathlib import Path

from utils import (
    _display_detected_frame, detect_camera, detect_image, detect_video, detect_webcam,
    load_model, calculate_bmi, calculate_tdee_mifflin_st_jeor,
    build_structured_facts, retrieve_context
)
import ui_components as ui

st.set_page_config(
    page_title="FoodDetector AI — Nhận diện & Tư vấn Dinh dưỡng",
    page_icon="🍜",
    layout="wide",
)

GOAL_VI = {"lose": "Giảm cân", "maintain": "Giữ cân", "gain": "Tăng cân"}
SEX_VI = {"male": "Nam", "female": "Nữ"}
ACTIVITY_VI = {
    1.2: "Ít vận động (sedentary)",
    1.375: "Vận động nhẹ (lightly active)",
    1.55: "Vận động vừa (moderately active)",
    1.725: "Vận động nhiều (very active)",
    1.9: "Vận động nặng (extra active)",
}

DEFAULT_PROFILE = {
    "age": 25,
    "sex": "female",
    "weight": 65.0,
    "height": 170.0,
    "activity_factor": 1.2,
    "goal": "maintain",
}


def compute_rda(profile: dict, tdee: float) -> dict:
    """RDA mục tiêu hằng ngày từ TDEE và mục tiêu sức khỏe."""
    goal = profile["goal"]
    if goal == "lose":
        calories_target = tdee - 500
    elif goal == "gain":
        calories_target = tdee + 500
    else:
        calories_target = tdee
    return {
        "Calories": max(1200.0, calories_target),
        "Protein": profile["weight"] * 1.6,
        "Fat": (calories_target * 0.25) / 9.0,
        "Saturates": (calories_target * 0.08) / 9.0,
        "Sugar": 50.0,
        "Salt": 6.0,
    }


# Initialize default user profile and RDA
if "user_profile" not in st.session_state:
    st.session_state.user_profile = dict(DEFAULT_PROFILE)

if "user_rda" not in st.session_state:
    w = st.session_state.user_profile["weight"]
    h = st.session_state.user_profile["height"]
    tdee = calculate_tdee_mifflin_st_jeor(
        w, h, st.session_state.user_profile["age"],
        st.session_state.user_profile["sex"], st.session_state.user_profile["activity_factor"])
    st.session_state.user_rda = compute_rda(st.session_state.user_profile, tdee)


# ═════════════════════════════════════════════════════════════════════
# SIDEBAR — thương hiệu + tóm tắt thể trạng + cấu hình AI
# ═════════════════════════════════════════════════════════════════════
def render_sidebar():
    with st.sidebar:
        st.markdown(f"""
        <div class="sidebar-brand">
            <span class="brand-badge">🍜</span>
            <div class="sidebar-brand-name">FoodDetector AI
                <small>Nhận diện món Việt · Dinh dưỡng cá nhân hóa</small>
            </div>
        </div>
        """, unsafe_allow_html=True)

        profile = st.session_state.user_profile
        bmi = calculate_bmi(profile["weight"], profile["height"])
        tdee = calculate_tdee_mifflin_st_jeor(
            profile["weight"], profile["height"], profile["age"],
            profile["sex"], profile["activity_factor"])
        rda = st.session_state.user_rda

        st.markdown(f"""
        <div class="sidebar-stat-row">
            <div class="sidebar-stat"><span class="v">{bmi:.1f}</span><span class="l">BMI</span></div>
            <div class="sidebar-stat"><span class="v">{tdee:.0f}</span><span class="l">TDEE (kcal)</span></div>
            <div class="sidebar-stat"><span class="v">{rda['Calories']:.0f}</span><span class="l">Mục tiêu</span></div>
        </div>
        <div style="font-size:11px; color: var(--text-faint); margin-bottom: 0.4rem;">
            {SEX_VI.get(profile['sex'], profile['sex'])} · {profile['age']} tuổi ·
            {profile['weight']:.0f} kg · {profile['height']:.0f} cm · {GOAL_VI.get(profile['goal'], '')}
        </div>
        """, unsafe_allow_html=True)

        st.divider()
        render_sidebar_ai_config()

        st.divider()
        st.markdown(
            f'<div style="font-size:11px; color: var(--text-faint); line-height:1.7;">'
            f'🍜 FoodDetector AI — đồ án tốt nghiệp<br>'
            f'<a href="{ui.GITHUB_URL}" target="_blank">Mã nguồn trên GitHub</a></div>',
            unsafe_allow_html=True)


def render_sidebar_ai_config():
    st.markdown(ui.section_header("🔑 Cấu hình Trợ lý AI",
                                  "API key chỉ lưu trong phiên làm việc của bạn."), unsafe_allow_html=True)

    llm_provider = st.selectbox(
        "Chọn nhà cung cấp AI",
        ["Gemini", "Cerebras", "OpenRouter", "Cloudflare"],
        index=0,
        help="Đánh giá bữa ăn và chat tư vấn sẽ dùng nhà cung cấp này.",
    )
    st.session_state.llm_provider = llm_provider

    if llm_provider == "Gemini":
        api_key_input = st.text_input("Gemini API Key", type="password",
                                      help="Nhận key miễn phí từ Google AI Studio")
        if api_key_input:
            st.session_state.gemini_api_key = api_key_input
        if st.session_state.get("gemini_api_key"):
            st.success("✅ Đã sẵn sàng", icon=None)
    elif llm_provider == "Cerebras":
        cerebras_api_key_input = st.text_input("Cerebras API Key", type="password",
                                               help="Key miễn phí từ Cerebras Cloud Console — tốc độ siêu nhanh!")
        if cerebras_api_key_input:
            st.session_state.cerebras_api_key = cerebras_api_key_input
        cerebras_api_key = st.session_state.get("cerebras_api_key", "")
        if cerebras_api_key:
            available_models = get_cerebras_models(cerebras_api_key)
            st.session_state.cerebras_model = st.selectbox(
                "Mô hình Cerebras", available_models, index=0)
            st.success("✅ Đã sẵn sàng", icon=None)
    elif llm_provider == "OpenRouter":
        openrouter_api_key_input = st.text_input("OpenRouter API Key", type="password",
                                                 help="Truy cập hàng trăm mô hình AI (có bản miễn phí)!")
        if openrouter_api_key_input:
            st.session_state.openrouter_api_key = openrouter_api_key_input
        openrouter_api_key = st.session_state.get("openrouter_api_key", "")
        if openrouter_api_key:
            available_models = get_openrouter_models(openrouter_api_key)
            st.session_state.openrouter_model = st.selectbox(
                "Mô hình OpenRouter", available_models, index=0)
            st.success("✅ Đã sẵn sàng", icon=None)
    else:  # Cloudflare
        cloudflare_account_id_input = st.text_input(
            "Cloudflare Account ID", type="password",
            help="Lấy từ Cloudflare Dashboard → Workers & Pages")
        cloudflare_api_token_input = st.text_input(
            "Cloudflare API Token", type="password",
            help="Tạo Token có quyền Workers AI từ My Profile → API Tokens")
        if cloudflare_account_id_input:
            st.session_state.cloudflare_account_id = cloudflare_account_id_input
        if cloudflare_api_token_input:
            st.session_state.cloudflare_api_token = cloudflare_api_token_input
        if st.session_state.get("cloudflare_account_id") and st.session_state.get("cloudflare_api_token"):
            available_models = get_cloudflare_models(
                st.session_state.cloudflare_account_id, st.session_state.cloudflare_api_token)
            st.session_state.cloudflare_model = st.selectbox(
                "Mô hình Cloudflare Workers AI", available_models, index=0)
            st.success("✅ Đã sẵn sàng", icon=None)


def get_cerebras_models(api_key):
    try:
        import requests
        headers = {"Authorization": f"Bearer {api_key}"}
        response = requests.get("https://api.cerebras.ai/v1/models", headers=headers, timeout=5)

        if response.status_code == 200:
            data = response.json()
            models = [model["id"] for model in data.get("data", [])]
            if models:
                return models

    except Exception:
        pass

    # Trả về danh sách mặc định nếu API lỗi hoặc không có quyền truy cập
    return ["gpt-oss-120b", "gemma-4-31b", "zai-glm-4.7"]


def get_openrouter_models(api_key):
    try:
        import requests
        headers = {"Authorization": f"Bearer {api_key}"}
        response = requests.get("https://openrouter.ai/api/v1/models", headers=headers, timeout=5)
        if response.status_code == 200:
            data = response.json()
            models = [model["id"] for model in data.get("data", [])]
            if models:
                # Put some common models first
                favorites = ["google/gemini-2.5-flash", "google/gemini-2.5-flash:free", "openrouter/auto"]
                for fav in reversed(favorites):
                    if fav in models:
                        models.remove(fav)
                        models.insert(0, fav)
                return models
    except Exception:
        pass
    return ["google/gemini-2.5-flash:free", "meta-llama/llama-3-8b-instruct:free", "openrouter/auto"]


def get_cloudflare_models(account_id="", api_token=""):
    free_defaults = [
        "@cf/meta/llama-3.1-8b-instruct",
        "@cf/deepseek-ai/deepseek-r1-distill-qwen-32b",
        "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
        "@cf/meta/llama-3-8b-instruct",
        "@cf/qwen/qwen1.5-7b-chat-awq",
        "@cf/mistral/mistral-7b-instruct-v0.2"
    ]
    if account_id and api_token:
        try:
            import requests
            headers = {"Authorization": f"Bearer {api_token}"}
            response = requests.get(
                f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/models/search?task=Text%20Generation",
                headers=headers, timeout=5)
            if response.status_code == 200:
                data = response.json()
                fetched = [model["name"] for model in data.get("result", []) if "name" in model]
                # Filter out paid-only models
                filtered = [m for m in fetched if not any(p in m.lower() for p in ["glm-5", "claude", "gpt-4"])]
                if filtered:
                    if "@cf/meta/llama-3.1-8b-instruct" in filtered:
                        filtered.remove("@cf/meta/llama-3.1-8b-instruct")
                        filtered.insert(0, "@cf/meta/llama-3.1-8b-instruct")
                    return filtered
        except Exception:
            pass
    return free_defaults


# ═════════════════════════════════════════════════════════════════════
# TAB 1 — THỂ TRẠNG & MỤC TIÊU
# ═════════════════════════════════════════════════════════════════════
def render_health_tab():
    st.markdown(ui.section_header(
        "🧬 Thể trạng & Mục tiêu dinh dưỡng",
        "Nhập thông tin của bạn — app tính BMI, TDEE và nhu cầu dinh dưỡng hằng ngày để "
        "so sánh với mọi bữa ăn bạn quét."), unsafe_allow_html=True)

    profile = st.session_state.user_profile

    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.number_input("Tuổi", min_value=1, max_value=120, value=int(profile["age"]), key="profile_age")
        height = st.number_input("Chiều cao (cm)", min_value=50.0, max_value=250.0,
                                 value=float(profile["height"]), key="profile_height")
    with c2:
        sex = st.selectbox("Giới tính", ["male", "female"],
                           index=0 if profile["sex"] == "male" else 1,
                           format_func=lambda x: SEX_VI[x], key="profile_sex")
        weight = st.number_input("Cân nặng (kg)", min_value=1.0, max_value=300.0,
                                 value=float(profile["weight"]), key="profile_weight")
    with c3:
        af_keys = list(ACTIVITY_VI.keys())
        af_index = af_keys.index(profile["activity_factor"]) if profile["activity_factor"] in af_keys else 0
        activity_factor = st.selectbox("Mức độ vận động", af_keys, index=af_index,
                                       format_func=lambda x: ACTIVITY_VI[x],
                                       key="profile_activity_factor")
        goal = st.selectbox("Mục tiêu sức khỏe", list(GOAL_VI.keys()),
                            index=list(GOAL_VI.keys()).index(profile["goal"]) if profile["goal"] in GOAL_VI else 1,
                            format_func=lambda x: GOAL_VI[x], key="profile_goal")

    profile = {"age": age, "sex": sex, "weight": weight, "height": height,
               "activity_factor": activity_factor, "goal": goal}
    st.session_state.user_profile = profile

    bmi = calculate_bmi(weight, height)
    tdee = calculate_tdee_mifflin_st_jeor(weight, height, age, sex, activity_factor)
    user_rda = compute_rda(profile, tdee)
    st.session_state.user_rda = user_rda
    bmi_label, bmi_color = ui.bmi_classify(bmi)

    st.markdown(ui.divider_dot("Chỉ số của bạn"), unsafe_allow_html=True)

    st.markdown(ui.kpi_row([
        ui.kpi_card("BMI", f"{bmi:.1f}", unit="kg/m²", sub=f"Phân loại: {bmi_label}",
                    icon="⚖️", accent="amber"),
        ui.kpi_card("TDEE", f"{tdee:.0f}", unit="kcal/ngày", sub="Tổng năng lượng tiêu hao",
                    icon="🔥", accent="red"),
        ui.kpi_card("Calo mục tiêu", f"{user_rda['Calories']:.0f}", unit="kcal/ngày",
                    sub=f"Mục tiêu: {GOAL_VI[goal]}", icon="🎯", accent="green"),
    ]), unsafe_allow_html=True)

    st.markdown(ui.kpi_row([
        ui.kpi_card("Protein mục tiêu", f"{user_rda['Protein']:.0f}", unit="g/ngày",
                    sub="1.6 g/kg cân nặng", icon="🍗", accent="blue"),
        ui.kpi_card("Chất béo", f"{user_rda['Fat']:.0f}", unit="g/ngày",
                    sub="25% năng lượng", icon="🥑", accent="purple"),
        ui.kpi_card("Đường (giới hạn)", f"{user_rda['Sugar']:.0f}", unit="g/ngày",
                    sub="khuyến nghị tối đa", icon="🍬", accent="teal"),
        ui.kpi_card("Muối (giới hạn)", f"{user_rda['Salt']:.0f}", unit="g/ngày",
                    sub="khuyến nghị tối đa", icon="🧂", accent="red"),
    ]), unsafe_allow_html=True)

    left, right = st.columns([1.05, 1], gap="large")
    with left:
        st.markdown(f'<div class="result-panel"><div class="section-title">📏 Thước đo BMI</div>'
                    f'{ui.bmi_gauge(bmi)}</div>', unsafe_allow_html=True)
        st.markdown(ui.macro_target_panel(user_rda, user_rda["Calories"]), unsafe_allow_html=True)
    with right:
        st.markdown('<div class="section-title">📈 Dự báo cân nặng 12 tuần</div>'
                    'Nếu duy trì mức thặng hụt/thặng dư calo của mục tiêu hiện tại '
                    '(±500 kcal/ngày ≈ ±0.45 kg/tuần):', unsafe_allow_html=True)
        st.plotly_chart(ui.weight_projection_fig(weight, height, goal),
                        width="stretch")


# ═════════════════════════════════════════════════════════════════════
# TAB 2 — QUÉT MÓN ĂN
# ═════════════════════════════════════════════════════════════════════
def render_scan_tab():
    st.markdown(ui.section_header(
        "🍜 Quét món ăn",
        "Chọn một món ăn mẫu để thử ngay, hoặc tải ảnh/video từ thiết bị của bạn."),
        unsafe_allow_html=True)

    col_conf, col_note = st.columns([1.1, 1.6], gap="large")
    with col_conf:
        confidence = float(st.slider(
            "Ngưỡng tin cậy (confidence threshold)",
            min_value=10, max_value=100, value=50,
            help="Cao hơn → dự đoán chính xác hơn nhưng có thể bỏ sót món. "
                 "Thấp hơn → phát hiện được nhiều vật thể hơn.")) / 100
    with col_note:
        with st.expander("🚩 Nên chọn ngưỡng tin cậy bao nhiêu?"):
            st.markdown("""
            - **Ngưỡng cao (≥ 50%)**: mô hình chỉ báo những món chắc chắn — độ chính xác cao, ít báo nhầm, nhưng có thể bỏ sót món khó nhận diện.
            - **Ngưỡng thấp (< 50%)**: phát hiện được nhiều vật thể hơn (độ bao phủ cao), nhưng dễ báo nhầm món không có trong ảnh.
            - **📊 Màu dinh dưỡng**: giá trị được đánh giá theo **hệ thống đèn giao thông** — xem giải thích bên dưới.
            """)

    import volume_integration as _vi
    import pipeline_config as _pc
    _foodsam_ok = os.path.isdir(os.path.join(_pc.FOODSAM_REPO, "env"))
    _config_default = getattr(_pc, "SEGMENTATION_BACKEND", "sam2")

    def _backend_label(b):
        if b == "sam2":
            label = "SAM2 — box-prompted, nhanh"
        else:
            label = "FoodSAM — SAM2 + SETR FoodSeg103 + ingredient"
        if b == _config_default:
            label += "  ★ mặc định (pipeline_config)"
        return label

    seg_backend = st.selectbox(
        "Backend phân đoạn (ước lượng khẩu phần)",
        ["sam2", "foodsam"],
        format_func=_backend_label,
        index=1 if _config_default == "foodsam" else 0,
        help=("SAM2: phân đoạn theo bbox YOLO (nhanh). FoodSAM: chạy trong môi "
              "trường riêng, thêm nhãn ngữ nghĩa món và tách nguyên liệu "
              "(trứng, rau...) ngoài bbox món — chậm hơn (~1-3 phút/ảnh). "
              "Giá trị ban đầu lấy từ SEGMENTATION_BACKEND trong pipeline_config.py."
              + ("" if _foodsam_ok else " ⚠️ Môi trường FoodSAM chưa sẵn sàng — "
                 "chọn FoodSAM sẽ fallback về dinh dưỡng phần chuẩn.")),
    )
    st.session_state.seg_backend = seg_backend

    with st.expander("🚦 Hệ thống màu giao thông cho giá trị dinh dưỡng"):
        ui.traffic_legend(st)

    st.markdown(ui.divider_dot("Chọn nguồn đầu vào"), unsafe_allow_html=True)

    model1 = load_model()

    input_mode = st.radio(
        "Nguồn ảnh / video",
        ["🖼️ Ảnh mẫu (Demo)", "📁 Tải ảnh lên", "🎥 Video", "📷 Webcam", "📡 IP Camera"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if input_mode == "🖼️ Ảnh mẫu (Demo)":
        _render_demo_mode(confidence, model1)
    elif input_mode == "📁 Tải ảnh lên":
        _render_image_mode(confidence, model1)
    elif input_mode == "🎥 Video":
        _render_video_mode(confidence, model1)
    elif input_mode == "📷 Webcam":
        _render_webcam_mode(confidence, model1)
    else:
        _render_ip_camera_mode(confidence, model1)


def _render_demo_mode(confidence, model1):
    """Chọn ảnh mẫu có sẵn và phân tích ngay — không cần upload."""
    demo_images = sorted(ui.DEMO_DIR.glob("*.jpg")) + sorted(ui.DEMO_DIR.glob("*.png"))
    if not demo_images:
        st.info("📁 Chưa có ảnh mẫu trong thư mục `assets/demo/`. "
                "Hãy thêm ảnh món ăn (đặt tên theo món, vd `pho.jpg`) để tạo bộ demo.")
        return

    st.markdown("**Chọn một món ăn mẫu bên dưới:**", unsafe_allow_html=True)
    cols = st.columns(min(len(demo_images), 6))
    choice = st.session_state.get("demo_choice")
    for idx, img_path in enumerate(demo_images):
        with cols[idx % len(cols)]:
            st.image(str(img_path), width="stretch")
            label = img_path.stem.replace("_", " ").title()
            is_selected = choice == str(img_path)
            if st.button(label, key=f"demo_{img_path.stem}",
                         width="stretch",
                         type="primary" if is_selected else "secondary"):
                st.session_state.demo_choice = str(img_path)
                choice = str(img_path)

    if choice:
        st.success(f"✅ Đã chọn: **{Path(choice).stem.replace('_', ' ').title()}** — "
                   "nhấn **Dự đoán** bên dưới để phân tích.")
        # Gọi mỗi lần rerun (giống chế độ upload) để luồng Dự đoán/Đặt lại
        # trong detect_image giữ được trạng thái.
        detect_image(confidence, uploaded_file=Path(choice), model=model1)
    else:
        st.info("👆 Bấm vào một món ăn để chọn, sau đó nhấn **Dự đoán** để phân tích.")


def _render_image_mode(confidence, model1):
    with st.expander("📖 Hướng dẫn: tải ảnh lên"):
        st.markdown("""
        - Tải ảnh món ăn từ máy (**PNG/JPG/JPEG**) — ảnh chụp ngang, đủ sáng cho kết quả tốt nhất.
        - Sau khi dự đoán: ảnh có khung nhận diện, bảng dinh dưỡng từng món, biểu đồ macro và các nút tải xuống (ảnh / CSV / JSON).
        - Nếu ảnh có cảnh quan đủ tốt, app sẽ ước lượng **khối lượng thực tế** bằng SAM2 + bản đồ chiều sâu (mất thêm ~1 phút).
        """)
    uploaded_file = st.file_uploader("Chọn ảnh món ăn", accept_multiple_files=False,
                                     type=["png", "jpg", "jpeg"])
    if uploaded_file:
        detect_image(confidence, model=model1, uploaded_file=uploaded_file)


def _render_video_mode(confidence, model1):
    with st.expander("📖 Hướng dẫn: video & YouTube"):
        st.markdown("""
        - Tải video (**MP4**) từ máy, hoặc dán liên kết **YouTube / YouTube Shorts** để dự đoán trực tiếp.
        - Kết quả: tổng dinh dưỡng các món xuất hiện trong video + file CSV.
        """)
    uploaded_clip = st.file_uploader("Chọn video", accept_multiple_files=False, type=["mp4"])
    if uploaded_clip:
        detect_video(conf=confidence, uploaded_file=uploaded_clip, model=model1)
    else:
        st.markdown("##### 🔗 Hoặc dán liên kết YouTube")
        with st.form("youtube_form"):
            col1, col2 = st.columns([0.8, 0.2], gap="medium")
            with col1:
                youtube_url = st.text_input("URL", label_visibility="collapsed",
                                            placeholder="https://youtu.be/LNwODJXcvt4")
            with col2:
                submitted = st.form_submit_button("Dự đoán", width="stretch", type="primary")
        if submitted and youtube_url:
            _display_detected_frame(conf=confidence, model=model1, youtube_url=youtube_url)


def _render_webcam_mode(confidence, model1):
    with st.expander("📖 Hướng dẫn: webcam"):
        st.markdown("""
        - Dùng [streamlit-webrtc](https://github.com/whitphx/streamlit-webrtc) để kết nối webcam thực tế.
        - Chọn nguồn camera rồi bật **START** — món ăn sẽ được nhận diện trực tiếp.
        - Không tạo file kết quả vì quá trình chạy liên tục.
        """)
    detect_webcam(confidence, model=model1)


def _render_ip_camera_mode(confidence, model1):
    with st.expander("📖 Hướng dẫn: IP Camera (RTSP)"):
        st.markdown("""
        - Nhập địa chỉ **RTSP** của camera (đã mở truy cập từ mạng ngoài).
        - Định dạng: `user:mật khẩu@địa-chỉ-ip:554/...`
        """)
    with st.form("ip_camera_form"):
        st.text("Nhập địa chỉ RTSP của camera:")
        col1, col2 = st.columns([2, 8])
        with col1:
            st.write("rtsp://admin:")
        with col2:
            address = st.text_input("Địa chỉ RTSP", label_visibility="collapsed",
                                    placeholder="hd543211@192.168.14.106:554/Streaming/channels/101")
        col1, col2 = st.columns([2, 1.35])
        with col1:
            submitted = st.form_submit_button("Kết nối", type="primary")
        with col2:
            cancel = st.form_submit_button("Ngắt kết nối")

        if submitted:
            if address:
                detect_camera(confidence, model1, address=address)
            else:
                st.error("Vui lòng nhập địa chỉ RTSP hợp lệ")
        if cancel:
            if address:
                detect_camera(confidence, model1, address="")
                st.toast("Đã ngắt kết nối", icon="✅")


# ═════════════════════════════════════════════════════════════════════
# TAB 3 — AI NUTRITIONIST (CHAT)
# ═════════════════════════════════════════════════════════════════════
def render_chat_tab():
    st.markdown(ui.section_header(
        "💬 Trợ lý Dinh dưỡng AI",
        "Hỏi về calo, chất béo, thực đơn lành mạnh — trợ lý hiểu bối cảnh bữa ăn "
        "vừa quét và thể trạng của bạn."), unsafe_allow_html=True)

    provider = st.session_state.get("llm_provider", "Gemini")
    key_status = {
        "Gemini": bool(st.session_state.get("gemini_api_key")),
        "Cerebras": bool(st.session_state.get("cerebras_api_key")),
        "OpenRouter": bool(st.session_state.get("openrouter_api_key")),
        "Cloudflare": bool(st.session_state.get("cloudflare_account_id")
                           and st.session_state.get("cloudflare_api_token")),
    }
    chips = [f'Nhà cung cấp: <b>{provider}</b>']
    chips.append("✅ Đã kết nối" if key_status.get(provider) else "🔒 Chưa nhập API key (Sidebar)")
    chips_html = "".join(f'<span class="metric-chip">{c}</span>' for c in chips)
    st.markdown(f'<div class="metric-chip-row">{chips_html}</div>', unsafe_allow_html=True)

    llm_provider = provider
    api_key = st.session_state.get("gemini_api_key", "")
    cerebras_api_key = st.session_state.get("cerebras_api_key", "")
    openrouter_api_key = st.session_state.get("openrouter_api_key", "")
    cloudflare_account_id = st.session_state.get("cloudflare_account_id", "")
    cloudflare_api_token = st.session_state.get("cloudflare_api_token", "")

    if llm_provider == "Gemini":
        if not api_key:
            st.info("💡 **Gợi ý**: nhập **Gemini API Key** ở thanh bên để kích hoạt trợ lý. "
                    "Lấy key miễn phí từ [Google AI Studio](https://aistudio.google.com/).")
            _render_chat_welcome(
                "Xin chào! Tôi là Trợ lý Dinh dưỡng AI. 🥗\n\n"
                "Sau khi bạn cấu hình khóa API ở thanh bên, tôi có thể giúp bạn:\n"
                "- Phân tích hàm lượng calo và dinh dưỡng trong thực đơn của bạn.\n"
                "- Đưa ra lời khuyên ăn uống lành mạnh phù hợp với các món ăn Việt Nam.\n"
                "- Thiết kế chế độ ăn kiêng, tăng cơ, giảm mỡ,...\n\n"
                "*Hãy nhập API Key ở thanh bên để bắt đầu trò chuyện nhé!*")
            return
        try:
            genai.configure(api_key=api_key)
        except Exception as e:
            st.error(f"Lỗi cấu hình API Key: {e}")
            return
    elif llm_provider == "Cerebras":
        if not cerebras_api_key:
            st.info("💡 **Gợi ý**: nhập **Cerebras API Key** ở thanh bên để kích hoạt trợ lý.")
            _render_chat_welcome(
                "Xin chào! Tôi là Trợ lý Dinh dưỡng AI (sử dụng Cerebras). 🥗\n\n"
                "Sau khi bạn cấu hình khóa API ở thanh bên, tôi có thể tư vấn dinh dưỡng "
                "cho bạn với tốc độ cực nhanh!\n"
                "*Hãy nhập Cerebras API Key ở thanh bên để bắt đầu trò chuyện nhé!*")
            return
    elif llm_provider == "OpenRouter":
        if not openrouter_api_key:
            st.info("💡 **Gợi ý**: nhập **OpenRouter API Key** ở thanh bên để kích hoạt trợ lý.")
            _render_chat_welcome(
                "Xin chào! Tôi là Trợ lý Dinh dưỡng AI (sử dụng OpenRouter). 🥗\n\n"
                "Sau khi bạn cấu hình khóa API ở thanh bên, tôi có thể tư vấn dinh dưỡng "
                "cho bạn bằng hàng trăm mô hình AI khác nhau!\n"
                "*Hãy nhập OpenRouter API Key ở thanh bên để bắt đầu trò chuyện nhé!*")
            return
    elif llm_provider == "Cloudflare":
        if cloudflare_account_id and cloudflare_account_id.strip().startswith("cfut_"):
            st.error("⚠️ **Nhập nhầm**: chuỗi `cfut_...` là **Cloudflare API Token**, không phải "
                     "**Account ID**! Vui lòng nhập đúng chuỗi 32 ký tự **Account ID** "
                     "(Cloudflare Dashboard → Workers & Pages, cột bên phải).")
            return
        if not cloudflare_account_id or not cloudflare_api_token:
            st.info("💡 **Gợi ý**: nhập **Cloudflare Account ID** và **API Token** ở thanh bên "
                    "để kích hoạt Cloudflare Workers AI.")
            _render_chat_welcome(
                "Xin chào! Tôi là Trợ lý Dinh dưỡng AI (sử dụng Cloudflare Workers AI). 🥗\n\n"
                "Sau khi bạn cấu hình Account ID và API Token ở thanh bên, tôi có thể tư vấn "
                "dinh dưỡng bằng các mô hình AI chạy trên mạng lưới toàn cầu của Cloudflare "
                "(miễn phí 10,000 neurons/ngày)!\n"
                "*Hãy nhập thông tin ở thanh bên để bắt đầu trò chuyện nhé!*")
            return

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    chat_container = st.container(height=620, border=True)

    with chat_container:
        for message in st.session_state.chat_messages:
            content_str = str(message.get("content", "")) if message.get("content") is not None else ""
            if content_str and content_str.strip() and content_str.strip().lower() != "none":
                with st.chat_message(message["role"]):
                    if message.get("reasoning_details"):
                        with st.expander("💭 Suy nghĩ của AI (Reasoning)"):
                            st.write(message["reasoning_details"])
                    st.markdown(content_str)

        selected_suggestion = None
        if not st.session_state.chat_messages:
            st.markdown("**💡 Câu hỏi gợi ý:**")
            suggestions = [
                "Món phở bò chứa bao nhiêu calo và protein?",
                "Gợi ý thực đơn tăng cơ với các món ăn Việt",
                "Bữa ăn có bún chả và chả giò có lành mạnh không?",
                "Làm sao để giảm cân mà vẫn ăn cơm tấm?"
            ]
            last_dishes = st.session_state.get("last_detected_dishes")
            if last_dishes:
                dishes_names = list(last_dishes.keys())
                if dishes_names:
                    suggestions.insert(0, f"Tư vấn dinh dưỡng cho bữa ăn vừa quét ({', '.join(dishes_names)})")
                    if len(suggestions) > 4:
                        suggestions.pop()
            cols = st.columns(2)
            for idx, sug in enumerate(suggestions):
                col = cols[idx % 2]
                if col.button(sug, key=f"sug_{idx}", width="stretch"):
                    selected_suggestion = sug

    prompt = st.chat_input("Hỏi tôi bất cứ điều gì về dinh dưỡng món Việt...")

    if selected_suggestion:
        prompt = selected_suggestion

    if prompt:
        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)
            st.session_state.chat_messages.append({"role": "user", "content": prompt})

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                with st.spinner("Đang suy nghĩ..."):
                    try:
                        last_dishes = st.session_state.get("last_detected_dishes")
                        last_nutri = st.session_state.get("last_total_nutrition")
                        user_profile = st.session_state.get("user_profile")
                        user_rda = st.session_state.get("user_rda")
                        meal_context = ""
                        if last_dishes and last_nutri and user_profile and user_rda:
                            facts = build_structured_facts(last_nutri, user_profile, user_rda, last_dishes)
                            meal_context = (
                                f"\n[BỐI CẢNH BỮA ĂN VỪA PHÁT HIỆN DƯỚI DẠNG JSON FACTS]:\n"
                                f"{json.dumps(facts, indent=2, ensure_ascii=False)}\n"
                                f"Hãy sử dụng các sự thật (facts) có cấu trúc này để trả lời nếu người dùng hỏi về món ăn hiện tại, bữa ăn của họ, hoặc xin lời khuyên dinh dưỡng."
                            )

                        rag_context = retrieve_context(
                            query=prompt,
                            detected_foods=last_dishes,
                            top_k=3
                        )

                        system_context = (
                            "Bạn là một chuyên gia tư vấn dinh dưỡng AI chuyên nghiệp chuyên về ẩm thực Việt Nam.\n"
                            "Nhiệm vụ của bạn là tư vấn dinh dưỡng cho người dùng dựa trên câu hỏi của họ, "
                            "phân tích calo, chất béo, carb, protein, đường, muối của các món ăn Việt Nam "
                            "và đề xuất các mẹo ăn uống lành mạnh (ví dụ: bớt nước lèo khi ăn phở, ăn thêm rau xà lách...).\n"
                            "Hãy trả lời bằng tiếng Việt, giọng điệu lịch sự, khoa học, thực tế và ngắn gọn dễ hiểu.\n"
                            "Không nói dông dài, đi thẳng vào vấn đề chính. Chỉ đưa ra câu trả lời trực tiếp bằng tiếng Việt, không lặp lại bất kỳ mô tả vai trò, nhiệm vụ hay cấu hình hệ thống nào.\n"
                            f"{rag_context}"
                            f"{meal_context}"
                        )

                        if llm_provider.startswith("Cerebras"):
                            model_id = st.session_state.get("cerebras_model", "gpt-oss-120b")
                            import requests
                            headers = {
                                "Authorization": f"Bearer {cerebras_api_key}",
                                "Content-Type": "application/json"
                            }
                            payload = {
                                "model": model_id,
                                "messages": [{"role": "system", "content": system_context}] + st.session_state.chat_messages
                            }
                            res = requests.post("https://api.cerebras.ai/v1/chat/completions", headers=headers, json=payload)
                            if res.status_code == 200:
                                response_text = res.json()["choices"][0]["message"]["content"]
                                message_placeholder.markdown(response_text)
                                st.session_state.chat_messages.append({"role": "assistant", "content": response_text})
                            else:
                                message_placeholder.markdown(f"❌ Lỗi từ Cerebras API (Mã lỗi {res.status_code}): {res.text}")
                            return
                        if llm_provider == "OpenRouter":
                            model_id = st.session_state.get("openrouter_model", "google/gemini-2.5-flash:free")
                            import requests
                            headers = {
                                "Authorization": f"Bearer {openrouter_api_key}",
                                "Content-Type": "application/json"
                            }
                            payload_messages = []
                            for msg in st.session_state.chat_messages:
                                m = {"role": msg["role"], "content": msg["content"]}
                                if "reasoning_details" in msg and msg["reasoning_details"]:
                                    m["reasoning_details"] = msg["reasoning_details"]
                                payload_messages.append(m)
                            payload = {
                                "model": model_id,
                                "messages": [{"role": "system", "content": system_context}] + payload_messages,
                                "reasoning": {"enabled": True},
                                "max_tokens": 4000
                            }
                            res = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
                            if res.status_code == 200:
                                data = res.json()
                                assistant_msg = data["choices"][0]["message"]
                                response_text = assistant_msg.get("content", "")
                                reasoning_details = assistant_msg.get("reasoning_details")
                                if reasoning_details:
                                    with st.expander("💭 Suy nghĩ của AI (Reasoning)"):
                                        st.write(reasoning_details)
                                message_placeholder.markdown(response_text)
                                st.session_state.chat_messages.append({
                                    "role": "assistant",
                                    "content": response_text,
                                    "reasoning_details": reasoning_details
                                })
                            else:
                                message_placeholder.markdown(f"❌ Lỗi từ OpenRouter API (Mã lỗi {res.status_code}): {res.text}")
                            return
                        if llm_provider == "Cloudflare":
                            cf_account_id = st.session_state.get("cloudflare_account_id", "").strip()
                            cf_api_token = st.session_state.get("cloudflare_api_token", "").strip()
                            model_id = st.session_state.get("cloudflare_model", "@cf/meta/llama-3.1-8b-instruct").strip()
                            import requests
                            headers = {
                                "Authorization": f"Bearer {cf_api_token}",
                                "Content-Type": "application/json"
                            }
                            payload_messages = []
                            if system_context and str(system_context).strip():
                                payload_messages.append({"role": "system", "content": str(system_context).strip()})
                            for msg in st.session_state.chat_messages:
                                role = str(msg.get("role", "user"))
                                content = msg.get("content")
                                if content is None:
                                    continue
                                if isinstance(content, list):
                                    text_parts = [p.get("text", "") if isinstance(p, dict) else str(p) for p in content]
                                    content = " ".join(text_parts)
                                else:
                                    content = str(content)
                                if content.strip():
                                    payload_messages.append({"role": role, "content": content.strip()})
                            url = f"https://api.cloudflare.com/client/v4/accounts/{cf_account_id}/ai/v1/chat/completions"
                            res = requests.post(url, headers=headers, json={"model": model_id, "messages": payload_messages})
                            # If v1 OpenAI endpoint is not enabled or returns 404/400, fallback to direct run endpoint
                            if res.status_code in [404, 400]:
                                url_fallback = f"https://api.cloudflare.com/client/v4/accounts/{cf_account_id}/ai/run/{model_id}"
                                res_fallback = requests.post(url_fallback, headers=headers, json={"messages": payload_messages})
                                if res_fallback.status_code == 200:
                                    res = res_fallback
                            if res.status_code == 200:
                                data = res.json()
                                response_text = ""
                                if isinstance(data, dict):
                                    if "choices" in data and isinstance(data["choices"], list) and len(data["choices"]) > 0:
                                        msg_obj = data["choices"][0].get("message", {})
                                        if isinstance(msg_obj, dict) and msg_obj.get("content"):
                                            response_text = str(msg_obj["content"])
                                    if not response_text and "result" in data and isinstance(data["result"], dict):
                                        if data["result"].get("response"):
                                            response_text = str(data["result"]["response"])
                                        elif "choices" in data["result"] and isinstance(data["result"]["choices"], list) and len(data["result"]["choices"]) > 0:
                                            msg_obj = data["result"]["choices"][0].get("message", {})
                                            if isinstance(msg_obj, dict) and msg_obj.get("content"):
                                                response_text = str(msg_obj["content"])
                                if not response_text or not response_text.strip() or response_text.strip().lower() == "none":
                                    response_text = "❌ Không nhận được câu trả lời từ mô hình Cloudflare AI này (Phản hồi rỗng). Vui lòng đổi sang mô hình `@cf/meta/llama-3.1-8b-instruct` ở thanh bên."
                                message_placeholder.markdown(response_text)
                                st.session_state.chat_messages.append({"role": "assistant", "content": response_text})
                            else:
                                err_msg = f"❌ Lỗi từ Cloudflare Workers AI (Mã lỗi {res.status_code}): {res.text}"
                                if res.status_code == 403 and "Free plan" in res.text:
                                    err_msg = f"❌ **Lỗi 403**: Mô hình `{model_id}` yêu cầu tài khoản Cloudflare Workers trả phí (Paid plan).\n\n👉 Vui lòng chuyển sang mô hình miễn phí như `@cf/meta/llama-3.1-8b-instruct` hoặc `@cf/deepseek-ai/deepseek-r1-distill-qwen-32b` ở thanh bên."
                                message_placeholder.markdown(err_msg)
                            return

                        # Gemini (mặc định)
                        working_model = st.session_state.get("working_model_name", "")
                        if working_model:
                            model = genai.GenerativeModel(working_model, system_instruction=system_context)
                            chat = model.start_chat(history=[])
                            response = chat.send_message(prompt)
                            response_text = response.text
                            message_placeholder.markdown(response_text)
                            st.session_state.chat_messages.append({"role": "assistant", "content": response_text})
                        else:
                            models_to_try = []
                            try:
                                models = list(genai.list_models())
                                api_names = [m.name.replace("models/", "", 1) for m in models if "generateContent" in m.supported_generation_methods]
                                api_names.sort(reverse=True)
                                models_to_try = api_names
                            except Exception:
                                pass
                            if not models_to_try:
                                models_to_try = ["gemini-3.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]
                            for latest in ["gemini-3.5-flash", "gemini-2.0-flash"]:
                                if latest in models_to_try:
                                    models_to_try.remove(latest)
                                    models_to_try.insert(0, latest)
                            success = False
                            last_error = ""
                            response_text = ""
                            for model_name in models_to_try:
                                try:
                                    model = genai.GenerativeModel(model_name, system_instruction=system_context)
                                    chat = model.start_chat(history=[])
                                    response = chat.send_message(prompt)
                                    response_text = response.text
                                    success = True
                                    st.session_state.working_model_name = model_name
                                    break
                                except Exception as e:
                                    last_error = str(e)
                                    if "404" in last_error or "429" in last_error or "quota" in last_error.lower() or "not found" in last_error.lower() or "available" in last_error.lower() or "support" in last_error.lower():
                                        continue
                                    else:
                                        break
                            if success:
                                message_placeholder.markdown(response_text)
                                st.session_state.chat_messages.append({"role": "assistant", "content": response_text})
                            else:
                                message_placeholder.markdown(f"❌ Đã xảy ra lỗi khi kết nối với Gemini API: {last_error}")
                    except Exception as e:
                        message_placeholder.markdown(f"❌ Đã xảy ra lỗi khi kết nối với API: {e}")


def _render_chat_welcome(welcome_text):
    with st.chat_message("assistant"):
        st.markdown(welcome_text)
    if not st.session_state.get("chat_messages"):
        st.markdown("**💡 Câu hỏi gợi ý:**")
        suggestions = [
            "Món phở bò chứa bao nhiêu calo và protein?",
            "Gợi ý thực đơn tăng cơ với các món ăn Việt",
            "Bữa ăn có bún chả và chả giò có lành mạnh không?",
            "Làm sao để giảm cân mà vẫn ăn cơm tấm?"
        ]
        cols = st.columns(2)
        for idx, sug in enumerate(suggestions):
            cols[idx % 2].button(sug, key=f"welcome_sug_{idx}", width="stretch")


# ═════════════════════════════════════════════════════════════════════
# TAB 4 — VỀ MÔ HÌNH & DỮ LIỆU
# ═════════════════════════════════════════════════════════════════════
def render_about_tab():
    st.markdown(ui.section_header(
        "📖 Về FoodDetector AI",
        "Hệ thống nhận diện món ăn Việt Nam và phân tích dinh dưỡng tự động."), unsafe_allow_html=True)

    st.markdown("""
    <div class="about-callout">
        <p><b>🍜 FoodDetector AI</b> là hệ thống hoàn chỉnh gồm 4 giai đoạn: <b>nhận diện</b> món ăn bằng
        YOLOv26 fine-tuned trên bộ dữ liệu VietFood67 → <b>phân đoạn</b> vùng món ăn bằng SAM 2 →
        <b>ước lượng thể tích & khối lượng</b> bằng bản đồ chiều sâu (Depth Anything V2) →
        <b>tính dinh dưỡng và tư vấn</b> cá nhân hóa bằng AI (RAG + LLM).</p>
        <p>Ứng dụng hỗ trợ quét từ <b>ảnh, video, webcam và IP camera</b>, đánh giá dinh dưỡng theo
        <b>hệ thống đèn giao thông</b> và so sánh với nhu cầu hằng ngày (RDA) được tính từ BMI/TDEE của người dùng.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(ui.metric_chips([
        ("30,360", "ảnh trong dataset"),
        ("68", "lớp món ăn"),
        ("123,644", "ảnh sau tăng cường"),
        ("mAP50 · 0.95", "độ chính xác"),
    ]), unsafe_allow_html=True)

    st.divider()
    st.markdown("#### 🗃️ Bộ dữ liệu VietFood67")
    st.markdown("""
    Bộ dữ liệu gồm **30,360 ảnh** với **68 lớp**, bao gồm thêm một lớp nhận diện **con người** —
    giúp hệ thống theo dõi hoạt động ăn uống và cho kết quả toàn diện hơn (thời lượng ăn có thể
    suy ra từ việc phát hiện người cùng với các món ăn).

    Bộ dữ liệu được chia **70% / 20% / 10%**: **21,264** ảnh train · **6,074** ảnh test · **3,022** ảnh valid.
    """)

    import class_names as cn_module
    rows = "\n".join(
        f"| {i} | {c['name']} |" for i, c in enumerate(cn_module.class_names[:34]))
    rows2 = "\n".join(
        f"| {i + 34} | {c['name']} |" for i, c in enumerate(cn_module.class_names[34:]))
    header = "| ID | Món ăn |\n|----|--------|"
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown(header + "\n" + rows)
    with col2:
        st.markdown(header + "\n" + rows2)

    st.divider()
    st.markdown("#### 🔍 Thu thập dữ liệu")
    st.markdown("""
    Ảnh được thu thập từ nhiều nguồn khác nhau để đảm bảo tính đa dạng và phức tạp:
    - **Google, Facebook, ShopeeFood**: phần lớn ảnh được tìm theo tên món với từ khóa như "review đồ ăn", "nấu ăn".
    - **YouTube**: trích xuất khung hình từ video/shorts với sự hỗ trợ của công cụ [Roboflow](https://roboflow.com/).
    - **Bộ sưu tập cá nhân**: một số ảnh chụp bằng điện thoại để mô phỏng điều kiện nhận diện thực tế.
    """)

    st.markdown("#### ✍️ Gán nhãn dữ liệu")
    st.markdown("""
    Quá trình gán nhãn khung giới hạn (bounding box) sử dụng công cụ [Roboflow](https://roboflow.com/).
    Để tăng tốc, một mô hình YOLOv10m được huấn luyện trên một phần dữ liệu rồi dùng tính năng
    **Auto Label** để gán nhãn tự động phần còn lại, sau đó kiểm tra thủ công.
    """)

    st.markdown("#### ⚙️ Xử lý & tăng cường dữ liệu")
    st.markdown("""
    Các kỹ thuật tăng cường được áp dụng để mô hình tổng quát tốt và giải quyết mất cân bằng giữa các lớp:
    - **Cắt bounding box**: zoom tối thiểu 5%, tối đa 20%.
    - **Lật bounding box**: lật theo chiều dọc.
    - **Chỉnh độ sáng**: từ −15% đến +15%.
    - **Augmentation Mosaic**.

    Tổng cộng thu được **123,644 ảnh** sau quá trình tăng cường để huấn luyện mô hình.
    """)


# ═════════════════════════════════════════════════════════════════════
# APP SHELL
# ═════════════════════════════════════════════════════════════════════
def main():
    ui.inject_css()
    st.markdown('<div id="top-section"></div>', unsafe_allow_html=True)
    st.markdown(ui.navbar(), unsafe_allow_html=True)

    render_sidebar()

    st.markdown(ui.hero(), unsafe_allow_html=True)

    tab_health, tab_scan, tab_chat, tab_about = st.tabs(
        ["🧬 Thể trạng", "🍜 Quét món ăn", "💬 AI Nutritionist", "📖 Về mô hình"])

    with tab_health:
        render_health_tab()
    with tab_scan:
        render_scan_tab()
    with tab_chat:
        render_chat_tab()
    with tab_about:
        render_about_tab()

    # Nút cuộn lên đầu trang
    st.markdown(f"""
    <a href="#top-section" class="top-button" title="Lên đầu trang">
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 448 512"><path d="M240.971 130.524l194.343 194.343c9.373 9.373 9.373 24.569 0 33.941l-22.667 22.667c-9.357 9.357-24.522 9.375-33.901.04L224 227.495 69.255 381.516c-9.379 9.335-24.544 9.317-33.901-.04l-22.667-22.667c-9.373-9.373-9.373-24.569 0-33.941L207.03 130.525c9.372-9.373 24.568-9.373 33.941-.001z"/></svg>
    </a>
    <script>
    function smoothScroll(event, targetId) {{
        event.preventDefault();
        const targetElement = document.getElementById(targetId);
        if (targetElement) {{
            targetElement.scrollIntoView({{ behavior: 'smooth' }});
        }}
    }}
    </script>
    """, unsafe_allow_html=True)

    st.markdown('<div style="text-align:center; font-size:11px; color: var(--text-faint); '
                'padding: 1.2rem 0 2rem 0;">🍜 FoodDetector AI — Nhận diện món Việt & '
                'tư vấn dinh dưỡng cá nhân hóa · Giá trị dinh dưỡng chỉ mang tính tham khảo</div>',
                unsafe_allow_html=True)


if __name__ == "__main__":
    main()
