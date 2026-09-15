import av
from ultralytics import YOLO
import streamlit as st
import cv2
from PIL import Image, ImageOps
import tempfile
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer, VideoTransformerBase

import numpy as np
from io import BytesIO
import queue

import time
from collections import deque

import csv
import json
import re
import requests
import datetime
import os
import io
import base64
import pandas as pd
import plotly.graph_objects as go

import ui_components as ui

from class_names import class_names

# RAG: lazy import — only loaded when chatbot is active
# Import FoodKnowledgeBase via get_knowledge_base() to avoid heavy startup cost
def _get_rag_kb():
    """Safely import and return the RAG knowledge base. Returns None if unavailable."""
    try:
        from rag.knowledge_base import get_knowledge_base
        return get_knowledge_base()
    except Exception as e:
        print(f"[RAG] Knowledge base unavailable: {e}")
        return None

# Thứ tự chất dinh dưỡng hiển thị trên chip/card
_NUTRIENT_ORDER = ["Calories", "Protein", "Carbs", "Fat", "Saturates", "Sugar", "Salt"]


def create_fig(image, detected=False):

    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    image_data_uri = base64.b64encode(buffer.getvalue()).decode()

    fig = go.Figure()
    fig.add_layout_image(
        dict(
            source=f"data:image/png;base64,{image_data_uri}",
            x=0,
            y=image.size[1],
            xref="x",
            yref="y",
            sizex=image.size[0],
            sizey=image.size[1],
            layer="below"
        )
    )

    fig.update_layout(
        xaxis_range=[0, image.size[0]],
        yaxis_range=[0, image.size[1]],
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Be Vietnam Pro, Segoe UI, sans-serif", color="#B3A795", size=12),
        margin=dict(l=0, r=0, b=18, t=0),
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        annotations=[
            dict(
                x=0.5,
                y=-0.06,
                showarrow=False,
                text="📸 Ảnh đã nhận diện" if detected else "🖼️ Ảnh gốc",
                font=dict(size=13, color="#B3A795"),
                xref="paper",
                yref="paper"
            )
        ]
    )

    return fig


def _nutri_chip(nutrient, value, color, desc):
    """Chip traffic-light cho một chất dinh dưỡng (giá trị + nhãn tiếng Việt)."""
    unit = "kcal" if nutrient == "Calories" else "g"
    value_str = "—" if value is None else f"{value:.1f} {unit}"
    return ui.traffic_chip(nutrient, value_str, desc, color)


def _dish_card_for_detection(class_id, class_name, confident, serving, bbox_image_html):
    """Card món ăn dùng chung cho video/webcam/IP/YouTube (khẩu phần tham chiếu)."""
    if class_name == "Con nguoi (Human)":
        return (f"<div class='dish-human'>🙋 Phát hiện <b>Con người</b> — "
                f"độ tin cậy <b>{confident}%</b></div>", None)

    nutrition = class_names[int(class_id)]["nutrition"]
    if not nutrition:
        return None, None

    chips = []
    for nutrient in ["Calories", "Protein", "Fat", "Saturates", "Sugar", "Salt"]:
        color, desc = get_nutri_score_color(nutrient, nutrition.get(nutrient), serving)
        chips.append(_nutri_chip(nutrient, nutrition.get(nutrient), color, desc))

    thumb = bbox_image_html or ""
    card = ui.dish_card(class_name, confident, ui.vi_serving(serving), thumb, "".join(chips))
    return card, nutrition


def _totals_kpi_row(total_nutrition, std=None):
    """Hàng KPI tổng dinh dưỡng (có ± sai số nếu có)."""
    def _std(key):
        return f"{std[key]:.1f}" if std and std.get(key) else ""
    cards = [
        ui.kpi_card("Calo", f"{total_nutrition['Calories']:.1f}", unit="kcal", std=_std("Calories"),
                    icon="🔥", accent="amber"),
        ui.kpi_card("Protein", f"{total_nutrition.get('Protein', 0):.1f}", unit="g", std=_std("Protein"),
                    icon="🍗", accent="blue"),
        ui.kpi_card("Carb", f"{total_nutrition.get('Carbs', 0):.1f}", unit="g", std=_std("Carbs"),
                    icon="🍚", accent="teal"),
        ui.kpi_card("Chất béo", f"{total_nutrition['Fat']:.1f}", unit="g", std=_std("Fat"),
                    icon="🥑", accent="purple"),
        ui.kpi_card("Bão hòa", f"{total_nutrition['Saturates']:.1f}", unit="g", std=_std("Saturates"),
                    icon="🥓", accent="red"),
        ui.kpi_card("Đường", f"{total_nutrition['Sugar']:.1f}", unit="g", std=_std("Sugar"),
                    icon="🍬", accent="green"),
        ui.kpi_card("Muối", f"{total_nutrition['Salt']:.1f}", unit="g", std=_std("Salt"),
                    icon="🧂", accent="red"),
    ]
    return ui.kpi_row(cards)


def _new_total_nutrition():
    return {
        "Calories": 0,
        "Protein": 0,
        "Carbs": 0,
        "Fat": 0,
        "Saturates": 0,
        "Sugar": 0,
        "Salt": 0,
    }

def convert_youtube_url(url):
    pattern = r"(?:https?://)?(?:www\.)?(?:youtube\.com/shorts/|youtube\.com/watch\?v=|youtu\.be/)([\w\-]{11})"
    match = re.search(pattern, url)
    
    if match:
        video_id = match.group(1)
        return f"https://youtu.be/{video_id}"
    return None


def _display_detected_frame(conf, model, youtube_url=""):
    if youtube_url:
        youtube_id = convert_youtube_url(youtube_url)
        if youtube_id:
            valid_url = youtube_id
            st.toast("Đang kết nối", icon="🕒")

            try:
                results = model(source=valid_url, stream=True, conf=conf, imgsz=640, save=True, device="cpu", vid_stride=1, half=False)
                items = {}  # class_name -> card_html
                total_nutrition = _new_total_nutrition()
                nutrition_data = []
                current_time = datetime.datetime.now()
                time_format = current_time.strftime("%d-%m-%Y")

                stop_button = st.button("⏹ Dừng")
                stop_pressed = False

                st_frame = st.empty()

                frame_count = 0
                start_time = time.time()

                totals_placeholder = st.empty()
                results_placeholder = st.empty()
                st.markdown('<div class="section-title">🍽️ Kết quả nhận diện</div>'
                            '<div class="section-sub">Các món ăn được phát hiện trong video '
                            '(cập nhật trực tiếp).</div>', unsafe_allow_html=True)

                for r in results:
                    im_bgr = r.plot()
                    frame_count += 1
                    elapsed_time = time.time() - start_time
                    if elapsed_time >= 1.0:
                        fps = frame_count / elapsed_time
                        start_time = time.time()
                        frame_count = 0
                    cv2.putText(im_bgr, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 4)

                    im_rgb = Image.fromarray(im_bgr[..., ::-1])
                    im_rgb_resized = im_rgb.resize((640, 640))
                    st_frame.image(im_rgb_resized, caption='Video dự đoán', width="stretch")
                    for pred in r.boxes:
                        class_id = int(pred.cls[0].item())
                        class_name = class_names[int(class_id)]["name"]
                        confident = int(round(pred.conf[0].item(), 2)*100)
                        serving = class_names[int(class_id)]["serving_type"]

                        if isinstance(pred.xyxy, torch.Tensor):
                            boxes = pred.xyxy.cpu().numpy()
                        else:
                            boxes = pred.xyxy.numpy()

                        image_np = r.orig_img

                        bounding_box_images = extract_bounding_box_image(image_np, boxes)

                        bbox_image_html = ""
                        if bounding_box_images:
                            bbox_image = bounding_box_images[0]
                            bbox_image_pil = Image.fromarray(cv2.cvtColor(bbox_image, cv2.COLOR_BGR2RGB))

                            buffered = io.BytesIO()
                            bbox_image_pil.save(buffered, format="JPEG")
                            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                            bbox_image_html = ui.thumb_img(img_str)

                        if class_name not in items:
                            card_html, nutrition = _dish_card_for_detection(
                                class_id, class_name, confident, serving, bbox_image_html)
                            if card_html is not None:
                                items[class_name] = card_html
                                if nutrition:
                                    for key in total_nutrition:
                                        if key in nutrition:
                                            total_nutrition[key] += nutrition[key]
                                    nutrition_data.append({
                                        "name": class_name,
                                        "serving": ui.vi_serving(serving),
                                        "conf": confident,
                                        "nutrition": {k: nutrition.get(k) for k in _NUTRIENT_ORDER},
                                    })

                    totals_placeholder.markdown(_totals_kpi_row(total_nutrition), unsafe_allow_html=True)
                    if items:
                        cards = "".join(items.values())
                        results_placeholder.markdown(
                            f'<div class="result-panel">{cards}</div>', unsafe_allow_html=True)

                    if stop_button:
                        stop_pressed = True
                        stop_button = None
                        break

                st.session_state.last_detected_dishes = {name: 1 for name in items}
                st.session_state.last_total_nutrition = total_nutrition

                with tempfile.NamedTemporaryFile(delete=False, suffix='.csv', dir=tempfile.gettempdir()) as csv_file:
                    csv_filename = csv_file.name
                with open(csv_filename, mode='w', newline='', encoding='utf-8-sig') as file:
                    writer = csv.writer(file)
                    writer.writerow(["Tên món", "Khẩu phần", "Tin cậy (%)", "Calo (kcal)", "Protein (g)",
                                     "Carb (g)", "Chất béo (g)", "Bão hòa (g)", "Đường (g)", "Muối (g)"])
                    for d in nutrition_data:
                        n = d["nutrition"]
                        writer.writerow([d["name"], d["serving"], d["conf"],
                                         n["Calories"], n["Protein"], n["Carbs"], n["Fat"],
                                         n["Saturates"], n["Sugar"], n["Salt"]])
                with open(csv_filename, "rb") as file:
                    the_csv = file.read()

                st.toast("Hoàn tất dự đoán. Kết quả đã sẵn sàng để tải xuống.", icon="✅")
                download_csv = st.download_button(label="📊 Kết quả dự đoán (CSV)",
                                data=the_csv,
                                file_name=f"{time_format}.csv",
                                width="stretch",
                                key=f"download_csv3_button_{time_format}")
                if download_csv:
                    os.remove(csv_filename)
            except ConnectionError as e:
                st.error(f"Không thể mở luồng video YouTube: {e}")
        else:
            st.error("Liên kết YouTube không hợp lệ hoặc không trích xuất được video ID.")
    else:
        st.error("Vui lòng nhập liên kết YouTube.")

@st.cache_resource
def load_model():
    modelpath = r"./model/yolov26/best.onnx"
    
    model = YOLO(modelpath, task="detect")
    return model

def resize_image(image):
    return image.resize((640, 640))

# Define color variables
COLOR_HIGH = "#FF4A3F"    # Red for high values
COLOR_MEDIUM = "#FECB02"  # Yellow for medium values
COLOR_LOW = "#85BB2F"     # Green for low values

# Traffic-light thresholds: (low_below, medium_below) per nutrient.
# "per_100g" for per-100 g values; "portion" for any per-serving quantity
# ("1 serving", "reference serving (450 g)", volume-based "estimated portion ...").
_NUTRI_THRESHOLDS = {
    "per_100g": {
        "Calories":  (100, 200),
        "Fat":       (3, 17.5),
        "Saturates": (1.5, 5),
        "Sugar":     (5, 22.5),
        "Salt":      (0.3, 1.5),
    },
    "portion": {
        "Calories":  (150, 300),
        "Fat":       (5, 21),
        "Saturates": (2, 6),
        "Sugar":     (6, 27),
        "Salt":      (0.4, 1.8),
    },
}


def get_nutri_score_color(nutrient, value, serving_type):
    """Traffic-light color for a nutrient value. Never returns None."""
    if value is None:
        return COLOR_LOW, "—"
    thresholds = _NUTRI_THRESHOLDS["per_100g"] if serving_type == "per 100g" else _NUTRI_THRESHOLDS["portion"]
    limits = thresholds.get(nutrient)
    if limits is None:
        return COLOR_LOW, "—"
    low, medium = limits
    if value < low:
        return COLOR_LOW, "Low"
    elif value < medium:
        return COLOR_MEDIUM, "Medium"
    return COLOR_HIGH, "High"


def calculate_nutrient_percentage(nutrition):
    total_nutrition_value = (
        nutrition.get('Calories', 0) + 
        nutrition.get('Fat', 0) + 
        nutrition.get('Saturates', 0) + 
        nutrition.get('Sugar', 0) + 
        nutrition.get('Salt', 0) +
        nutrition.get('Protein', 0)
    )
    if total_nutrition_value == 0:
        return {key: 0 for key in nutrition}

    percentage = {
        "Calories": (nutrition.get('Calories', 0) / total_nutrition_value) * 100,
        "Fat": (nutrition.get('Fat', 0) / total_nutrition_value) * 100,
        "Saturates": (nutrition.get('Saturates', 0) / total_nutrition_value) * 100,
        "Sugar": (nutrition.get('Sugar', 0) / total_nutrition_value) * 100,
        "Salt": (nutrition.get('Salt', 0) / total_nutrition_value) * 100,
        "Protein": (nutrition.get('Protein', 0) / total_nutrition_value) * 100,
    }
    return percentage

import torch

def extract_bounding_box_image(image, boxes):
    h, w = image.shape[:2]
    extracted_images = []

    for box in boxes:
        if isinstance(box, torch.Tensor):
            box = box.cpu().numpy()
        x1, y1, x2, y2 = box
        startX, startY, endX, endY = int(x1), int(y1), int(x2), int(y2)
 
        startX, startY = max(0, startX), max(0, startY)
        endX, endY = min(w, endX), min(h, endY)

        bbox_image = image[startY:endY, startX:endX]
        if bbox_image.size > 0:
            extracted_images.append(bbox_image)
    return extracted_images

def calculate_bmi(weight_kg: float, height_cm: float) -> float:
    height_m = height_cm / 100
    return weight_kg / (height_m ** 2)

def calculate_tdee_mifflin_st_jeor(
    weight_kg: float, height_cm: float, age: int, sex: str, activity_factor: float
) -> float:
    """Mifflin-St Jeor equation. sex: 'male' or 'female'. activity_factor: 1.2 (sedentary)
    to 1.9 (very active) — standard multipliers, don't invent custom ones."""
    if sex == "male":
        bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age + 5
    elif sex == "female":
        bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age - 161
    else:
        raise ValueError("sex must be 'male' or 'female'")
    return bmr * activity_factor

def build_structured_facts(meal_nutrition: dict, user_profile: dict, rda: dict, detected_foods: dict = None) -> dict:
    """This JSON is the ENTIRE input the LLM will see for this meal.
    Nothing outside this dict should reach the LLM — if the LLM needs a new
    fact, add it here explicitly; don't let it infer things from partial data."""
    facts = {"deficits": {}, "excesses": {}, "on_target": [], "user_goal": user_profile["goal"]}

    for nutrient, consumed in meal_nutrition.items():
        target = rda.get(nutrient)
        if target is None:
            continue
        pct_of_target = consumed / target * 100
        if pct_of_target < 80:
            facts["deficits"][nutrient] = {"consumed": consumed, "target": target, "pct": pct_of_target}
        elif pct_of_target > 120:
            facts["excesses"][nutrient] = {"consumed": consumed, "target": target, "pct": pct_of_target}
        else:
            facts["on_target"].append(nutrient)

    if detected_foods:
        facts["detected_foods"] = detected_foods

    return facts

def retrieve_context(query: str, detected_foods: dict = None, top_k: int = 3) -> str:
    """
    Retrieve relevant nutrition knowledge chunks from the RAG vector store.

    Embeds the user query (optionally enriched with detected food names),
    searches Qdrant Cloud for the top-k most similar documents, and returns
    a formatted string ready to be injected into the LLM system context.

    Returns an empty string if RAG is unavailable or no relevant chunks found.
    """
    kb = _get_rag_kb()
    if kb is None:
        return ""

    # Enrich query with detected food names for better retrieval precision
    if detected_foods:
        food_names = ", ".join(list(detected_foods.keys())[:4])
        enriched_query = f"{query} {food_names}"
    else:
        enriched_query = query

    chunks = kb.retrieve(enriched_query, top_k=top_k)
    if not chunks:
        return ""

    formatted = "\n\n".join(
        f"[Tài liệu tham khảo {i + 1}]:\n{chunk}" for i, chunk in enumerate(chunks)
    )
    return (
        "\n\n[KIẾN THỨC DINH DƯỠNG THAM KHẢO TỪ CƠ SỞ DỮ LIỆU]:\n"
        f"{formatted}\n"
        "[Hãy ưu tiên sử dụng các thông tin trên để trả lời, chỉ sử dụng kiến thức nội tại nếu tài liệu không đủ.]\n"
    )


def generate_nutrition_advice(count_dict_names, total_nutrition):
    provider = st.session_state.get("llm_provider", "Gemini")
    
    st.markdown("### 🥗 AI Đánh Giá Dinh Dưỡng Bữa Ăn")
    advice_placeholder = st.empty()
    advice_placeholder.markdown("*Đang kết nối với AI để phân tích dinh dưỡng...*")

    # Fetch profile and rda
    user_profile = st.session_state.get("user_profile")
    user_rda = st.session_state.get("user_rda")
    
    # Calculate facts JSON using the rule engine
    facts = build_structured_facts(total_nutrition, user_profile, user_rda, count_dict_names)
    
    # The prompt consists strictly of the structured facts JSON
    import json
    prompt = json.dumps(facts, indent=2, ensure_ascii=False)

    # Construct system prompt context to interpret the structured facts JSON
    system_context = (
        "Bạn là một chuyên gia tư vấn dinh dưỡng AI chuyên nghiệp chuyên về ẩm thực Việt Nam.\n"
        "Người dùng sẽ cung cấp cho bạn một chuỗi JSON chứa các dữ liệu thực tế về bữa ăn của họ so với nhu cầu dinh dưỡng khuyến nghị hàng ngày (RDA) được tính toán dựa trên chỉ số BMI và TDEE của họ.\n"
        "Dữ liệu JSON bao gồm:\n"
        "- 'user_goal': Mục tiêu sức khỏe của người dùng.\n"
        "- 'deficits': Các chất dinh dưỡng bị thiếu hụt nghiêm trọng trong bữa ăn này so với nhu cầu hàng ngày.\n"
        "- 'excesses': Các chất dinh dưỡng bị dư thừa quá mức trong bữa ăn này so với nhu cầu hàng ngày.\n"
        "- 'on_target': Các chất dinh dưỡng đạt mục tiêu lý tưởng.\n"
        "- 'detected_foods': Các món ăn được nhận diện trong bữa ăn này.\n\n"
        "Nhiệm vụ của bạn là:\n"
        "1. Nhận xét ngắn gọn về các món ăn trong phần 'detected_foods'.\n"
        "2. Đánh giá bữa ăn này có lành mạnh hay không dựa trên các chất thiếu hụt ('deficits'), dư thừa ('excesses') hoặc đạt mục tiêu ('on_target') so với nhu cầu RDA hàng ngày.\n"
        "3. Đưa ra lời khuyên thiết thực (ăn thêm gì, bớt gì) để giúp người dùng đạt được mục tiêu 'user_goal'.\n\n"
        "Hãy trả lời trực tiếp bằng tiếng Việt, giọng điệu lịch sự, khoa học, thực tế và ngắn gọn dễ hiểu (khoảng 3-4 đoạn ngắn). Không lặp lại hay giải thích bất kỳ định dạng hệ thống nào."
    )

    if provider.startswith("Cerebras"):
        cerebras_api_key = st.session_state.get("cerebras_api_key", "")
        if not cerebras_api_key:
            st.info("💡 **Gợi ý**: Hãy cấu hình **Cerebras API Key** ở thanh bên (Sidebar) để nhận tư vấn bằng Cerebras.")
            advice_placeholder.empty()
            return

        model_id = st.session_state.get("cerebras_model", "gpt-oss-120b")

        headers = {
            "Authorization": f"Bearer {cerebras_api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": model_id,
            "messages": [
                {"role": "system", "content": system_context},
                {"role": "user", "content": prompt}
            ]
        }

        try:
            import requests
            response = requests.post("https://api.cerebras.ai/v1/chat/completions", headers=headers, json=payload)
            if response.status_code == 200:
                response_text = response.json()["choices"][0]["message"]["content"]
                advice_placeholder.markdown(response_text)
            else:
                advice_placeholder.markdown(f"❌ Lỗi từ Cerebras API (Mã lỗi {response.status_code}): {response.text}")
        except Exception as e:
            advice_placeholder.markdown(f"❌ Đã xảy ra lỗi khi kết nối với Cerebras API: {e}")
        return
    if provider == "OpenRouter":
        openrouter_api_key = st.session_state.get("openrouter_api_key", "")
        if not openrouter_api_key:
            st.info("💡 **Gợi ý**: Hãy cấu hình **OpenRouter API Key** ở thanh bên (Sidebar) để nhận tư vấn bằng OpenRouter.")
            advice_placeholder.empty()
            return

        model_id = st.session_state.get("openrouter_model", "google/gemini-2.5-flash:free")

        headers = {
            "Authorization": f"Bearer {openrouter_api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": model_id,
            "messages": [
                {"role": "system", "content": system_context},
                {"role": "user", "content": prompt}
            ],
            "reasoning": {"enabled": True},
            "max_tokens": 4000
        }

        try:
            import requests
            response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
            if response.status_code == 200:
                data = response.json()
                assistant_msg = data["choices"][0]["message"]
                response_text = assistant_msg.get("content", "")
                reasoning_details = assistant_msg.get("reasoning_details")
                
                if reasoning_details:
                    with st.expander("💭 Suy nghĩ của AI (Reasoning)"):
                        st.write(reasoning_details)
                advice_placeholder.markdown(response_text)
            else:
                advice_placeholder.markdown(f"❌ Lỗi từ OpenRouter API (Mã lỗi {response.status_code}): {response.text}")
        except Exception as e:
            advice_placeholder.markdown(f"❌ Đã xảy ra lỗi khi kết nối với OpenRouter API: {e}")
        return

    # Fall back to Gemini
    api_key = st.session_state.get("gemini_api_key", "")
    if not api_key:
        st.info("💡 **Gợi ý**: Hãy nhập **Gemini API Key** ở thanh bên (Sidebar) để nhận đánh giá món ăn và phân tích dinh dưỡng tự động từ AI!")
        advice_placeholder.empty()
        return

    working_model = st.session_state.get("working_model_name", "")
    
    # Configure API
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
    except Exception as e:
        advice_placeholder.markdown(f"❌ Lỗi cấu hình API Key: {e}")
        return

    if working_model:
        try:
            model = genai.GenerativeModel(working_model, system_instruction=system_context)
            response = model.generate_content(prompt)
            advice_placeholder.markdown(response.text)
            return
        except Exception:
            st.session_state.working_model_name = ""

    # Fallback search loop
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
            response = model.generate_content(prompt)
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
        advice_placeholder.markdown(response_text)
    else:
        advice_placeholder.markdown(f"❌ Đã xảy ra lỗi khi kết nối với Gemini API: {last_error}")

def _find_volume_estimation(volume_estimations, bbox):
    """Match a YOLO box to the volume-pipeline estimation with the largest IoU."""
    best, best_iou = None, 0.0
    x1, y1, x2, y2 = np.ravel(bbox)[:4]
    for est in volume_estimations:
        ex1, ey1, ex2, ey2 = est.bbox
        ix1, iy1 = max(x1, ex1), max(y1, ey1)
        ix2, iy2 = min(x2, ex2), min(y2, ey2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        union = ((x2 - x1) * (y2 - y1)) + ((ex2 - ex1) * (ey2 - ey1)) - inter
        iou = inter / union if union > 0 else 0.0
        if iou > best_iou:
            best, best_iou = est, iou
    return best if best_iou > 0.5 else None


def _match_suppressed_dish(suppressed_dishes, bbox):
    """Return the suppressed-dish audit record whose bbox best overlaps a YOLO
    box (mixed-plate contradiction heuristic removed it from accounting)."""
    best, best_iou = None, 0.0
    x1, y1, x2, y2 = np.ravel(bbox)[:4]
    for sup in suppressed_dishes or []:
        sb = sup.get("bbox") or [0, 0, 0, 0]
        ex1, ey1, ex2, ey2 = sb[0], sb[1], sb[2], sb[3]
        ix1, iy1 = max(x1, ex1), max(y1, ey1)
        ix2, iy2 = min(x2, ex2), min(y2, ey2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        union = ((x2 - x1) * (y2 - y1)) + ((ex2 - ex1) * (ey2 - ey1)) - inter
        iou = inter / union if union > 0 else 0.0
        if iou > best_iou:
            best, best_iou = sup, iou
    return best if best_iou > 0.5 else None


def _show_volume_visualizations(volume_result):
    """Hiển thị các ảnh phân tích của volume pipeline: segmentation, depth, scale."""
    if volume_result is None:
        return
    panels = [
        (getattr(volume_result, "mask_overlay_b64", None),
         "Phân đoạn món ăn (SAM2)"),
        (getattr(volume_result, "depth_colored_b64", None),
         "Bản đồ độ sâu (Depth Anything V2)"),
        (getattr(volume_result, "scale_overlay_b64", None),
         "Thước đo tỷ lệ (mm/px)"),
    ]
    panels = [(b64, cap) for b64, cap in panels if b64]
    if not panels:
        return
    st.markdown('<div class="section-title">🔬 Phân tích khối lượng ảnh</div>'
                '<div class="section-sub">Kết quả phân đoạn, bản đồ độ sâu và thước đo '
                'tỷ lệ được dùng để ước lượng khẩu phần.</div>', unsafe_allow_html=True)
    cols = st.columns(len(panels), gap="small")
    for col, (b64, caption) in zip(cols, panels):
        with col:
            img = Image.open(io.BytesIO(base64.b64decode(b64)))
            st.image(np.asarray(img), caption=caption)


def detect_image_result(detected_image, model, source_image=None, image_path=None,
                        bbox_scale=None, yolo_time_s=None):
    boxes = detected_image[0].boxes
    seg_backend = st.session_state.get("seg_backend", "sam2")

    # FoodSAM ingredient recovery: with ZERO YOLO boxes the FoodSAM backend
    # can still produce nutrition from SETR semantic components (HomeCook
    # YOLO-miss case) — don't short-circuit into the empty state.
    foodsam_recovery = (
        not boxes
        and source_image is not None
        and seg_backend == "foodsam"
    )

    if boxes or foodsam_recovery:
        detected_img_arr_RGB = detected_image[0].plot()[:, :, ::1]
        detected_img_arr_BGR = detected_image[0].plot()[:, :, ::-1]
        fig_detected = create_fig(detected_img_arr_BGR, detected=True)
        st.plotly_chart(fig_detected, width="stretch")

        current_time = datetime.datetime.now()
        time_format = current_time.strftime("%d-%m-%Y")

        # ── Volume-based portion estimation (Image tab) ──
        # Falls back to fixed per-serving nutrition when unavailable or failed.
        volume_result = None
        volume_estimations = []
        volume_status_note = ""
        suppressed_dishes = []
        volume_time_s = None
        if source_image is not None:
            import volume_integration
            if volume_integration.volume_pipeline_available(seg_backend):
                t_volume_start = time.time()
                try:
                    backend_label = ("FoodSAM (SAM2 + SETR + nguyên liệu)"
                                     if seg_backend == "foodsam" else "SAM2")
                    time_hint = ("1–3 phút" if seg_backend == "foodsam" else "~1 phút")
                    with st.spinner(f"Đang ước lượng khẩu phần ({backend_label} "
                                    "+ chiều sâu — có thể mất tới " + time_hint + ")..."):
                        v_dets = volume_integration.extract_yolo_detections(
                            detected_image, class_names, bbox_scale=bbox_scale)
                        volume_result = volume_integration.estimate_nutrition_volume(
                            source_image, v_dets, image_path, backend=seg_backend
                        )
                except Exception as e:
                    print(f"[Volume] Pipeline error: {e}")
                    volume_result = None
                # Đo cả trường hợp lỗi/timeout để hiển thị thời gian đã trôi
                volume_time_s = time.time() - t_volume_start
            else:
                volume_integration.last_error = (
                    f"Backend '{seg_backend}' chưa sẵn sàng" if seg_backend != "sam2"
                    else "Volume pipeline dependencies unavailable")
            if volume_result is not None:
                volume_estimations = list(volume_result.estimations)
                suppressed_dishes = list(
                    getattr(volume_result, "suppressed_dishes", []) or [])
                volume_status_note = ui.volume_note_html(
                    volume_result.scale_result.scale_source,
                    requested_backend=seg_backend,
                    backend_used=getattr(volume_result, "seg_backend_used", ""),
                    fallback_used=getattr(volume_result, "seg_fallback_used", False),
                    seg_error=getattr(volume_result, "seg_error", None))
            else:
                volume_status_note = ui.volume_note_html(
                    None, error=getattr(volume_integration, "last_error", None),
                    requested_backend=seg_backend)


        # with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg', dir='/tmp') as img_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg', dir=tempfile.gettempdir()) as img_file:
            img_filename = img_file.name
            cv2.imwrite(img_filename, detected_img_arr_RGB)
        with open(img_filename, 'rb') as file:
            the_img = file.read()
        
            detection_results = ""
            count_results = ""
            count_dict = {}
            food_names = []
            nutrition_data = []
            confidences = []
            counts = []
            items = []
            matched_estimation_ids = set()
            total_nutrition = _new_total_nutrition()

            total_nutrition_std = {key: 0.0 for key in total_nutrition}

            total_nutrition_placeholder = st.empty()
            
            for r in detected_image[0]:
                for box in r.boxes:
                    class_id = int(box.cls[0].item())
                    class_name = class_names[int(class_id)]["name"] 
                    food_names.append(class_name)
                    conf = int(round(box.conf[0].item(), 2)*100)
                    confidences.append(conf)
                    serving = class_names[int(class_id)]["serving_type"]

                    if isinstance(box.xyxy, torch.Tensor):
                        boxes = box.xyxy.cpu().numpy()
                    else:
                        boxes = box.xyxy.numpy()
                
                    image_np = r.orig_img 
                    
                    bounding_box_images = extract_bounding_box_image(image_np, boxes)

                    bbox_image_html = ""
                    if bounding_box_images:
                        bbox_image = bounding_box_images[0]
                        bbox_image_pil = Image.fromarray(cv2.cvtColor(bbox_image, cv2.COLOR_BGR2RGB))

                        buffered = io.BytesIO()
                        bbox_image_pil.save(buffered, format="JPEG")
                        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                        bbox_image_html = ui.thumb_img(img_str)

                    

                    if class_id in count_dict:
                        count_dict[class_id] += 1
                    else:
                        count_dict[class_id] = 1

                    thumb_html = bbox_image_html

                    if class_name == "Con nguoi (Human)":
                        items.append((f"🙋 Con người · {conf}%",
                                      f"<div class='dish-human'>🙋 Phát hiện <b>Con người</b> — "
                                      f"độ tin cậy <b>{conf}%</b></div>"))

                    else:
                        # Volume estimations live in the SOURCE image's
                        # coordinates (aspect-preserving); YOLO boxes here are
                        # in the 640x640 display space — rescale before IoU
                        # matching or every match fails and items fall back
                        # to reference servings.
                        if bbox_scale is not None:
                            sx, sy = bbox_scale
                            match_box = (boxes[0][0] * sx, boxes[0][1] * sy,
                                         boxes[0][2] * sx, boxes[0][3] * sy)
                        else:
                            match_box = boxes[0]
                        # Mixed-plate contradiction: this YOLO hypothesis was
                        # removed from accounting — show the audit card, never
                        # per-serving nutrition for it.
                        sup = _match_suppressed_dish(suppressed_dishes, match_box)
                        if sup is not None:
                            card_html = ui.dish_card(
                                class_name, conf,
                                "món bị ẩn (mixed-plate contradiction)",
                                thumb_html, "",
                                "<span class='portion-data-note'>⚠️ Giả thuyết món "
                                f"<b>{class_name}</b> mâu thuẫn với thành phần "
                                "FoodSAM phát hiện — dinh dưỡng ước lượng từ "
                                "các nguyên liệu bên dưới.</span>")
                            items.append((f"⚠️ {class_name} · {conf}% — món bị ẩn",
                                          card_html))
                            continue
                        volume_est = _find_volume_estimation(volume_estimations, match_box)
                        if volume_est is not None:
                            matched_estimation_ids.add(id(volume_est))
                        if volume_est is not None and volume_est.nutrition:
                            nutrition = dict(volume_est.nutrition)
                            nutrition_std = dict(volume_est.nutrition_std)
                            serving = (f"khẩu phần ước lượng {volume_est.mass_g:.0f} ± "
                                       f"{volume_est.mass_std_g:.0f} g")
                            portion_mass = f"{volume_est.mass_g:.0f} ± {volume_est.mass_std_g:.0f} g"
                            portion_volume = f"{volume_est.volume_cm3:.0f} cm³"
                            # Dùng ảnh crop có segmentation mask + nhãn thay cho bbox trơn
                            if getattr(volume_est, "crop_b64", None):
                                thumb_html = ui.thumb_img(volume_est.crop_b64)
                            portion_note = ui.portion_badge_html(
                                volume_est.volume_cm3, volume_est.mass_g,
                                volume_est.mass_std_g, volume_est.confidence_level)
                            if volume_est.warnings:
                                shown = " · ".join(
                                    w.replace("⚠", "").strip() for w in volume_est.warnings[:2]
                                )
                                portion_note += (
                                    f"<br><span class='portion-data-note'>⚠ {shown}</span>"
                                )
                        else:
                            nutrition = dict(class_names[int(class_id)]["nutrition"])
                            nutrition_std = None
                            serving = class_names[int(class_id)]["serving_type"]
                            portion_mass = "—"
                            portion_volume = "—"
                            portion_note = ""
                        if nutrition:
                            chips = []
                            for nutrient in ["Calories", "Protein", "Fat", "Saturates", "Sugar", "Salt"]:
                                color, desc = get_nutri_score_color(nutrient, nutrition.get(nutrient), serving)
                                chips.append(_nutri_chip(nutrient, nutrition.get(nutrient), color, desc))

                            item_label = (f"🍜 {class_name} · {conf}% — "
                                          f"{nutrition.get('Calories', 0):.0f} kcal")
                            card_html = ui.dish_card(class_name, conf, ui.vi_serving(serving),
                                                     thumb_html, "".join(chips), portion_note)
                            items.append((item_label, card_html))

                            for key in total_nutrition:
                                if key in nutrition:
                                    total_nutrition[key] += nutrition[key]
                                    if nutrition_std and key in nutrition_std:
                                        prev_sq = total_nutrition_std[key] ** 2
                                        total_nutrition_std[key] = (
                                            (prev_sq + nutrition_std[key] ** 2) ** 0.5
                                        )

                            nutrition_data.append({
                                "name": class_name,
                                "serving": ui.vi_serving(serving),
                                "conf": conf,
                                "nutrition": {k: nutrition.get(k) for k in _NUTRIENT_ORDER},
                                "mass": portion_mass if portion_mass != "—" else "",
                                "volume": portion_volume if portion_volume != "—" else "",
                                "method": volume_est.estimation_method if volume_est is not None else "",
                            })

            # ── FoodSAM ingredient extras (không có YOLO box) ──
            # Group per-class at PRESENTATION level only; totals stay the sum
            # of per-component estimations.
            extras_grouped = {}
            for est in volume_estimations:
                if getattr(est, "source", "yolo") != "foodsam_ingredient":
                    continue
                if id(est) in matched_estimation_ids:
                    continue
                g = extras_grouped.setdefault(est.class_name, {
                    "count": 0, "mass": 0.0, "mass_std": 0.0,
                    "volume": 0.0, "nutrition": {}, "nutrition_std": {},
                    "thumbs": [], "purity": [],
                })
                g["count"] += 1
                g["mass"] += est.mass_g
                g["mass_std"] = (g["mass_std"] ** 2 + est.mass_std_g ** 2) ** 0.5
                g["volume"] += est.volume_cm3
                for key, value in est.nutrition.items():
                    g["nutrition"][key] = g["nutrition"].get(key, 0) + value
                for key, value in (est.nutrition_std or {}).items():
                    g["nutrition_std"][key] = (
                        g["nutrition_std"].get(key, 0) ** 2 + value ** 2) ** 0.5
                if getattr(est, "crop_b64", None):
                    g["thumbs"].append(est.crop_b64)
                if getattr(est, "semantic_purity", None) is not None:
                    g["purity"].append(est.semantic_purity)
                # per-component rows go to exports
                nutrition_data.append({
                    "name": est.class_name,
                    "serving": f"FoodSAM ingredient ({est.component_id})",
                    "conf": int(round((est.semantic_purity or 0) * 100)),
                    "nutrition": {k: est.nutrition.get(k) for k in _NUTRIENT_ORDER},
                    "mass": f"{est.mass_g:.0f} ± {est.mass_std_g:.0f} g",
                    "volume": f"{est.volume_cm3:.0f} cm³",
                    "method": est.estimation_method,
                })

            for class_name, g in extras_grouped.items():
                purity_avg = (sum(g["purity"]) / len(g["purity"])) if g["purity"] else None
                label = (f"🥗 {class_name} ×{g['count']} — "
                         f"{g['nutrition'].get('Calories', 0):.0f} kcal")
                portion_note = ui.portion_badge_html(
                    g["volume"], g["mass"], g["mass_std"], "medium")
                thumb_html = ui.thumb_img(g["thumbs"][0]) if g["thumbs"] else ""
                card_html = ui.dish_card(
                    f"{class_name} ×{g['count']}",
                    int(round((purity_avg or 0) * 100)),
                    "FoodSAM ingredient — ngoài bbox YOLO",
                    thumb_html,
                    "".join(
                        _nutri_chip(n, g["nutrition"].get(n), *get_nutri_score_color(
                            n, g["nutrition"].get(n), f"{g['mass']:.0f} g"))
                        for n in ["Calories", "Protein", "Fat", "Saturates", "Sugar", "Salt"]
                    ),
                    portion_note,
                )
                items.append((label, card_html))

                for key in total_nutrition:
                    if key in g["nutrition"]:
                        total_nutrition[key] += g["nutrition"][key]
                        std_sq = g["nutrition_std"].get(key, 0) ** 2
                        prev_sq = total_nutrition_std[key] ** 2
                        total_nutrition_std[key] = (prev_sq + std_sq) ** 0.5


            # ── Hiển thị kết quả ──
            if not items and foodsam_recovery:
                st.info(
                    "🥗 YOLO không phát hiện món — FoodSAM cũng không tìm thấy "
                    "nguyên liệu đã map (trứng, cơm, rau...) trong ảnh này. "
                    "Thử ảnh chụp gần và đủ sáng hơn."
                )
            time_html = ui.time_note_html(
                yolo_s=yolo_time_s,
                pipeline_s=volume_time_s,
                backend_label=("FoodSAM" if seg_backend == "foodsam" else "SAM2")
                if volume_time_s is not None else None,
            )
            total_nutrition_placeholder.markdown(
                volume_status_note + time_html + _totals_kpi_row(total_nutrition, total_nutrition_std),
                unsafe_allow_html=True)

            _show_volume_visualizations(volume_result)

            col_chart, col_rda = st.columns([1, 1.15], gap="large")
            with col_chart:
                st.markdown('<div class="section-title">🍩 Tỷ lệ macro bữa ăn</div>',
                            unsafe_allow_html=True)
                st.plotly_chart(ui.donut_macro_fig(
                    total_nutrition.get("Protein", 0),
                    total_nutrition.get("Fat", 0),
                    total_nutrition.get("Carbs", 0),
                    total_nutrition.get("Calories", 0)), width="stretch")
            with col_rda:
                st.markdown('<div class="section-title">🎯 So với nhu cầu hằng ngày</div>',
                            unsafe_allow_html=True)
                user_rda_scan = st.session_state.get("user_rda")
                if user_rda_scan:
                    st.markdown(ui.rda_panel(total_nutrition, user_rda_scan), unsafe_allow_html=True)
                else:
                    st.info("💡 Nhập hồ sơ sức khỏe ở tab **Thể trạng** để so sánh bữa ăn "
                            "với mục tiêu dinh dưỡng của bạn.")

            # Bảng dữ liệu chi tiết từng món
            st.markdown('<div class="section-title">📋 Bảng dinh dưỡng chi tiết</div>',
                        unsafe_allow_html=True)
            df_rows = [{
                "Món ăn": d["name"],
                "Tin cậy (%)": d["conf"],
                "Khẩu phần": d["serving"],
                "Khối lượng (g)": d["mass"] or "—",
                "Thể tích (cm³)": d["volume"] or "—",
                "Calo (kcal)": d["nutrition"]["Calories"],
                "Protein (g)": d["nutrition"]["Protein"],
                "Carb (g)": d["nutrition"]["Carbs"],
                "Chất béo (g)": d["nutrition"]["Fat"],
                "Bão hòa (g)": d["nutrition"]["Saturates"],
                "Đường (g)": d["nutrition"]["Sugar"],
                "Muối (g)": d["nutrition"]["Salt"],
                "Phương pháp": d["method"],
            } for d in nutrition_data]
            if df_rows:
                st.dataframe(pd.DataFrame(df_rows), width="stretch", hide_index=True)
            else:
                st.info("Chỉ phát hiện con người trong ảnh — không có món ăn nào để thống kê.")

            # Chi tiết từng món
            st.markdown('<div class="section-title">🍽️ Chi tiết từng món</div>'
                        '<div class="section-sub">Màu của chip thể hiện mức độ dinh dưỡng theo '
                        'hệ thống đèn giao thông.</div>', unsafe_allow_html=True)
            for item_label, card_html in items:
                with st.expander(item_label):
                    st.markdown(card_html, unsafe_allow_html=True)

            # ── Tải xuống: ảnh / CSV / JSON ──
            csv_headers = ["Tên món", "Khẩu phần", "Tin cậy (%)", "Calo (kcal)", "Protein (g)",
                           "Carb (g)", "Chất béo (g)", "Bão hòa (g)", "Đường (g)", "Muối (g)",
                           "Khối lượng (g)", "Thể tích (cm³)", "Phương pháp ước lượng"]
            with tempfile.NamedTemporaryFile(delete=False, suffix='.csv', dir=tempfile.gettempdir()) as csv_file:
                csv_filename = csv_file.name
            with open(csv_filename, mode='w', newline='', encoding='utf-8-sig') as file:
                writer = csv.writer(file)
                writer.writerow(csv_headers)
                for d in nutrition_data:
                    n = d["nutrition"]
                    writer.writerow([d["name"], d["serving"], d["conf"],
                                     n["Calories"], n["Protein"], n["Carbs"], n["Fat"],
                                     n["Saturates"], n["Sugar"], n["Salt"],
                                     d["mass"], d["volume"], d["method"]])
            with open(csv_filename, 'rb') as file:
                the_csv = file.read()

            count_dict_names = {}
            for object_type, count in count_dict.items():
                count_dict_names[class_names[object_type]["name"]] = count

            json_payload = {
                "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
                "profile": st.session_state.get("user_profile"),
                "daily_targets": st.session_state.get("user_rda"),
                "seg_backend": st.session_state.get("seg_backend", "sam2"),
                "scale_source": volume_result.scale_result.scale_source if volume_result else None,
                "total_nutrition": total_nutrition,
                "total_nutrition_std": total_nutrition_std,
                "items": nutrition_data,
            }
            the_json = json.dumps(json_payload, ensure_ascii=False, indent=2).encode("utf-8")

        col1, col2, col3 = st.columns(3, gap="large")
        with col1:
            download_pic = st.download_button(label="📷 Ảnh dự đoán (JPG)",
                                    data=the_img,
                                    mime="image/jpg",
                                    file_name=f"{time_format}.jpg",
                                    width="stretch",
                                    key=f"download_pic_button_{time_format}")
            if download_pic:
                os.remove(img_filename)
        with col2:
            download_csv = st.download_button(label="📊 Kết quả dự đoán (CSV)",
                               data=the_csv,
                               file_name=f"{time_format}.csv",
                               width="stretch",
                               key=f"download_csv_button_{time_format}")
            if download_csv:
                os.remove(csv_filename)
        with col3:
            download_json = st.download_button(label="🧾 Dữ liệu đầy đủ (JSON)",
                               data=the_json,
                               mime="application/json",
                               file_name=f"{time_format}.json",
                               width="stretch",
                               key=f"download_json_button_{time_format}")
        st.divider()

        st.session_state.last_detected_dishes = count_dict_names
        st.session_state.last_total_nutrition = total_nutrition

        generate_nutrition_advice(count_dict_names, total_nutrition)

    else:
        st.markdown(ui.no_food_alert(), unsafe_allow_html=True)


def detect_image(conf, uploaded_file, model, url=False):
        if "button_clicked" not in st.session_state:
            st.session_state.button_clicked = False
        
        if "is_reset" not in st.session_state:
            st.session_state.is_reset = False
        
        if "show_image" not in st.session_state:
            st.session_state.show_image = True

        reset_button = None
        predict_button = None
        
        def toggle_button(reset = False):
            st.session_state.button_clicked = not st.session_state.button_clicked
            st.session_state.show_image = not st.session_state.show_image
            if reset == True:
                st.session_state.is_reset = not st.session_state.is_reset
        
        original_image = st.empty()

        if url==False:
            uploaded_image = Image.open(uploaded_file)
        else:
            response = requests.get(uploaded_file)
            response.raise_for_status()
            uploaded_image = Image.open(BytesIO(response.content))

        # Apply EXIF orientation ONCE here so display, YOLO and the volume
        # pipeline all share the same upright frame. Photos with orientation=6
        # (90° rotated) previously diverged: YOLO boxes were mapped onto a
        # transposed volume source — rotated overlays, wrong masks, tiny kcal.
        uploaded_image = ImageOps.exif_transpose(uploaded_image)

        resized_uploaded_image = resize_image(uploaded_image)

        # Volume pipeline source: aspect-preserving (the fixed 640x640 stretch
        # corrupts ArUco/scale geometry — a 900x700 photo stretched to 640x640
        # made the ArUco detector fire a false positive and distorted real
        # marker scales ~4x). Capped at 2000px for pipeline speed.
        # (EXIF orientation was already applied to uploaded_image above.)
        volume_source_image = uploaded_image.copy()
        volume_source_image.thumbnail((2000, 2000))
        bbox_scale = (volume_source_image.width / 640.0,
                      volume_source_image.height / 640.0)

        if st.session_state.show_image and not st.session_state.is_reset and not st.session_state.button_clicked:   
            original_image = st.image(resized_uploaded_image, output_format="JPEG", width="stretch")

        if not st.session_state.is_reset:
            col1, col2 = st.columns([0.8, 0.2], gap="large")
            with col1:
                if st.session_state.show_image and not st.session_state.button_clicked and not original_image == st.empty():
                    st.markdown("**🖼️ Ảnh gốc**")
                elif not st.session_state.show_image and st.session_state.button_clicked:
                    st.markdown("**📸 Ảnh dự đoán**")
            with col2:
                if not st.session_state.button_clicked:
                    predict_button = st.button("🔍 Dự đoán", width="stretch", type="primary", on_click=toggle_button)
                else:
                    reset_button = st.button("🔄 Đặt lại", width="stretch", type="primary", on_click=toggle_button, args=[True])
                    uploaded_file = None
        if st.session_state.show_image and st.session_state.is_reset and not st.session_state.button_clicked:
            st.session_state.is_reset = False

        if st.session_state.button_clicked and not reset_button:
            with st.spinner("Đang phân tích món ăn..."):
                t_yolo_start = time.time()
                detected_image = model.predict(resized_uploaded_image, conf=conf, imgsz=640, half=False)
                yolo_time_s = time.time() - t_yolo_start
                image_path = getattr(uploaded_file, "name", None) if not isinstance(uploaded_file, str) else None
                detect_image_result(detected_image, model,
                                    source_image=volume_source_image,
                                    image_path=image_path,
                                    bbox_scale=bbox_scale,
                                    yolo_time_s=yolo_time_s)

def detect_camera(conf, model, address):
    vid_cap = cv2.VideoCapture('rtsp://admin:' + address)
    vid_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    vid_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
    vid_cap.set(cv2.CAP_PROP_FPS, 15)
    fps = vid_cap.get(cv2.CAP_PROP_FPS)

    while True:
        if vid_cap.isOpened():
            st.toast("Đã kết nối camera", icon="✅")
            break
        else:
            vid_cap.release()
            return
    try:
        st_frame = st.empty()

        items = {}
        total_nutrition = _new_total_nutrition()

        totals_placeholder = st.empty()
        results_placeholder = st.empty()
        st.markdown('<div class="section-title">🍽️ Kết quả nhận diện</div>'
                    '<div class="section-sub">Các món ăn được phát hiện từ IP camera '
                    '(cập nhật trực tiếp).</div>', unsafe_allow_html=True)

        frame_count = 0
        start_time = time.time()
        while True:
            success, image = vid_cap.read()
            if success:
                results = model.track(source=image, conf=conf, imgsz=640, save=False, device="cpu", stream=True, half=False)

                for r in results:
                    im_bgr = r.plot()
                    frame_count += 1
                    elapsed_time = time.time() - start_time
                    if elapsed_time >= 1.0:
                        fps = frame_count / elapsed_time
                        start_time = time.time()
                        frame_count = 0
                    cv2.putText(im_bgr, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    im_rgb = Image.fromarray(im_bgr[..., ::-1])
                    st_frame.image(im_rgb, caption='Camera IP', width="stretch")

                    for pred in r.boxes:
                        class_id = int(pred.cls[0].item())
                        class_name = class_names[int(class_id)]["name"]
                        confident = int(round(pred.conf[0].item(), 2)*100)
                        serving = class_names[int(class_id)]["serving_type"]

                        if isinstance(pred.xyxy, torch.Tensor):
                            boxes = pred.xyxy.cpu().numpy()
                        else:
                            boxes = pred.xyxy.numpy()

                        image_np = r.orig_img

                        bounding_box_images = extract_bounding_box_image(image_np, boxes)

                        bbox_image_html = ""
                        if bounding_box_images:
                            bbox_image = bounding_box_images[0]
                            bbox_image_pil = Image.fromarray(cv2.cvtColor(bbox_image, cv2.COLOR_BGR2RGB))

                            buffered = io.BytesIO()
                            bbox_image_pil.save(buffered, format="JPEG")
                            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                            bbox_image_html = ui.thumb_img(img_str)

                        if class_name not in items:
                            card_html, nutrition = _dish_card_for_detection(
                                class_id, class_name, confident, serving, bbox_image_html)
                            if card_html is not None:
                                items[class_name] = card_html
                                if nutrition:
                                    for key in total_nutrition:
                                        if key in nutrition:
                                            total_nutrition[key] += nutrition[key]

                totals_placeholder.markdown(_totals_kpi_row(total_nutrition), unsafe_allow_html=True)
                if items:
                    cards = "".join(items.values())
                    results_placeholder.markdown(
                        f'<div class="result-panel">{cards}</div>', unsafe_allow_html=True)

                st.session_state.last_detected_dishes = {name: 1 for name in items}
                st.session_state.last_total_nutrition = total_nutrition
            else:
                break
    except Exception as e:
        st.error(f"Lỗi tải video: {str(e)}")
    finally:
        vid_cap.release()

from typing import List, NamedTuple
result_queue = queue.Queue(maxsize=12)

class Detection(NamedTuple):
    class_id: int
    class_name: str
    confident: float
    serving: str
    bbox_image_html: str

class VideoTransformer(VideoTransformerBase):
    def __init__(self, conf, model):
        self.conf = conf
        self.model = model
        self.prev_time = time.time()
        self.frame_count = 0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        mirrored_frame = cv2.flip(img, 1)
        results = self.model(source=mirrored_frame, conf=self.conf, imgsz=640, save=False, device="cpu", stream=True, vid_stride=80, half=False)
        # results = self.model.track(source=mirrored_frame, conf=self.conf, imgsz=640, save=False, device="cpu", stream=True)
        detections = []
        for r in results:
            im_bgr = r.plot()
            for pred in r.boxes:
                class_id = int(pred.cls[0].item())
                if isinstance(pred.xyxy, torch.Tensor):
                    boxes = pred.xyxy.cpu().numpy()
                else:
                    boxes = pred.xyxy.numpy()
            
                image_np = r.orig_img 
                
                bounding_box_images = extract_bounding_box_image(image_np, boxes)

                bbox_image_html = ""
                if bounding_box_images:
                    bbox_image = bounding_box_images[0]
                    bbox_image_pil = Image.fromarray(cv2.cvtColor(bbox_image, cv2.COLOR_BGR2RGB))

                    buffered = io.BytesIO()
                    bbox_image_pil.save(buffered, format="JPEG")
                    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                    bbox_image_html = ui.thumb_img(img_str)
                detections.append(
                    Detection(
                        class_id = int(pred.cls[0].item()),
                        class_name = class_names[int(class_id)]["name"],
                        confident = int(round(pred.conf[0].item(), 2)*100),
                        serving = class_names[int(class_id)]["serving_type"],
                        bbox_image_html = bbox_image_html
                    )
                )

        im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
        
        self.frame_count += 1
        current_time = time.time()
        elapsed_time = current_time - self.prev_time

        if elapsed_time >= 1.0:
            fps = self.frame_count / elapsed_time
            self.prev_time = current_time
            self.frame_count = 0
        else:
            fps = self.frame_count / elapsed_time
  

        cv2.putText(im_rgb, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if not result_queue.full():
            result_queue.put(detections)

        return av.VideoFrame.from_ndarray(im_rgb, format="rgb24")   



def detect_webcam(conf, model):
    webrtc_ctx = webrtc_streamer(
        key="webcam_1",
        mode=WebRtcMode.SENDRECV,
        video_transformer_factory=lambda: VideoTransformer(conf, model),
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    if webrtc_ctx.state.playing:
        totals_placeholder = st.empty()
        results_placeholder = st.empty()
        st.markdown('<div class="section-title">🍽️ Kết quả nhận diện</div>'
                    '<div class="section-sub">Các món ăn được phát hiện từ webcam '
                    '(cập nhật trực tiếp).</div>', unsafe_allow_html=True)

        items = {}
        total_nutrition = _new_total_nutrition()

        while True:
            detections = result_queue.get()

            for detection in detections:
                class_id = detection.class_id
                class_name = detection.class_name
                confident = detection.confident
                serving = detection.serving
                bbox_image_html = detection.bbox_image_html

                if class_name not in items:
                    card_html, nutrition = _dish_card_for_detection(
                        class_id, class_name, confident, serving, bbox_image_html)
                    if card_html is not None:
                        items[class_name] = card_html
                        if nutrition:
                            for key in total_nutrition:
                                if key in nutrition:
                                    total_nutrition[key] += nutrition[key]

            totals_placeholder.markdown(_totals_kpi_row(total_nutrition), unsafe_allow_html=True)
            if items:
                cards = "".join(items.values())
                results_placeholder.markdown(
                    f'<div class="result-panel">{cards}</div>', unsafe_allow_html=True)

            st.session_state.last_detected_dishes = {name: 1 for name in items}
            st.session_state.last_total_nutrition = total_nutrition

    result_queue.queue.clear()


import onnxruntime as ort

model_path = "./model/yolov26/best.onnx"


@st.cache_resource
def load_onnx_model():
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    return session, input_name, output_name

def preprocess(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (640, 640))
    img = img.astype(np.float32) / 255.0

    img = img.astype(np.float16)


    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img



def postprocess(outputs, frame, original_size, conf, model):
    h, w, _ = original_size
    for output_array in outputs:
        for output in output_array[0]:
            x1, y1, x2, y2, score, class_id = output[:6]
            if score > conf: 
                x1, y1, x2, y2 = x1 * w / 640, y1 * h / 640, x2 * w / 640, y2 * h / 640
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                if class_id < len(class_names):
                    class_name = class_names[int(class_id)]["name"]
                label = f"{class_name}: {score:.2f}"

                font_scale = 1.0  
                thickness = 3     
                
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)
        return frame

def detect_video(conf, uploaded_file, model):
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_input_file:
            temp_input_file.write(uploaded_file.read())
            temp_input_file_path = temp_input_file.name
        detect_from_file(conf=conf, video_file=temp_input_file_path, model=model)

def detect_from_file(conf, video_file, model):
    if video_file:
        cap = cv2.VideoCapture(video_file)

    current_time = datetime.datetime.now()
    timestamp = current_time.strftime("%d-%m-%Y")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 25.0

    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4', dir=tempfile.gettempdir()) as mp4_file:
        mp4_filename = mp4_file.name
        out = cv2.VideoWriter(mp4_filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    st_frame = st.empty()

    col1, col2, col3 = st.columns(3, gap="large")
    with col1:
        rewind_button = st.button("⏪ Lùi 10 giây", width="stretch")
    with col2:
        stop_button = st.button("⏹ Dừng", width="stretch")
        stop_pressed = False
    with col3:
        fast_forward_button = st.button("⏩ Tới 10 giây", width="stretch")

    frame_count = 0
    start_time = time.time()

    stop_pressed = False
    skip_frames = 0

    totals_placeholder = st.empty()
    results_placeholder = st.empty()
    st.markdown('<div class="section-title">🍽️ Kết quả nhận diện</div>'
                '<div class="section-sub">Các món ăn được phát hiện trong video '
                '(cập nhật trực tiếp).</div>', unsafe_allow_html=True)

    items = {}
    nutrition_data = []
    total_nutrition = _new_total_nutrition()

    while True:
        success, image = cap.read()

        if skip_frames > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES) + skip_frames)
            skip_frames = 0
        if rewind_button:
            skip_frames = -int(fps * 10)
        if fast_forward_button:
            skip_frames = int(fps * 10)
        if stop_button:
            stop_pressed = True

        if not success or stop_pressed:
            break

        results = model.predict(source=image, conf=conf, imgsz=640, save=False, device="cpu", half=False)

        for r in results:
            im_bgr = r.plot()
            frame_count += 1
            elapsed_time = time.time() - start_time
            if elapsed_time >= 1.0:
                fps = frame_count / elapsed_time
                start_time = time.time()
                frame_count = 0
            cv2.putText(im_bgr, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            out.write(im_bgr)

            im_rgb = Image.fromarray(im_bgr[..., ::-1])
            st_frame.image(im_rgb, caption='Video dự đoán', width="stretch")

            for pred in r.boxes:
                class_id = int(pred.cls[0].item())
                class_name = class_names[int(class_id)]["name"]
                confident = int(round(pred.conf[0].item(), 2)*100)
                serving = class_names[int(class_id)]["serving_type"]

                if isinstance(pred.xyxy, torch.Tensor):
                    boxes = pred.xyxy.cpu().numpy()
                else:
                    boxes = pred.xyxy.numpy()

                image_np = r.orig_img

                bounding_box_images = extract_bounding_box_image(image_np, boxes)

                bbox_image_html = ""
                if bounding_box_images:
                    bbox_image = bounding_box_images[0]
                    bbox_image_pil = Image.fromarray(cv2.cvtColor(bbox_image, cv2.COLOR_BGR2RGB))

                    buffered = io.BytesIO()
                    bbox_image_pil.save(buffered, format="JPEG")
                    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                    bbox_image_html = ui.thumb_img(img_str)

                if class_name not in items:
                    card_html, nutrition = _dish_card_for_detection(
                        class_id, class_name, confident, serving, bbox_image_html)
                    if card_html is not None:
                        items[class_name] = card_html
                        if nutrition:
                            for key in total_nutrition:
                                if key in nutrition:
                                    total_nutrition[key] += nutrition[key]
                            nutrition_data.append({
                                "name": class_name,
                                "serving": ui.vi_serving(serving),
                                "conf": confident,
                                "nutrition": {k: nutrition.get(k) for k in _NUTRIENT_ORDER},
                            })

        totals_placeholder.markdown(_totals_kpi_row(total_nutrition), unsafe_allow_html=True)
        if items:
            cards = "".join(items.values())
            results_placeholder.markdown(
                f'<div class="result-panel">{cards}</div>', unsafe_allow_html=True)

        if stop_button:
            stop_pressed = True
            stop_button = None
            break

    cap.release()
    out.release()
    st.session_state.last_detected_dishes = {name: 1 for name in items}
    st.session_state.last_total_nutrition = total_nutrition

    with tempfile.NamedTemporaryFile(delete=False, suffix='.csv', dir=tempfile.gettempdir()) as csv_file:
        csv_filename = csv_file.name
    with open(csv_filename, mode='w', newline='', encoding='utf-8-sig') as file:
        writer = csv.writer(file)
        writer.writerow(["Tên món", "Khẩu phần", "Tin cậy (%)", "Calo (kcal)", "Protein (g)",
                         "Carb (g)", "Chất béo (g)", "Bão hòa (g)", "Đường (g)", "Muối (g)"])
        for d in nutrition_data:
            n = d["nutrition"]
            writer.writerow([d["name"], d["serving"], d["conf"],
                             n["Calories"], n["Protein"], n["Carbs"], n["Fat"],
                             n["Saturates"], n["Sugar"], n["Salt"]])
    with open(csv_filename, "rb") as file:
        the_csv = file.read()

    # Đọc video đã xử lý SAU khi ghi xong (trước đây bị đọc trước → file tải về rỗng)
    with open(mp4_filename, "rb") as file:
        the_mp4 = file.read()

    col1, col2 = st.columns(2, gap="large")
    with col1:
        download_video = st.download_button(label="🎬 Video đã xử lý (MP4)",
                                data=the_mp4,
                                mime="video/mp4",
                                file_name=f"{timestamp}.mp4",
                                width="stretch",
                                key=f"download_video_button_{timestamp}")
        if download_video:
            os.remove(mp4_filename)
    with col2:
        download_csv = st.download_button(label="📊 Kết quả dự đoán (CSV)",
                            data=the_csv,
                            file_name=f"{timestamp}.csv",
                            width="stretch",
                            key=f"download_csv_video_button_{timestamp}")
        if download_csv:
            os.remove(csv_filename)
