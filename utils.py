import av
from ultralytics import YOLO
import streamlit as st
import cv2
from PIL import Image
import tempfile
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer, VideoTransformerBase

import numpy as np
from io import BytesIO
import queue

import time
from collections import deque

import csv
import re
import requests
import datetime
import os
import io
import base64
import plotly.graph_objects as go

from class_names import class_names

def styling_css():
    with open('./assets/css/general-style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
        
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
        template="plotly_white",
        margin=dict(l=0, r=0, b=0, t=0),
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=True),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=True),
        annotations=[
            dict(
                x=0.5,
                y=-0.1,
                showarrow=False,
                text="Detected Image" if detected else "Original Image",
                xref="paper",
                yref="paper"
            )
        ]
    )
    
    return fig

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
            st.toast('Connecting', icon="🕒")
            
            try:
                results = model(source=valid_url, stream=True, conf=conf, imgsz=640, save=True, device="cpu", vid_stride=1, half=False)
                displayed_dishes = set()
                total_nutrition = {
                    "Calories": 0,
                    "Fat": 0,
                    "Saturates": 0,
                    "Sugar": 0,
                    "Salt": 0,
                    "Protein": 0
                }
                detection_results = ""
                new_detections = False
                nutrition_data = []
                current_time = datetime.datetime.now()
                time_format = current_time.strftime("%d-%m-%Y")
                
                stop_button = st.button("Stop")
                stop_pressed = False

                st_frame = st.empty()

                frame_count = 0
                start_time = time.time()

                total_nutrition_placeholder = st.empty()
                st.markdown("""<br>
                        <h5 class="detection-results">Detection Results</h5><p class="small-text-below-results">We found the following foods in your meal</p>""", unsafe_allow_html=True)
                # nutrition_placeholder = st.empty()

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
                    st_frame.image(im_rgb_resized, caption='Predicted Video', use_column_width=True)      
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
                            bbox_image_html = f'<img src="data:image/jpeg;base64,{img_str}" class="img-each-nutri" ">'


                        if class_name == "Con nguoi (Human)" and class_name not in displayed_dishes:
                            detection_results += f"<p class='human-class-name'><b>Class name:</b> {class_name}</p><p class='human-confident'><b>Confidence:</b> {confident}%</p><hr style='border: none; border-top: 1px dashed black; width: 80%;'>"

                            displayed_dishes.add(class_name)
                            new_detections = True
                        elif class_name not in displayed_dishes:
                            nutrition = class_names[int(class_id)]["nutrition"]
                            if nutrition:
                                displayed_dishes.add(class_name)
                                new_detections = True

                                calories_desc = get_nutri_score_color("Calories", nutrition.get('Calories'), serving)
                                fat_color, fat_desc = get_nutri_score_color("Fat", nutrition.get('Fat'), serving)
                                saturates_color, saturates_desc = get_nutri_score_color("Saturates", nutrition.get('Saturates'), serving)
                                sugar_color, sugar_desc = get_nutri_score_color("Sugar", nutrition.get('Sugar'), serving)
                                salt_color, salt_desc = get_nutri_score_color("Salt", nutrition.get('Salt'), serving)

                                percentage_contribution = calculate_nutrient_percentage(nutrition)

                                nutrition_str = f"""
<div class="each-nutri-container">
    <div class="each-nutri-box" style="background-color: "transparent";">
        {bbox_image_html}
    </div>
    <div  id="calo-each-nutri-box" class="each-nutri-box" style="background-color: transparent;">
        <span class="each-nutri-name">Calories</span><br>
        <p class="each-nutri-number">{nutrition.get('Calories')} kcal</p>
        <span id="calo-each-nutri-percentage" class="each-nutri-percentage">{percentage_contribution['Calories']:.1f}%</span>
    </div>
    <div id="protein-each-nutri-box" class="each-nutri-box">
        <span class="each-nutri-name">Protein</span><br>
        <p class="each-nutri-number">{nutrition.get('Protein')} gram</p>
        <span id="protein-each-nutri-percentage" class="each-nutri-percentage">{percentage_contribution.get('Protein', 0):.1f}%</span>
    </div>
    <div class="each-nutri-box" style="background-color: {fat_color};">
        <span class="each-nutri-name">Fat</span><br>
        <p class="each-nutri-number">{nutrition.get('Fat')} gram</p>
        <span class="each-nutri-percentage">{percentage_contribution['Fat']:.1f}%</span>
    </div>
    <div class="each-nutri-box" style="background-color: {saturates_color};">
        <span class="each-nutri-name">Saturates</span><br>
        <p class="each-nutri-number">{nutrition.get('Saturates')} gram</p>
        <span class="each-nutri-percentage">{percentage_contribution['Saturates']:.1f}%</span>
    </div>
    <div class="each-nutri-box" style="background-color: {sugar_color};">
        <span class="each-nutri-name">Sugar</span><br>
        <p class="each-nutri-number">{nutrition.get('Sugar')} gram</p>
        <span class="each-nutri-percentage">{percentage_contribution['Sugar']:.1f}%</span>
    </div>
    <div class="each-nutri-box" style="background-color: {salt_color};">
        <span class="each-nutri-name">Salt</span><br>
        <p class="each-nutri-number">{nutrition.get('Salt')} gram</p>
        <span class="each-nutri-percentage">{percentage_contribution['Salt']:.1f}%</span>
    </div>
</div>
                            """

                                detection_results += (
                    f"""<p class="item-header">{confident}%: <b>{class_name}</b></p>
                    <p class="nutrition-header">Nutrition ({serving})</p>
                    <p class="nutrition-facts">{nutrition_str}</p>
                    <hr style="border: none; border-top: 1px dashed black; width: 80%;">
                    """)

                                for key in total_nutrition:
                                    if key in nutrition:
                                        total_nutrition[key] += nutrition[key]

                            nutrition_data.append((
                                class_name,
                                serving,
                                confident,
                                nutrition.get('Calories'),
                                nutrition.get('Protein'),
                                nutrition.get('Fat'),
                                nutrition.get('Saturates'),
                                nutrition.get('Sugar'),
                                nutrition.get('Salt')
                            ))
                    

                    if stop_button:
                        stop_pressed = True
                        stop_button = None
                        break                       
                    
                if new_detections:
                    scrollable_textbox = f"""<div class="result-nutri-container">{detection_results}</div>"""
                    
                    st.markdown(scrollable_textbox, unsafe_allow_html=True)
                    

                total_nutrition_str = f"""
    <h5 class="total-nutrition-title">Total Nutrition Values</h5>
    <div class="total-nutrition-container">
        <div class="total-nutri-box" id="calo-box">
            <span class="total-nutri-name">Calories</span><br>
            <span class="total-nutri-num">{total_nutrition['Calories']:.1f} kcal</span>
        </div>
        <div class="total-nutri-box" id="protein-box">
            <span class="total-nutri-name">Protein</span><br>
            <span class="total-nutri-num">{total_nutrition.get('Protein', 0.0):.1f} gram</span>
        </div>
        <div class="total-nutri-box">
            <span class="total-nutri-name">Fat</span><br>
            <span class="total-nutri-num">{total_nutrition['Fat']:.1f} gram</span>
        </div>
        <div class="total-nutri-box">
            <span class="total-nutri-name">Saturates</span><br>
            <span class="total-nutri-num">{total_nutrition['Saturates']:.1f} gram</span>
        </div>
        <div class="total-nutri-box">
            <span class="total-nutri-name">Sugar</span><br>
            <span class="total-nutri-num">{total_nutrition['Sugar']:.1f} gram</span>
        </div>
        <div class="total-nutri-box">
            <span class="total-nutri-name">Salt</span><br>
            <span class="total-nutri-num">{total_nutrition['Salt']:.1f} gram</span>
        </div>
    </div>
"""

                total_nutrition_placeholder.markdown(total_nutrition_str, unsafe_allow_html=True)

                st.session_state.last_detected_dishes = {dish: 1 for dish in displayed_dishes}
                st.session_state.last_total_nutrition = total_nutrition

                displayed_dishes.clear()


                # with tempfile.NamedTemporaryFile(delete=False, suffix=".csv", dir="/tmp") as csv_file:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.csv', dir=tempfile.gettempdir()) as csv_file:
                    csv_filename = csv_file.name
                with open(csv_filename, mode='w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(["Food Name", "Serving", "Confidence (%)", "Calories (kcal)", "Protein (g)", "Fat (g)", "Saturates (g)", "Sugar (g)", "Salt (g)"])
                    writer.writerows(nutrition_data)
                with open(csv_filename, "rb") as file:
                    the_csv = file.read()  
                
                st.toast("Prediction completed. Results saved to CSV.", icon="✅")
                time.sleep(3000)
                download_csv = st.download_button(label="Download Predictions CSV",
                                data=the_csv,
                                file_name=f"{time_format}.csv", 
                                use_container_width=True,
                                key=f"download_csv3_button_{time_format}")
                if download_csv:
                    os.remove(csv_filename)
            except ConnectionError as e:
                st.error(f"Failed to open YouTube video stream: {e}")
        else:
            st.error("Invalid YouTube URL or unable to extract YouTube ID.")
    else:
        st.error("YouTube URL is required.")

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

def get_nutri_score_color(nutrient, value, serving_type):
    if serving_type == "per 100g":
        if nutrient == "Calories":
            if value < 100:
                return COLOR_LOW, "Low"
            elif value < 200:
                return COLOR_MEDIUM, "Medium"
            else:
                return COLOR_HIGH, "High"
        elif nutrient == "Fat":
            if value < 3:
                return COLOR_LOW, "Low"
            elif value < 17.5:
                return COLOR_MEDIUM, "Medium"
            else:
                return COLOR_HIGH, "High"
        elif nutrient == "Saturates":
            if value < 1.5:
                return COLOR_LOW, "Low"
            elif value < 5:
                return COLOR_MEDIUM, "Medium"
            else:
                return COLOR_HIGH, "High"
        elif nutrient == "Sugar":
            if value < 5:
                return COLOR_LOW, "Low"
            elif value < 22.5:
                return COLOR_MEDIUM, "Medium"
            else:
                return COLOR_HIGH, "High"
        elif nutrient == "Salt":
            if value < 0.3:
                return COLOR_LOW, "Low"
            elif value < 1.5:
                return COLOR_MEDIUM, "Medium"
            else:
                return COLOR_HIGH, "High"
   
    elif serving_type == "1 serving":
        if nutrient == "Calories":
            if value < 150:
                return COLOR_LOW, "Low"
            elif value < 300:
                return COLOR_MEDIUM, "Medium"
            else:
                return COLOR_HIGH, "High"
        elif nutrient == "Fat":
            if value < 5:
                return COLOR_LOW, "Low"
            elif value < 21:
                return COLOR_MEDIUM, "Medium"
            else:
                return COLOR_HIGH, "High"
        elif nutrient == "Saturates":
            if value < 2:
                return COLOR_LOW, "Low"
            elif value < 6:
                return COLOR_MEDIUM, "Medium"
            else:
                return COLOR_HIGH, "High"
        elif nutrient == "Sugar":
            if value < 6:
                return COLOR_LOW, "Low"
            elif value < 27:
                return COLOR_MEDIUM, "Medium"
            else:
                return COLOR_HIGH, "High"
        elif nutrient == "Salt":
            if value < 0.4:
                return COLOR_LOW, "Low"
            elif value < 1.8:
                return COLOR_MEDIUM, "Medium"
            else:
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
        extracted_images.append(bbox_image)

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

def detect_image_result(detected_image, model):
    boxes = detected_image[0].boxes

    if boxes:
        detected_img_arr_RGB = detected_image[0].plot()[:, :, ::1]
        detected_img_arr_BGR = detected_image[0].plot()[:, :, ::-1]
        fig_detected = create_fig(detected_img_arr_BGR, detected=True)
        st.plotly_chart(fig_detected, use_container_width=True)

        current_time = datetime.datetime.now()
        time_format = current_time.strftime("%d-%m-%Y")

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
            total_nutrition = {
                "Calories": 0,
                "Fat": 0,
                "Saturates": 0,
                "Sugar": 0,
                "Salt": 0,
                "Protein": 0
            }

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
                        bbox_image_html = f'<img src="data:image/jpeg;base64,{img_str}" class="img-each-nutri" ">'

                    

                    if class_id in count_dict:
                        count_dict[class_id] += 1
                    else:
                        count_dict[class_id] = 1

                    if class_name == "Con nguoi (Human)":
                        detection_results += f"<p class='human-class-name'><b>Class name:</b> {class_name}</p><p class='human-confident'><b>Confidence:</b> {conf}%</p><hr style='border: none; border-top: 1px dashed black; width: 80%;'>"

                    else:
                        nutrition = class_names[int(class_id)]["nutrition"]
                        if nutrition:
                            calories_desc = get_nutri_score_color("Calories", nutrition.get('Calories'), serving)
                            fat_color, fat_desc = get_nutri_score_color("Fat", nutrition.get('Fat'), serving)
                            saturates_color, saturates_desc = get_nutri_score_color("Saturates", nutrition.get('Saturates'), serving)
                            sugar_color, sugar_desc = get_nutri_score_color("Sugar", nutrition.get('Sugar'), serving)
                            salt_color, salt_desc = get_nutri_score_color("Salt", nutrition.get('Salt'), serving)

                            percentage_contribution = calculate_nutrient_percentage(nutrition)


                            nutrition_str = f"""
    <div class="each-nutri-container">
        <div class="each-nutri-box" style="background-color: "transparent";">
            {bbox_image_html}
        </div>
        <div  id="calo-each-nutri-box" class="each-nutri-box" style="background-color: transparent;">
            <span class="each-nutri-name">Calories</span><br>
            <p class="each-nutri-number">{nutrition.get('Calories')} kcal</p>
            <span id="calo-each-nutri-percentage" class="each-nutri-percentage">{percentage_contribution['Calories']:.1f}%</span>
        </div>
        <div id="protein-each-nutri-box" class="each-nutri-box">
            <span class="each-nutri-name">Protein</span><br>
            <p class="each-nutri-number">{nutrition.get('Protein')} gram</p>
            <span id="protein-each-nutri-percentage" class="each-nutri-percentage">{percentage_contribution.get('Protein', 0):.1f}%</span>
        </div>
        <div class="each-nutri-box" style="background-color: {fat_color};">
            <span class="each-nutri-name">Fat</span><br>
            <p class="each-nutri-number">{nutrition.get('Fat')} gram</p>
            <span class="each-nutri-percentage">{percentage_contribution['Fat']:.1f}%</span>
        </div>
        <div class="each-nutri-box" style="background-color: {saturates_color};">
            <span class="each-nutri-name">Saturates</span><br>
            <p class="each-nutri-number">{nutrition.get('Saturates')} gram</p>
            <span class="each-nutri-percentage">{percentage_contribution['Saturates']:.1f}%</span>
        </div>
        <div class="each-nutri-box" style="background-color: {sugar_color};">
            <span class="each-nutri-name">Sugar</span><br>
            <p class="each-nutri-number">{nutrition.get('Sugar')} gram</p>
            <span class="each-nutri-percentage">{percentage_contribution['Sugar']:.1f}%</span>
        </div>
        <div class="each-nutri-box" style="background-color: {salt_color};">
            <span class="each-nutri-name">Salt</span><br>
            <p class="each-nutri-number">{nutrition.get('Salt')} gram</p>
            <span class="each-nutri-percentage">{percentage_contribution['Salt']:.1f}%</span>
        </div>
    </div>
                                """


                        detection_results += (
                        f"""<p class="item-header">{count_dict[class_id]} ({conf}%): <b>{class_name}</b></p>
                        <p class="nutrition-header">Nutrition ({serving}) </p>
                        <p class="nutrition-facts">{nutrition_str}</p>
                        <hr style="border: none; border-top: 1px dashed black; width: 80%;">
                        """)

                        for key in total_nutrition:
                            if key in nutrition:
                                total_nutrition[key] += nutrition[key]

                        nutrition_data.append((
                            class_name,
                            serving,
                            conf,
                            nutrition.get('Calories'),
                            nutrition.get('Protein'),
                            nutrition.get('Fat'),
                            nutrition.get('Saturates'),
                            nutrition.get('Sugar'),
                            nutrition.get('Salt')
                        ))


            total_nutrition_str = f"""
    <h5 class="total-nutrition-title">Total Nutrition Values</h5>
    <div class="total-nutrition-container">
        <div class="total-nutri-box" id="calo-box">
            <span class="total-nutri-num">{total_nutrition['Calories']:.1f} kcal</span><br>
            <span class="total-nutri-value-name">Calories</span>
        </div>
        <div class="total-nutri-box" id="protein-box">
            <span class="total-nutri-num">{total_nutrition.get('Protein', 0.0):.1f} gram</span><br>
            <span class="total-nutri-value-name">Protein</span>
        </div>
        <div class="total-nutri-box">
            <span class="total-nutri-num">{total_nutrition['Fat']:.1f} gram</span><br>
            <span class="total-nutri-value-name">Fat</span>
        </div>
        <div class="total-nutri-box">
            <span class="total-nutri-num">{total_nutrition['Saturates']:.1f} gram</span><br>
            <span class="total-nutri-value-name">Saturates</span>
        </div>
        <div class="total-nutri-box">
            <span class="total-nutri-num">{total_nutrition['Sugar']:.1f} gram</span><br>
            <span class="total-nutri-value-name">Sugar</span>
        </div>
        <div class="total-nutri-box">
            <span class="total-nutri-num">{total_nutrition['Salt']:.1f} gram</span><br>
            <span class="total-nutri-value-name">Salt</span>
        </div>
    </div>
"""


            # detection_results += total_nutrition_str
            total_nutrition_placeholder.markdown(total_nutrition_str, unsafe_allow_html=True)

            for object_type, count in count_dict.items():
                the_name = class_names[object_type]["name"]
                counts.append(count)
                # detection_results += f"<b style='color: black;'>Count of {the_name}:</b> {count}<br>"
                count_results += f"""
                <p class="total-count-result-text">{the_name}: {count}<hr class="dash-line-below-count-results"></p>"""
                
            scrollable_textbox = f"""<div class="result-nutri-container">{detection_results}</div>"""
            
            st.markdown("""<br>
                        <h5 class="detection-results">Detection Results</h5><p class="small-text-below-results">We found the following foods in your meal</p>""", unsafe_allow_html=True)
            
            st.markdown(f'<div class="total-count-result-div">{count_results}</div>', unsafe_allow_html=True)
            st.markdown(scrollable_textbox, unsafe_allow_html=True)

            # rows = zip(food_names, confidences, counts)
            # rows = zip(nutrition_data)

            # with tempfile.NamedTemporaryFile(delete=False, suffix='.csv', dir='/tmp') as csv_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.csv', dir=tempfile.gettempdir()) as csv_file:
                csv_filename = csv_file.name
            with open(csv_filename, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Food Name", "Serving", "Confidence (%)", "Calories (kcal)", "Protein (g)", "Fat (g)", "Saturates (g)", "Sugar (g)", "Salt (g)"])
                writer.writerows(nutrition_data)
            with open(csv_filename, 'rb') as file:
                the_csv = file.read()
        col1, col2 = st.columns(2, gap="large")
        with col1:    
            download_pic = st.download_button(label="Download Predicted Image",
                                    data=the_img,
                                    mime="image/jpg",
                                    file_name=f"{time_format}.jpg", 
                                    use_container_width=True,
                                    key=f"download_pic_button_{time_format}")
            if download_pic:
                os.remove(img_filename)
        with col2: 
            download_csv = st.download_button(label="Download Predictions CSV", 
                               data=the_csv, 
                               file_name=f"{time_format}.csv", 
                               use_container_width=True,
                               key=f"download_csv_button_{time_format}")
            if download_csv:
                os.remove(csv_filename)
        st.divider()
        
        # AI Nutrition Advice and Evaluation
        count_dict_names = {}
        for object_type, count in count_dict.items():
            the_name = class_names[object_type]["name"]
            count_dict_names[the_name] = count
            
        st.session_state.last_detected_dishes = count_dict_names
        st.session_state.last_total_nutrition = total_nutrition
        
        generate_nutrition_advice(count_dict_names, total_nutrition)

    else:
        st.markdown("""<h5 class="total-count-result" id="no-food-detected">No food detected</h5>
                    <p class="result-nutri-container" id="no-food-descr">The model did not detect any foods in the uploaded image.  
            Please try with a different image or adjust the model's 
            confidence threshold and try again.</p>
                    """, unsafe_allow_html=True)  


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

        resized_uploaded_image = resize_image(uploaded_image)

        if st.session_state.show_image and not st.session_state.is_reset and not st.session_state.button_clicked:   
            original_image = st.image(resized_uploaded_image, output_format="JPEG", use_column_width=True)

        if not st.session_state.is_reset:
            col1, col2 = st.columns([0.8, 0.2], gap="large")
            with col1:
                if st.session_state.show_image and not st.session_state.button_clicked and not original_image == st.empty():
                    st.markdown("**Original Image**")
                elif not st.session_state.show_image and st.session_state.button_clicked:
                    st.markdown("**Predicted Image**")
            with col2:
                if not st.session_state.button_clicked:
                    predict_button = st.button("Predict", use_container_width=True, type="primary", on_click=toggle_button)
                else:
                    reset_button = st.button("Reset", use_container_width=True, type="primary", on_click=toggle_button, args=[True])
                    uploaded_file = None
        if st.session_state.show_image and st.session_state.is_reset and not st.session_state.button_clicked:
            st.session_state.is_reset = False

        if st.session_state.button_clicked and not reset_button:
            with st.spinner("Running..."):
                detected_image = model.predict(resized_uploaded_image, conf=conf, imgsz=640, half=False)
                detect_image_result(detected_image, model)        

def detect_camera(conf, model, address):
    vid_cap = cv2.VideoCapture('rtsp://admin:' + address)
    vid_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  
    vid_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)  
    vid_cap.set(cv2.CAP_PROP_FPS, 15)
    fps = vid_cap.get(cv2.CAP_PROP_FPS)

    
    while True:
        if vid_cap.isOpened():
            st.toast("Connected", icon="✅")
            break
        else:
            vid_cap.release()
            return
    try: 
        st_frame = st.empty()

        displayed_dishes = set()

        total_nutrition = {
            "Calories": 0,
            "Fat": 0,
            "Saturates": 0,
            "Sugar": 0,
            "Salt": 0,
            "Protein": 0
        }

        total_nutrition_placeholder = st.empty()
        
        st.markdown("""<br>
                        <h5 class="detection-results">Detection Results</h5><p class="small-text-below-results">We found the following foods in your meal</p>""", unsafe_allow_html=True)

        frame_count = 0
        start_time = time.time()
        while True:   
            success, image = vid_cap.read()
            if success:
                # mirrored_frame = cv2.flip(image, 1)
                results = model.track(source=image, conf=conf, imgsz=640, save=False, device="cpu", stream=True, half=False)


                new_detections = False  
                detection_results = ""

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
                    st_frame.image(im_rgb, caption='Camera IP', use_column_width=True)

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
                            bbox_image_html = f'<img src="data:image/jpeg;base64,{img_str}" class="img-each-nutri" ">'


                        if class_name == "Con nguoi (Human)" and class_name not in displayed_dishes:
                            detection_results += f"<p class='human-class-name'><b>Class name:</b> {class_name}</p><p class='human-confident'><b>Confidence:</b> {confident}%</p><hr style='border: none; border-top: 1px dashed black; width: 80%;'>"
                            displayed_dishes.add(class_name)
                            new_detections = True
                        elif class_name not in displayed_dishes:
                            nutrition = class_names[int(class_id)]["nutrition"]
                            if nutrition:
                                displayed_dishes.add(class_name)
                                new_detections = True

                                calories_desc = get_nutri_score_color("Calories", nutrition.get('Calories'), serving)
                                fat_color, fat_desc = get_nutri_score_color("Fat", nutrition.get('Fat'), serving)
                                saturates_color, saturates_desc = get_nutri_score_color("Saturates", nutrition.get('Saturates'), serving)
                                sugar_color, sugar_desc = get_nutri_score_color("Sugar", nutrition.get('Sugar'), serving)
                                salt_color, salt_desc = get_nutri_score_color("Salt", nutrition.get('Salt'), serving)

                                percentage_contribution = calculate_nutrient_percentage(nutrition)

                                nutrition_str = f"""
<div class="each-nutri-container">
    <div class="each-nutri-box" style="background-color: "transparent";">
        {bbox_image_html}
    </div>
    <div  id="calo-each-nutri-box" class="each-nutri-box" style="background-color: transparent;">
        <span class="each-nutri-name">Calories</span><br>
        <p class="each-nutri-number">{nutrition.get('Calories')} kcal</p>
        <span id="calo-each-nutri-percentage" class="each-nutri-percentage">{percentage_contribution['Calories']:.1f}%</span>
    </div>
    <div id="protein-each-nutri-box" class="each-nutri-box">
        <span class="each-nutri-name">Protein</span><br>
        <p class="each-nutri-number">{nutrition.get('Protein')} gram</p>
        <span id="protein-each-nutri-percentage" class="each-nutri-percentage">{percentage_contribution.get('Protein', 0):.1f}%</span>
    </div>
    <div class="each-nutri-box" style="background-color: {fat_color};">
        <span class="each-nutri-name">Fat</span><br>
        <p class="each-nutri-number">{nutrition.get('Fat')} gram</p>
        <span class="each-nutri-percentage">{percentage_contribution['Fat']:.1f}%</span>
    </div>
    <div class="each-nutri-box" style="background-color: {saturates_color};">
        <span class="each-nutri-name">Saturates</span><br>
        <p class="each-nutri-number">{nutrition.get('Saturates')} gram</p>
        <span class="each-nutri-percentage">{percentage_contribution['Saturates']:.1f}%</span>
    </div>
    <div class="each-nutri-box" style="background-color: {sugar_color};">
        <span class="each-nutri-name">Sugar</span><br>
        <p class="each-nutri-number">{nutrition.get('Sugar')} gram</p>
        <span class="each-nutri-percentage">{percentage_contribution['Sugar']:.1f}%</span>
    </div>
    <div class="each-nutri-box" style="background-color: {salt_color};">
        <span class="each-nutri-name">Salt</span><br>
        <p class="each-nutri-number">{nutrition.get('Salt')} gram</p>
        <span class="each-nutri-percentage">{percentage_contribution['Salt']:.1f}%</span>
    </div>
</div>
                            """

                                detection_results += (
                    f"""<p class="item-header">{confident}%: <b>{class_name}</b></p>
                    <p class="nutrition-header">Nutrition ({serving})</p>
                    <p class="nutrition-facts">{nutrition_str}</p>
                    <hr style="border: none; border-top: 1px dashed black; width: 80%;">
                    """)
                                
                                

                                for key in total_nutrition:
                                    if key in nutrition:
                                        total_nutrition[key] += nutrition[key]
               
               
                if new_detections:
                    scrollable_textbox = f"""<div class="result-nutri-container">{detection_results}</div>"""
                    
                    st.markdown(scrollable_textbox, unsafe_allow_html=True)
                
                
                total_nutrition_str = f"""
    <h5 class="total-nutrition-title">Total Nutrition Values</h5>
    <div class="total-nutrition-container">
        <div class="total-nutri-box" id="calo-box">
            <span class="total-nutri-name">Calories</span><br>
            <span class="total-nutri-num">{total_nutrition['Calories']:.1f} kcal</span>
        </div>
        <div class="total-nutri-box" id="protein-box">
            <span class="total-nutri-name">Protein</span><br>
            <span class="total-nutri-num">{total_nutrition.get('Protein', 0.0):.1f} gram</span>
        </div>
        <div class="total-nutri-box">
            <span class="total-nutri-name">Fat</span><br>
            <span class="total-nutri-num">{total_nutrition['Fat']:.1f} gram</span>
        </div>
        <div class="total-nutri-box">
            <span class="total-nutri-name">Saturates</span><br>
            <span class="total-nutri-num">{total_nutrition['Saturates']:.1f} gram</span>
        </div>
        <div class="total-nutri-box">
            <span class="total-nutri-name">Sugar</span><br>
            <span class="total-nutri-num">{total_nutrition['Sugar']:.1f} gram</span>
        </div>
        <div class="total-nutri-box">
            <span class="total-nutri-name">Salt</span><br>
            <span class="total-nutri-num">{total_nutrition['Salt']:.1f} gram</span>
        </div>
    </div>
"""

                
                total_nutrition_placeholder.markdown(total_nutrition_str, unsafe_allow_html=True)
                st.session_state.last_detected_dishes = {dish: 1 for dish in displayed_dishes}
                st.session_state.last_total_nutrition = total_nutrition
            else:
                break
    except Exception as e:
        st.error(f"Error loading video: {str(e)}")
    finally:
        vid_cap.release()
        displayed_dishes.clear()

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
                    bbox_image_html = f'<img src="data:image/jpeg;base64,{img_str}" class="img-each-nutri" ">'
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
        total_nutrition_placeholder = st.empty()
        st.markdown("""<br>
                        <h5 class="detection-results">Detection Results</h5><p class="small-text-below-results">We found the following foods in your meal</p>""", unsafe_allow_html=True)
        results_placeholder = st.empty()
        
        while True:
            total_nutrition = {
                "Calories": 0,
                "Fat": 0,
                "Saturates": 0,
                "Sugar": 0,
                "Salt": 0,
                "Protein": 0
            }
            detections = result_queue.get()

            detection_results = ""
            new_detections = False
            displayed_dishes = set()

            for detection in detections:
                class_id = detection.class_id
                class_name = detection.class_name
                confident = detection.confident
                serving = detection.serving
                bbox_image_html = detection.bbox_image_html
                
                if class_name == "Con nguoi (Human)" and class_name not in displayed_dishes:
                    detection_results += f"<p class='human-class-name'><b>Class name:</b> {class_name}</p><p class='human-confident'><b>Confidence:</b> {confident}%</p><hr style='border: none; border-top: 1px dashed black; width: 80%;'>"

                    displayed_dishes.add(class_name)
                    new_detections = True
                elif class_name not in displayed_dishes:
                    nutrition = class_names[int(class_id)]["nutrition"]
                    if nutrition:
                        displayed_dishes.add(class_name)
                        new_detections = True

                        calories_desc = get_nutri_score_color("Calories", nutrition.get('Calories'), serving)
                        fat_color, fat_desc = get_nutri_score_color("Fat", nutrition.get('Fat'), serving)
                        saturates_color, saturates_desc = get_nutri_score_color("Saturates", nutrition.get('Saturates'), serving)
                        sugar_color, sugar_desc = get_nutri_score_color("Sugar", nutrition.get('Sugar'), serving)
                        salt_color, salt_desc = get_nutri_score_color("Salt", nutrition.get('Salt'), serving)

                        percentage_contribution = calculate_nutrient_percentage(nutrition)

                        nutrition_str = f"""
<div class="each-nutri-container">
    <div class="each-nutri-box" style="background-color: "transparent";">
        {bbox_image_html}
    </div>
    <div  id="calo-each-nutri-box" class="each-nutri-box" style="background-color: transparent;">
        <span class="each-nutri-name">Calories</span><br>
        <p class="each-nutri-number">{nutrition.get('Calories')} kcal</p>
        <span id="calo-each-nutri-percentage" class="each-nutri-percentage">{percentage_contribution['Calories']:.1f}%</span>
    </div>
    <div id="protein-each-nutri-box" class="each-nutri-box">
        <span class="each-nutri-name">Protein</span><br>
        <p class="each-nutri-number">{nutrition.get('Protein')} gram</p>
        <span id="protein-each-nutri-percentage" class="each-nutri-percentage">{percentage_contribution.get('Protein', 0):.1f}%</span>
    </div>
    <div class="each-nutri-box" style="background-color: {fat_color};">
        <span class="each-nutri-name">Fat</span><br>
        <p class="each-nutri-number">{nutrition.get('Fat')} gram</p>
        <span class="each-nutri-percentage">{percentage_contribution['Fat']:.1f}%</span>
    </div>
    <div class="each-nutri-box" style="background-color: {saturates_color};">
        <span class="each-nutri-name">Saturates</span><br>
        <p class="each-nutri-number">{nutrition.get('Saturates')} gram</p>
        <span class="each-nutri-percentage">{percentage_contribution['Saturates']:.1f}%</span>
    </div>
    <div class="each-nutri-box" style="background-color: {sugar_color};">
        <span class="each-nutri-name">Sugar</span><br>
        <p class="each-nutri-number">{nutrition.get('Sugar')} gram</p>
        <span class="each-nutri-percentage">{percentage_contribution['Sugar']:.1f}%</span>
    </div>
    <div class="each-nutri-box" style="background-color: {salt_color};">
        <span class="each-nutri-name">Salt</span><br>
        <p class="each-nutri-number">{nutrition.get('Salt')} gram</p>
        <span class="each-nutri-percentage">{percentage_contribution['Salt']:.1f}%</span>
    </div>
</div>
                            """

                        detection_results += (
                    f"""<p class="item-header">{confident}%: <b>{class_name}</b></p>
                    <p class="nutrition-header">Nutrition ({serving})</p>
                    <p class="nutrition-facts">{nutrition_str}</p>
                    <hr style="border: none; border-top: 1px dashed black; width: 80%;">
                    """)
                        for key in total_nutrition:
                                    if key in nutrition:
                                        total_nutrition[key] += nutrition[key]

            results_placeholder.markdown(f"""<div class="result-nutri-container">{detection_results}</div>""", unsafe_allow_html=True)
        
            total_nutrition_str = f"""
    <h5 class="total-nutrition-title">Total Nutrition Values</h5>
    <div class="total-nutrition-container">
        <div class="total-nutri-box" id="calo-box">
            <span class="total-nutri-name">Calories</span><br>
            <span class="total-nutri-num">{total_nutrition['Calories']:.1f} kcal</span>
        </div>
        <div class="total-nutri-box" id="protein-box">
            <span class="total-nutri-name">Protein</span><br>
            <span class="total-nutri-num">{total_nutrition.get('Protein', 0.0):.1f} gram</span>
        </div>
        <div class="total-nutri-box">
            <span class="total-nutri-name">Fat</span><br>
            <span class="total-nutri-num">{total_nutrition['Fat']:.1f} gram</span>
        </div>
        <div class="total-nutri-box">
            <span class="total-nutri-name">Saturates</span><br>
            <span class="total-nutri-num">{total_nutrition['Saturates']:.1f} gram</span>
        </div>
        <div class="total-nutri-box">
            <span class="total-nutri-name">Sugar</span><br>
            <span class="total-nutri-num">{total_nutrition['Sugar']:.1f} gram</span>
        </div>
        <div class="total-nutri-box">
            <span class="total-nutri-name">Salt</span><br>
            <span class="total-nutri-num">{total_nutrition['Salt']:.1f} gram</span>
        </div>
    </div>
"""

            total_nutrition_placeholder.markdown(total_nutrition_str, unsafe_allow_html=True)
            st.session_state.last_detected_dishes = {dish: 1 for dish in displayed_dishes}
            st.session_state.last_total_nutrition = total_nutrition
    
            displayed_dishes.clear()
    
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
    original_size = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 3)

    # with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4', dir='/tmp') as mp4_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4', dir=tempfile.gettempdir()) as mp4_file:
        mp4_filename = mp4_file.name
        out = cv2.VideoWriter(mp4_filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
    with open(mp4_filename, "rb") as file:
        the_mp4 = file.read()

    st_frame = st.empty()

    

    col1, col2, col3 = st.columns(3, gap="large")
    with col1:
        rewind_button = st.button("Rewind 10s", use_container_width=True)
    with col2:
        stop_button = st.button("Stop", use_container_width=True)
        stop_pressed = False
    with col3:
        fast_forward_button = st.button("Fast-forward 10s", use_container_width=True)

    frame_count = 0
    start_time = time.time()
        
    stop_pressed = False
    skip_frames = 0

    total_nutrition_placeholder = st.empty()
    st.markdown("""<br>
                        <h5 class="detection-results">Detection Results</h5><p class="small-text-below-results">We found the following foods in your meal</p>""", unsafe_allow_html=True)
    

    displayed_dishes = set()
    nutrition_data = []
    
    total_nutrition = {
        "Calories": 0,
        "Fat": 0,
        "Saturates": 0,
        "Sugar": 0,
        "Salt": 0,
        "Protein": 0
    }
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

        new_detections = False  
        detection_results = ""

            
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
            st_frame.image(im_rgb, caption='Predicted video', use_column_width=True)

    
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
                    bbox_image_html = f'<img src="data:image/jpeg;base64,{img_str}" class="img-each-nutri" ">'


                if class_name == "Con nguoi (Human)" and class_name not in displayed_dishes:
                    detection_results += f"<p class='human-class-name'><b>Class name:</b> {class_name}</p><p class='human-confident'><b>Confidence:</b> {confident}%</p><hr style='border: none; border-top: 1px dashed black; width: 80%;'>"
                    displayed_dishes.add(class_name)
                    new_detections = True
                elif class_name not in displayed_dishes:
                    nutrition = class_names[int(class_id)]["nutrition"]
                    if nutrition:
                        displayed_dishes.add(class_name)
                        new_detections = True

                        calories_desc = get_nutri_score_color("Calories", nutrition.get('Calories'), serving)
                        fat_color, fat_desc = get_nutri_score_color("Fat", nutrition.get('Fat'), serving)
                        saturates_color, saturates_desc = get_nutri_score_color("Saturates", nutrition.get('Saturates'), serving)
                        sugar_color, sugar_desc = get_nutri_score_color("Sugar", nutrition.get('Sugar'), serving)
                        salt_color, salt_desc = get_nutri_score_color("Salt", nutrition.get('Salt'), serving)

                        percentage_contribution = calculate_nutrient_percentage(nutrition)

                        nutrition_str = f"""
                            <div class="each-nutri-container">
                            <div class="each-nutri-box" style="background-color: "transparent";">
                                {bbox_image_html}
                            </div>
                            <div  id="calo-each-nutri-box" class="each-nutri-box" style="background-color: transparent;">
                            <span class="each-nutri-name">Calories</span><br>
                            <p class="each-nutri-number">{nutrition.get('Calories')} kcal</p>
                            <span id="calo-each-nutri-percentage" class="each-nutri-percentage">{percentage_contribution['Calories']:.1f}%</span>
                            </div>
                            <div id="protein-each-nutri-box" class="each-nutri-box">
                            <span class="each-nutri-name">Protein</span><br>
                            <p class="each-nutri-number">{nutrition.get('Protein')} gram</p>
                            <span id="protein-each-nutri-percentage" class="each-nutri-percentage">{percentage_contribution.get('Protein', 0):.1f}%</span>
                            </div>
                            <div class="each-nutri-box" style="background-color: {fat_color};">
                            <span class="each-nutri-name">Fat</span><br>
                            <p class="each-nutri-number">{nutrition.get('Fat')} gram</p>
                            <span class="each-nutri-percentage">{percentage_contribution['Fat']:.1f}%</span>
                            </div>
                            <div class="each-nutri-box" style="background-color: {saturates_color};">
                            <span class="each-nutri-name">Saturates</span><br>
                            <p class="each-nutri-number">{nutrition.get('Saturates')} gram</p>
                            <span class="each-nutri-percentage">{percentage_contribution['Saturates']:.1f}%</span>
                            </div>
                            <div class="each-nutri-box" style="background-color: {sugar_color};">
                            <span class="each-nutri-name">Sugar</span><br>
                            <p class="each-nutri-number">{nutrition.get('Sugar')} gram</p>
                            <span class="each-nutri-percentage">{percentage_contribution['Sugar']:.1f}%</span>
                            </div>
                            <div class="each-nutri-box" style="background-color: {salt_color};">
                            <span class="each-nutri-name">Salt</span><br>
                            <p class="each-nutri-number">{nutrition.get('Salt')} gram</p>
                            <span class="each-nutri-percentage">{percentage_contribution['Salt']:.1f}%</span>
                            </div>
                            </div>
                    """

                        detection_results += (
            f"""<p class="item-header">{confident}%: <b>{class_name}</b></p>
            <p class="nutrition-header">Nutrition ({serving})</p>
            <p class="nutrition-facts">{nutrition_str}</p>
            <hr style="border: none; border-top: 1px dashed black; width: 80%;">
            """)
                        
                        for key in total_nutrition:
                            if key in nutrition:
                                total_nutrition[key] += nutrition[key]

                        nutrition_data.append((
                            class_name,
                            serving,
                            confident,
                            nutrition.get('Calories'),
                            nutrition.get('Protein'),
                            nutrition.get('Fat'),
                            nutrition.get('Saturates'),
                            nutrition.get('Sugar'),
                            nutrition.get('Salt')
                        ))
              
        if new_detections:
            scrollable_textbox = f"""<div class="result-nutri-container">{detection_results}</div>"""
            
            st.markdown(scrollable_textbox, unsafe_allow_html=True)

        total_nutrition_str = f"""
    <h5 class="total-nutrition-title">Total Nutrition Values</h5>
    <div class="total-nutrition-container">
        <div class="total-nutri-box" id="calo-box">
            <span class="total-nutri-name">Calories</span><br>
            <span class="total-nutri-num">{total_nutrition['Calories']:.1f} kcal</span>
        </div>
        <div class="total-nutri-box" id="protein-box">
            <span class="total-nutri-name">Protein</span><br>
            <span class="total-nutri-num">{total_nutrition.get('Protein', 0.0):.1f} gram</span>
        </div>
        <div class="total-nutri-box">
            <span class="total-nutri-name">Fat</span><br>
            <span class="total-nutri-num">{total_nutrition['Fat']:.1f} gram</span>
        </div>
        <div class="total-nutri-box">
            <span class="total-nutri-name">Saturates</span><br>
            <span class="total-nutri-num">{total_nutrition['Saturates']:.1f} gram</span>
        </div>
        <div class="total-nutri-box">
            <span class="total-nutri-name">Sugar</span><br>
            <span class="total-nutri-num">{total_nutrition['Sugar']:.1f} gram</span>
        </div>
        <div class="total-nutri-box">
            <span class="total-nutri-name">Salt</span><br>
            <span class="total-nutri-num">{total_nutrition['Salt']:.1f} gram</span>
        </div>
    </div>
"""
        total_nutrition_placeholder.markdown(total_nutrition_str, unsafe_allow_html=True)


        if stop_button:
            stop_pressed = True
            stop_button = None
            break

    cap.release()
    out.release()
    st.session_state.last_detected_dishes = {dish: 1 for dish in displayed_dishes}
    st.session_state.last_total_nutrition = total_nutrition
    displayed_dishes.clear()


    # with tempfile.NamedTemporaryFile(delete=False, suffix=".csv", dir="/tmp") as csv_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.csv', dir=tempfile.gettempdir()) as csv_file:
        csv_filename = csv_file.name
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Food Name", "Serving", "Confidence (%)", "Calories (kcal)", "Protein (g)", "Fat (g)", "Saturates (g)", "Sugar (g)", "Salt (g)"])
        writer.writerows(nutrition_data)
    with open(csv_filename, "rb") as file:
        the_csv = file.read()
    
    col1, col2 = st.columns(2, gap="large")
    with col1:    
        download_video = st.download_button(label="Download Processed Video",
                                data=the_mp4,
                                mime="video/mp4",
                                file_name=f"{timestamp}.mp4", 
                                use_container_width=True,)
        if download_video:
            os.remove(mp4_filename)
    with col2: 
        download_csv = st.download_button(label="Download Predictions CSV", 
                            data=the_csv, 
                            file_name=f"{timestamp}.csv", 
                            use_container_width=True,)
        if download_csv:
            os.remove(csv_filename)