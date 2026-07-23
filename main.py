import streamlit as st
import google.generativeai as genai
import time
import base64 
from pathlib import Path
from streamlit_navigation_bar import st_navbar
from utils import (
    _display_detected_frame, detect_camera, detect_image, detect_video, detect_webcam, 
    load_onnx_model, load_model, calculate_bmi, calculate_tdee_mifflin_st_jeor, build_structured_facts
)

st.set_page_config(
    page_title="FoodDetector",
    page_icon=":microscope:",
    layout="wide"
)

# Initialize default user profile and RDA
if "user_profile" not in st.session_state:
    st.session_state.user_profile = {
        "age": 25,
        "sex": "female",
        "weight": 65.0,
        "height": 170.0,
        "activity_factor": 1.2,
        "goal": "maintain"
    }

if "user_rda" not in st.session_state:
    w = st.session_state.user_profile["weight"]
    h = st.session_state.user_profile["height"]
    a = st.session_state.user_profile["age"]
    s = st.session_state.user_profile["sex"]
    af = st.session_state.user_profile["activity_factor"]
    g = st.session_state.user_profile["goal"]
    
    tdee = calculate_tdee_mifflin_st_jeor(w, h, a, s, af)
    
    if g == "lose":
        calories_target = tdee - 500
    elif g == "gain":
        calories_target = tdee + 500
    else:
        calories_target = tdee
        
    st.session_state.user_rda = {
        "Calories": max(1200.0, calories_target),
        "Protein": w * 1.6,
        "Fat": (calories_target * 0.25) / 9.0,
        "Saturates": (calories_target * 0.08) / 9.0,
        "Sugar": 50.0,
        "Salt": 6.0
    }

# import streamlit as st
# from PIL import Image

# image = Image.open('./pages/bg-about-cuisine.jpg')

# st.image(image)
# st.markdown(f"""
# <style>
#     .stImage  {{
#         position: relative;
#         width: 100%;
#         height: calc(100px + 7vw);
#         overflow: hidden;
#     }}
# """, unsafe_allow_html=True)

# st.markdown(f"""
# <h1 class="header-title">📑 About FoodDetector</h1>
#             """, unsafe_allow_html=True)         

st.markdown('''
    <div id="top-section"></div>
    ''', unsafe_allow_html=True)
def img_to_base64(img_path):
    with open(img_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

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
        elif response.status_code == 402:
            print("Lỗi: Tài khoản hết hạn mức miễn phí hoặc yêu cầu thanh toán.")
        else:
            print(f"Lỗi API: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"Lỗi kết nối: {e}")
        
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

# Convert your image to base64
img_path = './assets/img/bg-about-cuisine.png'
img_base64 = img_to_base64(img_path)

# Convert your image to base64
img_path_nutrition = './assets/img/nutrition-table.png'
img_base64_nutrition = img_to_base64(img_path_nutrition)

st.markdown(f"""
<div class="header-container">
    <img src="data:image/jpg;base64,{img_base64}" class="header-image">
    <div class="header-overlay">
        <div class="header-title">Welcome to FoodDetector 🕵️</div>
        <div class="header-subtitle">An easy way to detect Vietnamese dishes!</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Import img
# def img_bg_cover(img_path):
#     with open(img_path, 'rb') as img_file:
#         return base64.b64encode(img_file.read()).decode('utf-8')

# current_path = Path(__file__).parent
# img_path = current_path / 'pages' / 'img' / 'bg-about-cuisine.jpg'
# img_cover = img_bg_cover(img_path)

# # Cover
# st.markdown(f"""
# <style>
#     .header-container {{
#         position: relative;
#         width: 100%;
#         height: calc(100px + 7vw);
#         overflow: hidden;
#     }}
#     .header-image {{
#         position: absolute;
#         top: 0;
#         left: 0;
#         width: 100%;
#         height: 100%;
#         background-image: url(data:image/jpg;base64,{img_cover});
#         background-size: cover;
#         background-position: center top;
#     }}
    
#     .header-title {{
#     position: absolute;
#     bottom: 0;
#     /* left: 50%;
#     transform: translateX(-50%); */
#     width: 100%;
#     color: white;
#     background-color: rgba(0, 0, 0, 0.6);
#     padding: 5px 10px;
#     letter-spacing: 1px;
#     font-weight: 800;
#     }}
# </style>

# <div class="header-container">
#     <div class="header-image"></div>
#     <h1 class="header-title">📑 About FoodDetector</h1>
# </div>
# """, unsafe_allow_html=True)
# End Cover

def render_left_side():         
    with st.container():
        # st.title("Welcome to _:green[FoodDetector]_ :male-detective:")
        st.divider()

    #     st.markdown('''
    # FoodDetector uses the _YOLOv10m_ pretrained models for fine-tuning with `VietFood57`, a new custom-made Vietnamese food dataset created for detecting local dishes and achieved a `mAP50` of `0.92`.  
    # It can be used to detect <a href="/Dataset" target="_blank" style="color: #4CAF50; font-weight: bold; font-style: italic; text-decoration: none;">`57`</a> Vietnamese dishes from a picture, video, webcam, and an IP camera through RTSP.
    # ''', unsafe_allow_html=True)

        st.markdown(f'''
    <ul class="define introduction" style="margin-top: 0; margin-bottom: 0;">
        <li class="define-li home-page">FoodDetector uses the <strong>YOLOv26</strong> pretrained models for fine-tuning with <code>VietFood67</code>, 
        an enhanced custom-made Vietnamese food dataset created for detecting local dishes and achieved a <code>mAP50</code> of <code>0.92</code>.</li>
        <li class="define-li home-page">It can be used to detect <a href="/dataset" target="_self">67</a> Vietnamese dishes from a picture, video, webcam, and an IP camera through RTSP.</li>
    </ul>
                    ''', unsafe_allow_html=True)


        st.divider()

        st.markdown(f'''
                    <h4>Adjust the confident score 🚩</h4>
                    ''', unsafe_allow_html=True)
        confidence = float(st.slider(
            label="",label_visibility="collapsed", min_value=10, max_value=100, value=50 
        ))/ 100
        
        st.markdown(f'''
    <style>
        #quick-note {{
        margin-left: 0;
        margin-bottom: 0.5rem;
    }}
    
    p#quick-note.define {{
        margin-top: 0;
    }}

    .title-text-score {{
        font-weight: 700;
        border-radius: 5px;
        background-color: var(--grey);
        padding: 0.5rem;
        display: inline;
    }}

    p.define.subtitle-text-score {{
        margin-bottom: 1rem;
    }}
    
    </style>
    <p class="define" id="quick-note"><strong>Quick note 📝</strong>: consideration for selecting the best suited confident score:</p>
    <div class="adjust-section">
        <p class="define title-text-score">High confident score (>= 50%):</p>
        <p class="define subtitle-text-score">Set a higher threshold will make the model to predict with a higher accuracy detection but it will have a low recall as fewer object will 
        be detected because of the high precision constraint.</p>
        <p class="define title-text-score">Low confident score (< 50%):</p>        
        <p class="define subtitle-text-score">Set a lower threshold will enable the model to detect more object - 
    high recall because of the low precision constraint.</p>
    </div>     
                ''', unsafe_allow_html=True)

        st.divider()
        st.markdown(f'''
                    <h4>Nutrition value score📊</h4>
                    ''', unsafe_allow_html=True)

        st.markdown(f'''
    <ul class="define nutrition" style="margin-top: 0; margin-bottom: 0;">
        <li class="define-li home-page">Our nutrition values are based on the <strong>Traffic Light system</strong>🚦.</li>
        <li class="define-li home-page">All nutrition information provided is approximate.</li>
    </ul>
                    ''', unsafe_allow_html=True)
        

        
        expander = st.expander("See more")  
        expander.markdown(f'''
<div class="nutrition-container">
    <img src="data:image/jpg;base64,{img_base64_nutrition}" class="nutrition-img">
    <ul class="nutrition-explain">
        <li class="nutrition-explain-details"><strong class="color-section" id="green">Green (Low)</strong>: Very healthy. Enjoy without worry.</li>
        <li class="nutrition-explain-details"><strong class="color-section" id="yellow">Yellow (Medium)</strong>: Consume in moderation or combine with healthier options.</li>
        <li class="nutrition-explain-details"><strong class="color-section" id="red">Red (High)</strong>: Limit consumption and look for healthier alternatives.</li>
    </ul>
    <p class="nutrition-explain-details">For more information, please refer to the <a href="https://www.nutricalc.co.uk/case-study/case-study-uk-traffic-light-front-of-pack-colour-thresholds/">NutriCalc</a>,
    <a href="https://heas.health.vic.gov.au/resources/government-guidelines/traffic-light-system/">Healthy Eating Advisory Service</a></p>

<style>
    li.nutrition-explain-details {{
        margin-bottom: 0.5rem !important;
        margin-top: 0.5rem !important;
    }}
    
    p.nutrition-explain-details {{
        font-weight: 400 !important;
        margin: 1rem 0 !important;
    }}
    
    img.nutrition-img {{
        margin-top: 1rem;
        margin-bottom: 1rem;
        width: 90%;
        display: block;
        margin-left: auto;
        margin-right: auto;
    }} 
    
    .color-section {{
        padding: 3px 6px;
        border-radius: 5px;
    }}
    
    #green {{
        background-color: var(--green-nu);
    }}
    
    #yellow {{
        background-color: var(--yellow-nu);
    }}
    
    #red {{
        background-color: var(--red-nu);
    }}
</style>
''', unsafe_allow_html=True)
        
        st.markdown(f'''<br><br>''', unsafe_allow_html=True)        
        model1 = load_model()
        model = load_onnx_model()

        st.markdown("""
    <style>
    /* Style the tab labels */
    button[data-baseweb="tab"] {
        padding: calc(8px + 0.2vw) calc(8px + 0.5vw);
        gap: 0;

    }
    button[data-baseweb="tab"] p {
        font-size: calc(9px + 0.3vw) !important;
        font-weight: 500 !important;        
    }
    
    div[data-baseweb="tab-list"] {
        gap: 0;
    }
    /* Style the active tab */
    button[data-baseweb="tab"][aria-selected="true"] {
        background-color: var(--button-color-yellow); /* Active tab color */
        border-radius: 8px 7px 0 0;
        color: black;
    }

    /* Style the inactive tabs */
    button[data-baseweb="tab"][aria-selected="false"] {
        color: var(--grey-code-expander);
    }
    
    div[data-baseweb="tab-border"] {
    }
    </style>
""", unsafe_allow_html=True)
        
        tab1, tab2, tab3, tab4 = st.tabs(["Image", "Video", "Webcam", "IP Camera"])

        with tab1:
            st.subheader("Image Upload :frame_with_picture:")

            # Accordion
            expander = st.expander("Instructions: Image upload and URL")  
            expander.markdown('''
    - Uploading image files from the user's local machine or using an image URL is supported.
    - After the prediction process, two buttons will appear to download the results as an image file with bounding boxes or a CSV file.
    - The results are generated when the user clicks the button and are named in the format: `"%date-%month-%year".jpg/csv`.
            ''', unsafe_allow_html=True)
            st.markdown(f'''
    <style>
    [data-testid="stExpanderDetails"] ul li {{
        font-size: calc(12px + 0.1vw);
        margin: 1rem 0 1rem 1.5rem;
        color: black
    }}
    .stExpander p {{
        font-size: calc(13px + 0.1vw);
        font-weight: 700;
        color: var(--brown);
        padding-left: 0.5rem;
    }}
    .st-emotion-cache-1h9usn1 {{
        background-color: var(--button-color-yellor);
        font-size: calc(16px +1vw);
    }}

    [data-testid="stExpanderDetails"] {{
        background-color: var(--grey-light);
        border-radius: 8px;
    }}
    </style>
                        ''', unsafe_allow_html=True)

            uploaded_file = st.file_uploader("Choose a picture", accept_multiple_files=False, type=['png', 'jpg', 'jpeg'])

            if uploaded_file:
                detect_image(confidence, model=model1, uploaded_file=uploaded_file)

                # detections = detect_image_onnx(model, uploaded_file, confidence)

            # st.markdown('<br><br>', unsafe_allow_html=True)
            # st.subheader("Enter a picture URL 	:link:")
            # with st.form("picture_form"):
            #     col1, col2 = st.columns([0.8, 0.2], gap="medium")
            #     with col1:
            #         picture_url = st.text_input("Label", label_visibility="collapsed", placeholder="https://ultralytics.com/images/bus.jpg")
            #     with col2:
            #         submitted = st.form_submit_button("Predict", use_container_width=True)
            # if submitted and picture_url:
            #     detect_image(confidence, model=model1, uploaded_file=picture_url, url=True)            

        with tab2:
                        
            st.subheader("Video Upload :movie_camera:")
            expander = st.expander("Instructions: Video upload and URL")  
            expander.markdown('''
- Video: upload video files `(.mp4, .mpeg4, etc.)` from the user's local machine.
- Youtube video or shorts URL links are supported for real-time prediction.
- The results will be in a CSV file recording all dishes detected across all frames (no image results).
            ''', unsafe_allow_html=True)
            
            uploaded_clip = st.file_uploader("Choose a clip", accept_multiple_files=False, type=['mp4'])
            if uploaded_clip:
                detect_video(conf=confidence, uploaded_file=uploaded_clip, model=model1)

            else:
                st.markdown('<br><br>', unsafe_allow_html=True) 
                st.subheader("Enter YouTube URL :tv:")
                # tube = st.empty()
                with st.form("youtube_form"):
                    col1, col2 = st.columns([0.8, 0.2], gap="medium")
                    with col1:
                        youtube_url = st.text_input("Label", label_visibility="collapsed", placeholder="https://youtu.be/LNwODJXcvt4")
                    with col2:
                        submitted = st.form_submit_button("Predict", use_container_width=True)
                if submitted and youtube_url:            
                    _display_detected_frame(conf=confidence, model=model1, 
                                           
                                            youtube_url=youtube_url)

        with tab3:
            
            st.header("Webcam :camera:")
            expander = st.expander("Instructions: Webcam connection")  
            expander.markdown('''
- Webcam: [Streamlit-webrtc](https://github.com/whitphx/streamlit-webrtc) is used to handle local webcam connection due to deployment on [Streamlit Community Cloud](https://docs.streamlit.io/deploy/streamlit-community-cloud).
- Users can choose their webcam input for live detection.
- No result files will be generated as the process may run continuously.

            ''', unsafe_allow_html=True)
            detect_webcam(confidence, model=model1)

        with tab4:
            
            st.header("IP Camera :video_camera:")
            expander = st.expander("Instructions: IP Camera connection")  
            expander.markdown('''
- IP Camera: A RTSP address of the user’s camera must be provided.
- The camera must be configured beforehand to allow connection from an external network.
            ''', unsafe_allow_html=True)    
            
            st.text("Enter your Camera (RTSP) address: ")
            with st.form("ip_camera_form"):
                col1, col2 = st.columns([2, 8])
                with col1:
                    st.write("rtsp://admin:") 
                with col2:
                    address = st.text_input(
                        "Label", 
                        label_visibility="collapsed", 
                        placeholder="hd543211@192.168.14.106:554/Streaming/channels/101"
                    )
                    
                col1, col2 = st.columns([2, 1.35])
                with col1:
                    submitted = st.form_submit_button("Connect")
                with col2:
                    cancel = st.form_submit_button("Disconnect")
            
                if submitted:
                    if address:
                        detect_camera(confidence, model1, address=address)
                    else:
                        st.error("Please enter a valid RTSP camera URL")
                
                if cancel:
                    if address:
                        detect_camera(confidence, model1, address="")
                        st.toast("Disconnected", icon="✅")

    st.markdown('''
    <div>
        <a href="#top-section" class="top-button" onclick="smoothScroll(event, 'top-section')">
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 448 512" width="16" height="16">
            <path d="M240.971 130.524l194.343 194.343c9.373 9.373 9.373 24.569 0 33.941l-22.667 22.667c-9.357 9.357-24.522 9.375-33.901.04L224 227.495 69.255 381.516c-9.379 9.335-24.544 9.317-33.901-.04l-22.667-22.667c-9.373-9.373-9.373-24.569 0-33.941L207.03 130.525c9.372-9.373 24.568-9.373 33.941-.001z"/>
        </svg>
        </a>                
    </div>
    
    <script>
    function smoothScroll(event, targetId) {
        event.preventDefault();
        const targetElement = document.getElementById(targetId);
        if (targetElement) {
            targetElement.scrollIntoView({ behavior: 'smooth' });
        }
    }
    </script>
                ''', unsafe_allow_html=True)

def render_right_side():
    st.markdown("### 💬 AI Nutritionist (Tư vấn dinh dưỡng)")
    st.markdown("Hỏi chuyên gia dinh dưỡng AI về thực đơn, lượng calo, chất béo hoặc chế độ dinh dưỡng của các món ăn Việt Nam!")
    
    # User Profile Configuration
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 👤 Hồ sơ Sức khỏe")
    
    # Retrieve current values from session state to persist them
    profile = st.session_state.get("user_profile", {
        "age": 25,
        "sex": "female",
        "weight": 65.0,
        "height": 170.0,
        "activity_factor": 1.2,
        "goal": "maintain"
    })
    
    # Render inputs in sidebar
    age = st.sidebar.number_input("Tuổi", min_value=1, max_value=120, value=int(profile["age"]), key="profile_age")
    sex = st.sidebar.selectbox(
        "Giới tính", 
        ["male", "female"], 
        index=0 if profile["sex"] == "male" else 1, 
        format_func=lambda x: "Nam (Male)" if x == "male" else "Nữ (Female)",
        key="profile_sex"
    )
    weight = st.sidebar.number_input("Cân nặng (kg)", min_value=1.0, max_value=300.0, value=float(profile["weight"]), key="profile_weight")
    height = st.sidebar.number_input("Chiều cao (cm)", min_value=50.0, max_value=250.0, value=float(profile["height"]), key="profile_height")
    
    activity_factor_map = {
        1.2: "Ít vận động (sedentary)",
        1.375: "Vận động nhẹ (lightly active)",
        1.55: "Vận động vừa (moderately active)",
        1.725: "Vận động nhiều (very active)",
        1.9: "Vận động nặng (extra active)"
    }
    
    # Find current activity factor index
    af_keys = list(activity_factor_map.keys())
    af_index = af_keys.index(profile["activity_factor"]) if profile["activity_factor"] in af_keys else 0
    
    activity_factor = st.sidebar.selectbox(
        "Mức độ vận động",
        af_keys,
        index=af_index,
        format_func=lambda x: activity_factor_map[x],
        key="profile_activity_factor"
    )
    
    goal_map = {
        "lose": "Giảm cân (Lose weight)",
        "maintain": "Giữ cân (Maintain weight)",
        "gain": "Tăng cân (Gain weight)"
    }
    goal_keys = list(goal_map.keys())
    goal_index = goal_keys.index(profile["goal"]) if profile["goal"] in goal_keys else 1
    
    goal = st.sidebar.selectbox(
        "Mục tiêu sức khỏe",
        goal_keys,
        index=goal_index,
        format_func=lambda x: goal_map[x],
        key="profile_goal"
    )
    
    # Calculate BMI and TDEE using engine
    bmi = calculate_bmi(weight, height)
    tdee = calculate_tdee_mifflin_st_jeor(weight, height, age, sex, activity_factor)
    
    # Calculate RDA based on TDEE and Goal
    if goal == "lose":
        calories_target = tdee - 500
    elif goal == "gain":
        calories_target = tdee + 500
    else:
        calories_target = tdee
        
    user_rda = {
        "Calories": max(1200.0, calories_target),
        "Protein": weight * 1.6,
        "Fat": (calories_target * 0.25) / 9.0,
        "Saturates": (calories_target * 0.08) / 9.0,
        "Sugar": 50.0,
        "Salt": 6.0
    }
    
    # Save back to session state
    st.session_state.user_profile = {
        "age": age,
        "sex": sex,
        "weight": weight,
        "height": height,
        "activity_factor": activity_factor,
        "goal": goal
    }
    st.session_state.user_rda = user_rda
    
    # Display calculated indicators in sidebar
    st.sidebar.markdown(f"""
    **📊 Chỉ số sức khỏe:**
    - **BMI**: `{bmi:.1f}`
    - **TDEE**: `{tdee:.0f} kcal`
    - **RDA Calo Mục tiêu**: `{user_rda['Calories']:.0f} kcal`
    """)

    # Get API key from Sidebar or Session State
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔑 Cấu hình Trợ lý AI")
    
    llm_provider = st.sidebar.selectbox(
        "Chọn Trợ lý AI",
        ["Gemini", "Cerebras", "OpenRouter"],
        index=0,
        help="Chọn AI bạn muốn sử dụng để đánh giá dinh dưỡng và tư vấn."
    )
    st.session_state.llm_provider = llm_provider

    api_key_input = st.sidebar.text_input("Nhập Gemini API Key", type="password", help="Nhận key miễn phí từ Google AI Studio")
    cerebras_api_key_input = st.sidebar.text_input("Nhập Cerebras API Key (Tùy chọn)", type="password", help="Nhận key miễn phí từ Cerebras Cloud Console để sử dụng Llama/Gemma với tốc độ siêu nhanh!")
    openrouter_api_key_input = st.sidebar.text_input("Nhập OpenRouter API Key (Tùy chọn)", type="password", help="Nhận key từ OpenRouter để truy cập hàng trăm mô hình AI (có bản miễn phí)!")
    
    # Store key in session state
    if api_key_input:
        st.session_state.gemini_api_key = api_key_input
    if cerebras_api_key_input:
        st.session_state.cerebras_api_key = cerebras_api_key_input
    if openrouter_api_key_input:
        st.session_state.openrouter_api_key = openrouter_api_key_input
        
    # Check if API key is configured
    api_key = st.session_state.get("gemini_api_key", "")
    cerebras_api_key = st.session_state.get("cerebras_api_key", "")
    openrouter_api_key = st.session_state.get("openrouter_api_key", "")
    
    if llm_provider == "Cerebras" and cerebras_api_key:
        available_models = get_cerebras_models(cerebras_api_key)
        cerebras_model = st.sidebar.selectbox(
            "Chọn Mô hình Cerebras",
            available_models,
            index=0,
            help="Danh sách các mô hình khả dụng từ tài khoản Cerebras của bạn."
        )
        st.session_state.cerebras_model = cerebras_model

    if llm_provider == "OpenRouter" and openrouter_api_key:
        available_models = get_openrouter_models(openrouter_api_key)
        openrouter_model = st.sidebar.selectbox(
            "Chọn Mô hình OpenRouter",
            available_models,
            index=0,
            help="Danh sách các mô hình khả dụng từ OpenRouter."
        )
        st.session_state.openrouter_model = openrouter_model
    
    if llm_provider == "Gemini":
        if not api_key:
            st.info("💡 **Gợi ý**: Hãy nhập **Gemini API Key** ở thanh bên (Sidebar) để kích hoạt Trợ lý dinh dưỡng AI. Bạn có thể lấy khóa API miễn phí từ [Google AI Studio](https://aistudio.google.com/).")
            
            # Display mock system message when no key is set
            with st.chat_message("assistant"):
                st.markdown("""Xin chào! Tôi là Trợ lý Dinh dưỡng AI. 🥗
                
Sau khi bạn cấu hình khóa API ở Sidebar, tôi có thể giúp bạn:
- Phân tích hàm lượng calo và dinh dưỡng trong thực đơn của bạn.
- Đưa ra lời khuyên ăn uống lành mạnh phù hợp với các món ăn Việt Nam.
- Thiết kế chế độ ăn kiêng, tăng cơ, giảm mỡ,...

*Hãy nhập API Key ở thanh bên để bắt đầu trò chuyện nhé!*""")
            return

        # Initialize Gemini
        try:
            genai.configure(api_key=api_key)
        except Exception as e:
            st.error(f"Lỗi cấu hình API Key: {e}")
            return
    elif llm_provider == "Cerebras":
        if not cerebras_api_key:
            st.info("💡 **Gợi ý**: Hãy nhập **Cerebras API Key** ở thanh bên (Sidebar) để kích hoạt Trợ lý dinh dưỡng AI sử dụng Cerebras.")
            
            # Display mock system message when no key is set
            with st.chat_message("assistant"):
                st.markdown("""Xin chào! Tôi là Trợ lý Dinh dưỡng AI (sử dụng Cerebras). 🥗
                
Sau khi bạn cấu hình khóa API ở Sidebar, tôi có thể tư vấn dinh dưỡng cho bạn với tốc độ cực nhanh!
*Hãy nhập Cerebras API Key ở thanh bên để bắt đầu trò chuyện nhé!*""")
            return
    else: # OpenRouter
        if not openrouter_api_key:
            st.info("💡 **Gợi ý**: Hãy nhập **OpenRouter API Key** ở thanh bên (Sidebar) để kích hoạt Trợ lý dinh dưỡng AI sử dụng OpenRouter.")
            
            # Display mock system message when no key is set
            with st.chat_message("assistant"):
                st.markdown("""Xin chào! Tôi là Trợ lý Dinh dưỡng AI (sử dụng OpenRouter). 🥗
                
Sau khi bạn cấu hình khóa API ở Sidebar, tôi có thể tư vấn dinh dưỡng cho bạn bằng hàng trăm mô hình AI khác nhau!
*Hãy nhập OpenRouter API Key ở thanh bên để bắt đầu trò chuyện nhé!*""")
            return

    # Initialize chat history
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    # Dedicated scrollable container for chat history
    chat_container = st.container(height=560, border=True)

    # Display chat messages from history inside scroll container
    with chat_container:
        for message in st.session_state.chat_messages:
            with st.chat_message(message["role"]):
                if message.get("reasoning_details"):
                    with st.expander("💭 Suy nghĩ của AI (Reasoning)"):
                        st.write(message["reasoning_details"])
                st.markdown(message["content"])

        # Predefined suggestions when chat history is empty
        selected_suggestion = None
        if not st.session_state.chat_messages:
            st.markdown("**💡 Câu hỏi gợi ý tư vấn mẫu:**")
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
                if col.button(sug, key=f"sug_{idx}", use_container_width=True):
                    selected_suggestion = sug

    # React to user input
    prompt = st.chat_input("Hãy hỏi tôi về dinh dưỡng...")
    
    if selected_suggestion:
        prompt = selected_suggestion

    if prompt:
        with chat_container:
            # Display user message in chat message container
            with st.chat_message("user"):
                st.markdown(prompt)
            # Add user message to chat history
            st.session_state.chat_messages.append({"role": "user", "content": prompt})

            # Generate response using Gemini
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                with st.spinner("Đang suy nghĩ..."):
                    try:
                        # Build dynamic meal context from session state using structured facts JSON
                        last_dishes = st.session_state.get("last_detected_dishes")
                        last_nutri = st.session_state.get("last_total_nutrition")
                        user_profile = st.session_state.get("user_profile")
                        user_rda = st.session_state.get("user_rda")
                        meal_context = ""
                        if last_dishes and last_nutri and user_profile and user_rda:
                            facts = build_structured_facts(last_nutri, user_profile, user_rda, last_dishes)
                            import json
                            meal_context = (
                                f"\n[BỐI CẢNH BỮA ĂN VỪA PHÁT HIỆN DƯỚI DẠNG JSON FACTS]:\n"
                                f"{json.dumps(facts, indent=2, ensure_ascii=False)}\n"
                                f"Hãy sử dụng các sự thật (facts) có cấu trúc này để trả lời nếu người dùng hỏi về món ăn hiện tại, bữa ăn của họ, hoặc xin lời khuyên dinh dưỡng."
                            )

                        system_context = (
                            "Bạn là một chuyên gia tư vấn dinh dưỡng AI chuyên nghiệp chuyên về ẩm thực Việt Nam.\n"
                            "Nhiệm vụ của bạn là tư vấn dinh dưỡng cho người dùng dựa trên câu hỏi của họ, "
                            "phân tích calo, chất béo, carb, protein, đường, muối của các món ăn Việt Nam "
                            "và đề xuất các mẹo ăn uống lành mạnh (ví dụ: bớt nước lèo khi ăn phở, ăn thêm rau xà lách...).\n"
                            "Hãy trả lời bằng tiếng Việt, giọng điệu lịch sự, khoa học, thực tế và ngắn gọn dễ hiểu.\n"
                            "Không nói dông dài, đi thẳng vào vấn đề chính. Chỉ đưa ra câu trả lời trực tiếp bằng tiếng Việt, không lặp lại bất kỳ mô tả vai trò, nhiệm vụ hay cấu hình hệ thống nào.\n"
                            f"{meal_context}"
                        )

                        if llm_provider.startswith("Cerebras"):
                            model_id = st.session_state.get("cerebras_model", "gpt-oss-120b")

                            headers = {
                                "Authorization": f"Bearer {cerebras_api_key}",
                                "Content-Type": "application/json"
                            }
                            payload = {
                                "model": model_id,
                                "messages": [{"role": "system", "content": system_context}] + st.session_state.chat_messages
                            }

                            import requests
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

                            headers = {
                                "Authorization": f"Bearer {openrouter_api_key}",
                                "Content-Type": "application/json"
                            }
                            
                            # Build message payload passing back reasoning_details unmodified
                            payload_messages = []
                            for msg in st.session_state.chat_messages:
                                m = {
                                    "role": msg["role"],
                                    "content": msg["content"]
                                }
                                if "reasoning_details" in msg and msg["reasoning_details"]:
                                    m["reasoning_details"] = msg["reasoning_details"]
                                payload_messages.append(m)

                            payload = {
                                "model": model_id,
                                "messages": [{"role": "system", "content": system_context}] + payload_messages,
                                "reasoning": {"enabled": True},
                                "max_tokens": 4000
                            }

                            import requests
                            res = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
                            if res.status_code == 200:
                                data = res.json()
                                assistant_msg = data["choices"][0]["message"]
                                response_text = assistant_msg.get("content", "")
                                reasoning_details = assistant_msg.get("reasoning_details")
                                
                                # Display reasoning if available before main response
                                if reasoning_details:
                                    with st.expander("💭 Suy nghĩ của AI (Reasoning)"):
                                        st.write(reasoning_details)
                                message_placeholder.markdown(response_text)
                                
                                # Store in session state including reasoning_details
                                st.session_state.chat_messages.append({
                                    "role": "assistant",
                                    "content": response_text,
                                    "reasoning_details": reasoning_details
                                })
                            else:
                                message_placeholder.markdown(f"❌ Lỗi từ OpenRouter API (Mã lỗi {res.status_code}): {res.text}")
                            return
                        
                        # Check if we already found a working model in this session
                        working_model = st.session_state.get("working_model_name", "")
                        
                        if working_model:
                            model = genai.GenerativeModel(working_model, system_instruction=system_context)
                            chat = model.start_chat(history=[])
                            response = chat.send_message(prompt)
                            response_text = response.text
                            message_placeholder.markdown(response_text)
                            st.session_state.chat_messages.append({"role": "assistant", "content": response_text})
                        else:
                            # Find available models and prioritize them
                            models_to_try = []
                            try:
                                models = list(genai.list_models())
                                api_names = [m.name.replace("models/", "", 1) for m in models if "generateContent" in m.supported_generation_methods]
                                api_names.sort(reverse=True)
                                models_to_try = api_names
                            except Exception:
                                pass
                                
                            # Standard fallbacks if list_models failed
                            if not models_to_try:
                                models_to_try = ["gemini-3.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]
                            
                            # Ensure latest models are at the very beginning of the trial list
                            for latest in ["gemini-3.5-flash", "gemini-2.0-flash"]:
                                if latest in models_to_try:
                                    models_to_try.remove(latest)
                                models_to_try.insert(0, latest)
                            
                            # Loop and test models in sequence
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
                                    # Cache the working model
                                    st.session_state.working_model_name = model_name
                                    break
                                except Exception as e:
                                    last_error = str(e)
                                    # If it's a 404, 429, quota, not found, or unsupported model error, try the next one
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
                        message_placeholder.markdown(f"❌ Đã xảy ra lỗi khi kết nối với Gemini API: {e}")

def render_content():
    col_left, col_right = st.columns([0.55, 0.45], gap="large")
    with col_left:
        render_left_side()
    with col_right:
        render_right_side()

# Nav bar
def navbar(active_page):
    return f"""
    <div class="custom-navbar">
        <div class="nav-items">
            <a href="/main" target="_self" class="nav-item {'active' if active_page == 'Home' else ''}">🏠 Home</a>
            <a href="/dataset" target="_self" class="nav-item {'active' if active_page == 'About' else ''}">📄 About</a>
        </div>
        <a href="https://github.com/Jralik/VietNamese-Food-Nutrition-Cal" target="_blank" class="nav-item">
            <svg id="github-icon" height="32" aria-hidden="true" viewBox="0 0 16 16" version="1.1" width="32" data-view-component="true">
                <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z" fill="currentColor"></path>
            </svg>
        </a>
    </div>
    """

def styling_css():
    with open('./assets/css/general-style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
 
        
def home_page():
    st.markdown(navbar('Home'), unsafe_allow_html=True)
    

def about_page():
    st.markdown(navbar('About'), unsafe_allow_html=True)
    

# Main app logic
def main():
        # Get the current page from the URL
    styling_css()
    query_params = st.query_params
    path = query_params.get("page", ["home"])[0].lower()
    
    # Always render the navbar
    st.markdown(navbar('Home' if path == 'home' else 'About'), unsafe_allow_html=True)
    
    if path == "about":
        st.markdown('<h1 style="color: white; font-size: 40px;">About Section</h1>', unsafe_allow_html=True)
        st.write("This is the About section. Here you can add information about your project or organization.")
    else:
        render_content()

if __name__ == "__main__":
    main()