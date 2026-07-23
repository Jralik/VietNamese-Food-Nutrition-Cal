import streamlit as st
import base64 
from pathlib import Path

st.set_page_config(
    page_title="FoodDetector",
    page_icon=":microscope:",
    layout="wide"
)

def img_to_base64(img_path):
    with open(img_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Convert your image to base64
img_path = './assets/img/about-page.png'
img_base64 = img_to_base64(img_path)

st.markdown('''
    <div id="top-section"></div>
    ''', unsafe_allow_html=True)
st.markdown(f"""
<div class="header-container">
    <img src="data:image/jpg;base64,{img_base64}" class="header-image">
    <div class="header-overlay">
        <div class="header-title2">📃 About 📃</div>
    </div>
</div>
""", unsafe_allow_html=True)
    
# Define the buttons with custom styling
st.markdown("""
 <div class="button-container">
    <a href="#dataset-section" class="button about-section" onclick="smoothScroll(event, 'dataset-section')">Dataset</a>
    <a href="#data-gathering-section" class="button about-section" onclick="smoothScroll(event, 'data-gathering-section')">Data Gathering</a>
    <a href="#data-annotation-section" class="button about-section" onclick="smoothScroll(event, 'data-annotation-section')">Data Annotation</a>
    <a href="#data-processing-section" class="button about-section" onclick="smoothScroll(event, 'data-processing-section')">Data Processing</a>
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

    <style>
        .button-container {
            display: flex;
            justify-content: space-between;
            flex-wrap: wrap;
            margin-top: 20px;
        }

        .button {
            width: calc(25% - 5px);
            display: inline-block;
            padding: 5px;
            background-color: var(--button-color-yellow);
            color: black !important;
            font-size: calc(12px + 0.3vw);
            font-weight: 700;
            text-align: center;
            text-decoration: none;
            border-radius: 8px;
            border: 2px solid transparent;
            transition: 0.3s;
            white-space: nowrap;
            text-align: center;
        }

        @media (min-width: 2000px) {
            .button {
                font-size: calc(12px + 0.15vw);
            }
        }
        
        .button:hover {
            background-color: var(--button-color-yellow-hover);
            color: var(--grey-hover) !important;
            text-decoration: none;
        }
        
        @media screen and (max-width: 650px) {
            .button {
                width: calc(50% - 5px);
                display: inline-block;
                margin-bottom: 10px;
                padding: 10px;
                font-size: calc(12px + 0.4vw);
            }
        }
        
        @media screen and (max-width: 380px) {
            .button {
                width: calc(50% - 4px);
                display: inline-block;
                margin-bottom: 8px;
                padding: 8px;
                font-size: calc(10px + 0.4vw);
            }
        }
    </style>
    """, unsafe_allow_html=True)



def render_content():  
    # st.title(":package: Dataset")
    
    st.markdown('''
    <div id="dataset-section"></div>
    ''', unsafe_allow_html=True)
    st.divider()
    st.markdown('''
    <h4 id="dataset-section" class="dataset-page">VietFood67: A Dataset for Vietnamese Food Detection</h4>
    <ul class="define dataset-page">
        <li class="define-li dataset-page">This dataset contains <code>30,360</code> images with <code>68</code> classes which included an extra 
        class for recognizing human faces as the purpose of this research is to detect and monitor people eating activity so 
        being able to know the human existence during the detection can give a more wholesome result. After all, the eating duration 
        can also be derived from human detection along with the dishes.</li>
        <li class="define-li dataset-page">VietFood67 is divided in <code>70%</code>/<code>20%</code>/<code>10%</code> with <code>21,264</code> 
        images for <code>train</code> set, <code>6,074</code> images for <code>test</code> set and <code>3,022</code> images for <code>valid</code> set.</li>
    </ul>

    ''', unsafe_allow_html=True)
    st.markdown('''<br>''', unsafe_allow_html=True)

        
    markdown_table_1 = """
        | Class ID | Food Names                               |
        |----------|------------------------------------------|
        | 0        | Bánh canh (Vietnamese thick noodle soup) |
        | 1        | Bánh chưng (Square sticky rice cake)     |
        | 2        | Bánh cuốn (Rolled rice pancake)          |
        | 3        | Bánh khọt (Mini savory pancakes)         |
        | 4        | Bánh mì (Vietnamese baguette sandwich)   |
        | 5        | Bánh tráng (Rice paper)                  |
        | 6        | Bánh tráng trộn (Rice paper salad)       |
        | 7        | Bánh xèo (Vietnamese sizzling pancake)   |
        | 8        | Bò kho (Beef stew)                       |
        | 9        | Bò lá lốt (Grilled beef wrapped in betel leaves) |
        | 10       | Bông cải (Cauliflower)                   |
        | 11       | Bún (Rice vermicelli)                    |
        | 12       | Bún bò Huế (Spicy beef noodle soup)      |
        | 13       | Bún chả (Grilled pork with vermicelli)   |
        | 14       | Bún đậu (Vermicelli with tofu)           |
        | 15       | Bún mắm (Fermented fish noodle soup)     |
        | 16       | Bún riêu (Crab noodle soup)              |
        | 17       | Cá (Fish)                                |
        | 18       | Cà chua (Tomato)                         |
        | 19       | Cà pháo (Pickled eggplant)               |
        | 20       | Cà rốt (Carrot)                          |
        | 21       | Canh (Soup)                              |
        | 22       | Chả (Vietnamese pork roll)               |
        | 23       | Chả giò (Spring rolls)                   |
        | 24       | Chanh (Lime)                             |
        | 25       | Cơm (Rice)                               |
        | 26       | Cơm tấm (Broken rice)                    |
        | 27       | Con người (Human)                        |
        | 28       | Củ kiệu (Pickled scallion head)          |
        | 29       | Cua (Crab)                               |
        | 30       | Đậu hũ (Tofu)                            |
        | 31       | Dưa chua (Pickled vegetables)            |
        | 32       | Dưa leo (Cucumber)                       |
    """

    markdown_table_2 = """
        | Class ID | Food Names                               |
        |----------|------------------------------------------|
        | 33       | Gỏi cuốn (Fresh spring rolls)            |
        | 34       | Hamburger                                |
        | 35       | Heo quay (Roast pork)                    |
        | 36       | Hủ tiếu (Clear rice noodle soup)         |
        | 37       | Khổ qua thịt (Stuffed bitter melon soup) |
        | 38       | Khoai tây chiên (French fries)           |
        | 39       | Lẩu (Hotpot)                             |
        | 40       | Lòng heo (Pork offal)                    |
        | 41       | Mì (Egg noodles)                         |
        | 42       | Mực (Squid)                              |
        | 43       | Nấm (Mushroom)                           |
        | 44       | Ốc (Snails)                              |
        | 45       | Ớt chuông (Bell pepper)                  |
        | 46       | Phở (Vietnamese noodle soup)             |
        | 47       | Phô mai (Cheese)                         |
        | 48       | Rau (Vegetables)                         |
        | 49       | Salad (Salad)                            |
        | 50       | Thịt bò (Beef)                           |
        | 51       | Thịt gà (Chicken)                        |
        | 52       | Thịt heo (Pork)                          |
        | 53       | Thịt kho (Braised pork)                  |
        | 54       | Thịt nướng (Grilled meat)                |
        | 55       | Tôm (Shrimp)                             |
        | 56       | Trứng (Egg)                              |
        | 57       | Xôi (Sticky rice)                        |
        | 58       | Bánh bèo (Vietnamese savory steamed rice cake) |
        | 59       | Cao lầu (Cao lầu noodles)                |
        | 60       | Mì Quảng (Quang-style noodles)           |
        | 61       | Cơm chiên Dương Châu (Yangzhou fried rice)|
        | 62       | Bún chả cá (Fish cake noodle soup)       |
        | 63       | Cơm chiên gà (Fried rice with chicken)   |
        | 64       | Cháo lòng (Pork organ congee)            |
        | 65       | Nộm hoa chuối (Banana blossom salad)     |
        | 66       | Nui xào bò (Stir-fried macaroni with beef)|
        | 67       | Súp cua (Crab soup)                      |
    """



    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown(markdown_table_1)

    with col2:
        st.markdown(markdown_table_2)

    st.markdown('''
    <div id="data-gathering-section"></div>
    ''', unsafe_allow_html=True)
    st.divider()
    st.markdown('''
    <h4 id="data-gathering-section" class="dataset-page">🔍 Data Gathering 🔍</h4>
    <p class="define dataset-page">These pictures were collected from different sources to ensure its variety and complexity.</p>
    <ul class="define dataset-page">
        <li class="define-li dataset-page"><code>Google, Facebook, Shopee Food</code>: Most of the images were gathered from these platforms by searching the dish name with some keyword like "food review" or "cooking".</li>
        <li class="define-li dataset-page"><code>Youtube</code>: Frames from the video or shorts were extracted with the help from the <a href="https://roboflow.com/" target="_blank">Roboflow</a> annotation tools.</li>
        <li class="define-li dataset-page"><code>Personal Collection</code>: Some images were personally taken by using a smartphone to simulate the real-world situation of food detection.</li>
    </ul>
    ''', unsafe_allow_html=True)
    
    
    
    st.markdown('''
    <div id="data-annotation-section"></div>
    ''', unsafe_allow_html=True)
    st.divider()
    st.markdown('''
    <h4  id="data-annotation-section" class="dataset-page">✍️ Data Annotation ✍️</h4>
    <p class="define dataset-page">The bounding box annotation and labeling process was done by using <a href="https://roboflow.com/" target="_blank">Roboflow</a> tools. To speed up the process, a YOLOv10m model 
    was trained on a subset of the dataset and used for the <code>Auto Label</code> feature to help automatically annotate the remaining images before double-checking it manually.</p>
    ''', unsafe_allow_html=True)

    st.divider()
    st.markdown('''
    <div id="data-processing-section"></div>
    ''', unsafe_allow_html=True)
    st.markdown('''
    <h4 id="data-processing-section" class="dataset-page">⚙️ Data Processing ⚙️</h4>
    <p class="define dataset-page">Some augmentation techniques were used to make sure the model can generalize well and to resolve the imbalance volume between classes.</p>
    <ul class="define dataset-page">
        <li class="define-li dataset-page"><code>Bounding box cropping</code>: Minimum zoom of <code>5%</code> and a maximum of <code>20%</code>.</li>
        <li class="define-li dataset-page"><code>Bounding box flip</code>: Flip vertically.</li>
        <li class="define-li dataset-page"><code>Brightness adjustments</code>: Between <code>-15%</code> and <code>+15%</code>.</li>
        <li class="define-li dataset-page"><code>Mosaic augmentation</code></li>
    </ul>
    <p class="define dataset-page">Overall, the total images obtained for training the model after the augmentation process are 123,644 images.</p>
    ''', unsafe_allow_html=True)
        
    st.markdown('''
    <div>
        <a href="#top-section" class="top-button" onclick="smoothScroll(event, 'top-section')">
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 448 512" width="16" height="16">
            <path d="M240.971 130.524l194.343 194.343c9.373 9.373 9.373 24.569 0 33.941l-22.667 22.667c-9.357 9.357-24.522 9.375-33.901.04L224 227.495 69.255 381.516c-9.379 9.335-24.544 9.317-33.901-.04l-22.667-22.667c-9.373-9.373-9.373-24.569 0-33.941L207.03 130.525c9.372-9.373 24.568-9.373 33.941-.001z"/>
        </svg></a>                
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

def home_page():
    st.markdown(navbar('Home'), unsafe_allow_html=True)
    

def about_page():
    st.markdown(navbar('About'), unsafe_allow_html=True)
    
def styling_css():
    with open('./assets/css/general-style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def main():
    styling_css()
    
    query_params = st.query_params
    path = query_params.get("page", ["about"])[0].lower()
    
    # Determine the active page
    active_page = 'About' if path == "about" else 'Home'
    
    # Always render the navbar with the correct active page
    st.markdown(navbar(active_page), unsafe_allow_html=True)
    
    if path == "home":
        st.markdown('<h1 style="color: white; font-size: 40px;">About Section</h1>', unsafe_allow_html=True)
        st.write("This is the About section. Here you can add information about your project or organization.")
    else:
        render_content()
    
if __name__ == '__main__':
    main()
        