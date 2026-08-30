"""Thư viện component UI dùng chung cho FoodDetector AI.

Tất cả HTML/CSS trình bày (KPI card, gauge, progress, chip traffic-light,
card món ăn, hero, navbar, biểu đồ) được tập trung ở đây để không lặp lại
trong main.py / utils.py. Không chứa logic nhận diện hay AI.
"""
import base64
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

ASSET_DIR = Path(__file__).parent / "assets"
CSS_PATH = ASSET_DIR / "css" / "general-style.css"
IMG_DIR = ASSET_DIR / "img"
DEMO_DIR = ASSET_DIR / "demo"

GITHUB_URL = "https://github.com/Jralik/VietNamese-Food-Nutrition-Cal"

# Nhãn tiếng Việt cho các chất dinh dưỡng
VI_NUTRIENT = {
    "Calories": "Calo",
    "Protein": "Protein",
    "Carbs": "Carb",
    "Fat": "Chất béo",
    "Saturates": "Bão hòa",
    "Sugar": "Đường",
    "Salt": "Muối",
}

VI_LEVEL = {"Low": "Thấp", "Medium": "Vừa", "High": "Cao"}

UNIT = {"Calories": "kcal"}


def img_to_base64(img_path) -> str:
    with open(img_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _compact(html: str) -> str:
    """Gộp toàn bộ whitespace thành một dòng.

    Markdown của Streamlit kết thúc HTML block ở dòng trắng và biến dòng thụt
    đầu dòng >= 4 dấu cách thành code block, nên mọi HTML fragment phải là
    một dòng liền (không chứa <style>/<script> nên an toàn).
    """
    return " ".join(html.split())


def inject_css():
    """Nạp toàn bộ design system (general-style.css) vào app."""
    st.markdown(f"<style>{CSS_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


# ── Top navbar ────────────────────────────────────────────────────────
def navbar():
    return _compact(f"""
    <div class="top-navbar">
        <a class="navbar-brand" href="#top-section" target="_self">
            <span class="brand-badge">🍜</span>
            <span class="brand-name">Food<b>Detector</b> AI</span>
        </a>
        <div class="navbar-right">
            <a class="navbar-pill" href="{GITHUB_URL}" target="_blank">
                <svg viewBox="0 0 16 16" aria-hidden="true"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"/></svg>
                <span class="pill-text">GitHub</span>
            </a>
        </div>
    </div>
    """)


# ── Hero banner ───────────────────────────────────────────────────────
def hero(banner="bg-about-cuisine.png", eyebrow="Nhận diện & phân tích dinh dưỡng bằng AI",
         title='Bữa ăn Việt của bạn, <span class="grad">đo lường chính xác</span>',
         subtitle="Quét ảnh món ăn để nhận diện hơn 67 món Việt Nam — ước lượng khối lượng theo thể tích 3D và tư vấn dinh dưỡng cá nhân hóa theo thể trạng của bạn.",
         chips=None):
    if chips is None:
        chips = [
            ("YOLOv26", "mô hình nhận diện"),
            ("VietFood67", "món ăn Việt"),
            ("SAM2 + Depth", "ước lượng khẩu phần"),
            ("mAP50 · 0.95", "độ chính xác"),
        ]
    img_b64 = img_to_base64(IMG_DIR / banner)
    chips_html = "".join(
        f'<span class="hero-chip"><b>{v}</b> {label}</span>' for v, label in chips
    )
    return _compact(f"""
    <div class="hero">
        <img class="hero-img" src="data:image/png;base64,{img_b64}">
        <div class="hero-overlay">
            <span class="hero-eyebrow">✦ {eyebrow}</span>
            <div class="hero-title">{title}</div>
            <div class="hero-subtitle">{subtitle}</div>
            <div class="hero-chips">{chips_html}</div>
        </div>
    </div>
    """)


def section_header(title, sub=""):
    sub_html = f'<div class="section-sub">{sub}</div>' if sub else ""
    return f'<div class="section-title">{title}</div>{sub_html}'


def metric_chips(items):
    """Hàng chip thống kê ngắn: [(value, label), ...]"""
    chips = "".join(
        f'<span class="metric-chip"><b>{v}</b> {label}</span>' for v, label in items
    )
    return f'<div class="metric-chip-row">{chips}</div>'


def divider_dot(label):
    return f'<div class="divider-dot">{label}</div>'


# ── KPI cards ─────────────────────────────────────────────────────────
def kpi_card(label, value, unit="", sub="", icon="", accent="amber", std=""):
    std_html = f'<span class="kpi-std"> ± {std}</span>' if std else ""
    unit_html = f'<span class="kpi-unit">{unit}</span>' if unit else ""
    sub_html = f'<div class="kpi-sub">{sub}</div>' if sub else ""
    return _compact(f"""
    <div class="kpi-card kpi-{accent}">
        <span class="kpi-icon">{icon}</span>
        <div class="kpi-value">{value}{std_html}{unit_html}</div>
        <div class="kpi-label">{label}</div>
        {sub_html}
    </div>
    """)


def kpi_row(cards):
    return f'<div class="kpi-row">{"".join(cards)}</div>'


# ── BMI ───────────────────────────────────────────────────────────────
def bmi_classify(bmi):
    """(nhãn tiếng Việt, màu hex) theo ngưỡng BMI chuẩn."""
    if bmi < 18.5:
        return "Thiếu cân", "#60A5FA"
    if bmi < 25:
        return "Bình thường", "#4ADE80"
    if bmi < 30:
        return "Thừa cân", "#FBBF24"
    return "Béo phì", "#F87171"


def bmi_gauge(bmi):
    """Thước đo BMI có kim chỉ, vùng 14–40 chia theo ngưỡng WHO."""
    pos = (min(max(bmi, 14.0), 40.0) - 14.0) / 26.0 * 100
    label, color = bmi_classify(bmi)
    return _compact(f"""
    <div class="bmi-gauge-wrap">
        <div class="bmi-gauge-head">
            <span class="bmi-gauge-value">{bmi:.1f} <span class="u">kg/m²</span></span>
            <span class="bmi-class-chip" style="--chip-c: {color};">{label}</span>
        </div>
        <div class="bmi-track" style="position: relative;">
            <div class="bmi-zone z1"></div><div class="bmi-zone z2"></div>
            <div class="bmi-zone z3"></div><div class="bmi-zone z4"></div>
            <div class="bmi-pin" style="--pin-pos: {pos:.1f}%;"></div>
        </div>
        <div class="bmi-scale"><span>14</span><span>18.5</span><span>25</span><span>30</span><span>40</span></div>
        <div class="bmi-legend">
            <span class="bmi-legend-item"><span class="bmi-dot" style="background:#60A5FA;"></span>Thiếu cân (&lt;18.5)</span>
            <span class="bmi-legend-item"><span class="bmi-dot" style="background:#4ADE80;"></span>Bình thường (18.5–24.9)</span>
            <span class="bmi-legend-item"><span class="bmi-dot" style="background:#FBBF24;"></span>Thừa cân (25–29.9)</span>
            <span class="bmi-legend-item"><span class="bmi-dot" style="background:#F87171;"></span>Béo phì (≥30)</span>
        </div>
    </div>
    """)


# ── Progress / RDA ────────────────────────────────────────────────────
def progress_row(label, value, target, unit, kind="goal"):
    """Thanh tiến độ. kind='goal': gần 100% là tốt; kind='limit': vượt là xấu."""
    pct = 0 if not target else value / target * 100
    pct = max(0.0, pct)
    width = min(pct, 100.0)
    if kind == "limit":
        if pct <= 70:
            color = "#4ADE80"
        elif pct <= 100:
            color = "#FBBF24"
        else:
            color = "#F87171"
    else:
        if pct < 50:
            color = "#60A5FA"
        elif pct <= 110:
            color = "#4ADE80"
        else:
            color = "#FBBF24"
    return _compact(f"""
    <div class="rda-item">
        <div class="rda-head">
            <span class="rda-name">{label}</span>
            <span class="rda-val"><b>{value:.1f}</b> / {target:.0f} {unit} · {pct:.0f}%</span>
        </div>
        <div class="rda-bar"><div class="rda-fill" style="--fill-c: {color}; width: {width:.1f}%;"></div></div>
    </div>
    """)


def rda_panel(total_nutrition, rda):
    """So sánh bữa ăn vừa quét với nhu cầu khuyến nghị hằng ngày."""
    rows = []
    mapping = [
        ("Calories", "Calo", "goal", "kcal"),
        ("Protein", "Protein", "goal", "g"),
        ("Fat", "Chất béo", "limit", "g"),
        ("Sugar", "Đường", "limit", "g"),
        ("Salt", "Muối", "limit", "g"),
    ]
    for key, label, kind, unit in mapping:
        target = rda.get(key, 0) if rda else 0
        if not target:
            continue
        rows.append(progress_row(label, float(total_nutrition.get(key, 0) or 0),
                                 float(target), unit, kind))
    note = ""
    if rows:
        note = ('<div class="rda-note">💡 Thanh tiến độ so với <b>mục tiêu hằng ngày</b> của bạn '
                '(tính từ BMI/TDEE ở tab Thể trạng). Xanh: an toàn · Vàng: cần lưu ý · Đỏ: đã vượt mức.</div>')
    return f'<div class="rda-panel">{"".join(rows)}{note}</div>'


def macro_target_panel(rda, calories_target):
    """Phân bổ macro khuyến nghị theo % năng lượng (protein 20% · béo 25% · carb 55%)."""
    rows = []
    for label, kcal_pct, color, icon in [
        ("Carb (chính)", 0.55, "#FEC51C", "🍚"),
        ("Chất béo", 0.25, "#FF8A3D", "🥑"),
        ("Protein", 0.20, "#60A5FA", "🍗"),
    ]:
        kcal = calories_target * kcal_pct
        grams = kcal / (4 if kcal_pct != 0.25 else 9)
        pct = kcal_pct * 100
        rows.append(_compact(f"""
        <div class="rda-item">
            <div class="rda-head">
                <span class="rda-name">{icon} {label}</span>
                <span class="rda-val"><b>{grams:.0f} g</b>/ngày · {pct:.0f}% năng lượng · {kcal:.0f} kcal</span>
            </div>
            <div class="rda-bar"><div class="rda-fill" style="--fill-c: {color}; width: {pct:.0f}%;"></div></div>
        </div>
        """))
    note = ('<div class="rda-note">💡 Phân bổ macro theo khuyến nghị cho mục tiêu của bạn, '
            'dựa trên tổng năng lượng mục tiêu ở trên.</div>')
    return f'<div class="rda-panel">{"".join(rows)}{note}</div>'


# ── Biểu đồ ───────────────────────────────────────────────────────────
def _dark_layout(fig, height=300, legend=False):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Be Vietnam Pro, Segoe UI, sans-serif", color="#B3A795", size=12),
        margin=dict(l=10, r=10, t=28 if legend else 10, b=10),
        height=height,
        showlegend=legend,
    )
    if legend:
        fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                      xanchor="center", x=0.5,
                                      font=dict(color="#F5EFE6", size=11)))
    return fig


def donut_macro_fig(protein, fat, carbs, calories):
    """Biểu đồ tròn tỷ lệ macro của bữa ăn, hiển thị tổng calo ở giữa."""
    values = [protein or 0, fat or 0, carbs or 0]
    if sum(values) <= 0:
        values = [1, 1, 1]
    fig = go.Figure(go.Pie(
        labels=["Protein", "Chất béo", "Carb"],
        values=values,
        hole=0.66,
        marker=dict(colors=["#60A5FA", "#FF8A3D", "#FEC51C"],
                    line=dict(color="#1E1913", width=2)),
        textinfo="percent",
        textfont=dict(color="#14110C", size=13),
        hovertemplate="%{label}: %{value:.1f} g<extra></extra>",
        sort=False,
    ))
    _dark_layout(fig, height=280)
    fig.add_annotation(
        text=f"<b>{calories:.0f}</b><br><span style='font-size:11px;color:#B3A795'>kcal</span>",
        showarrow=False, font=dict(color="#F5EFE6", size=20), font_family="Be Vietnam Pro",
    )
    return fig


def weight_projection_fig(weight, height, goal, weeks=12):
    """Dự báo đường cân nặng 12 tuần nếu duy trì thặng hụt/thặng dư mục tiêu."""
    healthy_max = 24.9 * (height / 100) ** 2
    healthy_mid = 22.0 * (height / 100) ** 2
    weekly = {"lose": -0.45, "gain": +0.35, "maintain": 0.0}.get(goal, 0.0)

    x = list(range(weeks + 1))
    y = [weight + weekly * w for w in x]
    # Giảm cân dừng ở mức cân nặng khỏe mạnh, không giảm vô hạn
    if weekly < 0:
        y = [max(v, healthy_mid) for v in y]
    elif weekly > 0:
        y = [min(v, healthy_max) for v in y]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=[weight] * len(x), mode="lines", name="Cân nặng hiện tại",
        line=dict(color="#7D7264", width=1.5, dash="dot"),
        hovertemplate="Hiện tại: %{y:.1f} kg<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="lines+markers", name="Dự báo",
        line=dict(color="#FEC51C", width=3),
        marker=dict(size=6, color="#FF8A3D"),
        hovertemplate="Tuần %{x}: %{y:.1f} kg<extra></extra>",
    ))
    band_min, band_max = min(weight, healthy_mid), max(weight, healthy_mid)
    fig.add_hrect(y0=band_min, y1=band_max, fillcolor="#4ADE80", opacity=0.10,
                  line_width=0, annotation_text="Vùng cân nặng khỏe mạnh",
                  annotation_font=dict(color="#4ADE80", size=11))
    goal_label = {"lose": "giảm cân", "gain": "tăng cân", "maintain": "giữ cân"}.get(goal, "")
    fig.update_yaxes(title_text="Cân nặng (kg)", gridcolor="rgba(255,255,255,0.06)")
    fig.update_xaxes(title_text=f"Tuần (mục tiêu: {goal_label})",
                     gridcolor="rgba(255,255,255,0.06)", tick0=0, dtick=2)
    _dark_layout(fig, height=300, legend=True)
    return fig


# ── Traffic-light & dish cards ────────────────────────────────────────
def traffic_chip(nutrient, value_str, level_desc, color):
    """Chip màu giao thông cho một chất dinh dưỡng."""
    name = VI_NUTRIENT.get(nutrient, nutrient)
    level = VI_LEVEL.get(level_desc, level_desc)
    red_cls = " t-red" if color.upper() in ("#FB4D44", "#FF4A3F") else ""
    return _compact(f"""
    <div class="traffic-chip{red_cls}" style="background: {color};">
        <span class="t-name">{name}</span>
        <span class="t-val">{value_str}</span>
        <span class="t-lvl">{level}</span>
    </div>
    """)


def dish_card(name, conf, serving, thumb_html, chips_html, extra_html=""):
    """Card một món ăn: ảnh crop + tên + chip dinh dưỡng traffic-light."""
    conf_html = f'<span class="dish-conf">{conf}%</span>' if conf else ""
    return _compact(f"""
    <div class="dish-card">
        <div class="dish-head">
            {thumb_html}
            <div>
                <p class="dish-name">{name} {conf_html}</p>
                <span class="dish-serving">{serving}</span>
                {extra_html}
            </div>
        </div>
        <div class="dish-chips">{chips_html}</div>
    </div>
    """)


def thumb_img(b64_jpeg, cls="dish-thumb"):
    return f'<img class="{cls}" src="data:image/jpeg;base64,{b64_jpeg}">'


def vi_serving(serving):
    """Dịch chuỗi khẩu phần (từ class_names / volume pipeline) sang tiếng Việt."""
    if not serving:
        return ""
    s = str(serving)
    s = s.replace("reference serving", "khẩu phần tham chiếu")
    s = s.replace("estimated portion", "khẩu phần ước lượng")
    s = s.replace("per 100g", "mỗi 100 g")
    return s


def portion_badge_html(volume_cm3, mass_g, mass_std_g, confidence_level):
    return (f"<span class='portion-badge portion-{confidence_level}'>"
            f"{volume_cm3:.0f} cm³ · {mass_g:.0f} ± {mass_std_g:.0f} g · "
            f"độ tin cậy {confidence_level}</span>")


def volume_note_html(scale_source=None, error=None):
    if scale_source:
        return (f"<div class='volume-status-note'>⚖️ <b>Ước lượng theo khẩu phần</b> "
                f"(nguồn tỉ lệ: <b>{scale_source}</b>) — giá trị dinh dưỡng được tính theo "
                f"thể tích món ăn thực tế trong ảnh của bạn.</div>")
    note = ("<div class='volume-status-note'>ℹ️ Không khả dụng ước lượng thể tích — "
            "hiển thị dinh dưỡng theo <b>khẩu phần tham chiếu</b> chuẩn của từng món.")
    if error:
        short = error.strip().splitlines()[-1][:180] if error.strip() else ""
        note += f"<br><span style='font-size:0.75rem'>Chi tiết lỗi: {short}</span>"
    return note + "</div>"


def no_food_alert():
    return _compact("""
    <div class="no-food-alert">
        <h5>🍽️ Không phát hiện món ăn nào</h5>
        <p>Mô hình không nhận diện được món ăn nào trong ảnh. Hãy thử ảnh chụp gần và đủ sáng hơn,
        hoặc giảm ngưỡng tin cậy rồi thử lại.</p>
    </div>
    """)


def traffic_legend(expander):
    """Nội dung expander giải thích hệ thống màu giao thông."""
    img_b64 = img_to_base64(IMG_DIR / "nutrition-table.png")
    expander.markdown(f"""
<img class="legend-img" src="data:image/png;base64,{img_b64}">
<ul class="legend-list">
    <li><strong class="legend-color-chip" style="background: var(--green-nu);">Xanh (Thấp)</strong> — rất lành mạnh, thoải mái thưởng thức.</li>
    <li><strong class="legend-color-chip" style="background: var(--yellow-nu);">Vàng (Vừa)</strong> — ăn điều độ hoặc kết hợp với lựa chọn lành mạnh hơn.</li>
    <li><strong class="legend-color-chip" style="background: var(--red-nu);">Đỏ (Cao)</strong> — nên hạn chế và tìm lựa chọn thay thế lành mạnh hơn.</li>
</ul>
<p>Giá trị dinh dưỡng chỉ mang tính tham khảo. Xem thêm: <a href="https://www.nutricalc.co.uk/case-study/case-study-uk-traffic-light-front-of-pack-colour-thresholds/">NutriCalc</a>,
<a href="https://heas.health.vic.gov.au/resources/government-guidelines/traffic-light-system/">Healthy Eating Advisory Service</a></p>
""", unsafe_allow_html=True)
