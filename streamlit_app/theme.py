from textwrap import dedent

"""
- Chứa cấu hình màu + CSS cho toàn bộ app.
- Sửa màu ở dict THEME là đủ (1 bộ màu duy nhất cho tất cả file).
"""


# 1) CẤU HÌNH MÀU (THEME)
# THEME là nguồn sự thật duy nhất cho toàn bộ màu sắc và tham số kính (glassmorphism).
# Bạn chỉ cần chỉnh các giá trị hex hoặc thông số trong dictionary này,
# toàn bộ CSS sinh ra bên dưới sẽ tự động dùng các giá trị mới.
# - bg1, bg2, bg3: 3 lớp màu nền (dùng tạo gradient nền app)
# - text:          màu chữ chính
# - muted:         màu chữ phụ/diễn giải
# - accent:        màu nhấn 1 (điểm nhấn chính)
# - accent2:       màu nhấn 2 (điểm nhấn phụ)
# - gold:          màu nhấn bổ trợ (vàng)
# - panel_alpha:   độ trong suốt (alpha) của khung/card (0.0 -> trong suốt, 1.0 -> đục)
# - blur_px:       bán kính blur (hiệu ứng kính) cho các panel/card (đơn vị px)
THEME = {
    "bg1":    "#0f0f23",
    "bg2":    "#1a1a2e",
    "bg3":    "#16213e",
    "text":   "#E6F1FF",
    "muted":  "#8892b0",
    "accent": "#64ffda",
    "accent2":"#00d4ff",
    "gold":   "#ffd700",
    "panel_alpha": "0.05",  # độ trong suốt card
    "blur_px": "10px",      # độ mờ kính
}


# 2) HÀM SINH CSS TỪ THEME
# get_css(theme):
#   - Input: 1 dictionary theme (mặc định dùng THEME ở trên).
#   - Xử lý: đưa toàn bộ khóa/giá trị của theme vào :root để tạo CSS variables,
#            sau đó trả về một chuỗi CSS hoàn chỉnh (kèm fonts & icons import).
#   - Output: chuỗi HTML+CSS (dạng string) để bạn inject vào ứng dụng (VD: st.markdown(..., unsafe_allow_html=True)).
def get_css(theme: dict = THEME) -> str:
    # Đặt alias ngắn gọn để đọc dễ hơn; không đổi dữ liệu đầu vào.
    t = theme

    # dedent(): loại bỏ thụt lề thừa trong chuỗi nhiều dòng để CSS gọn gàng.
    return dedent(f"""
    <head>
        <link rel="stylesheet"
              href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap"
              rel="stylesheet">
    </head>

    <style>
    :root {{
        --bg1: {t["bg1"]};
        --bg2: {t["bg2"]};
        --bg3: {t["bg3"]};
        --text: {t["text"]};
        --muted: {t["muted"]};
        --accent: {t["accent"]};
        --accent2: {t["accent2"]};
        --gold: {t["gold"]};
        --panel-alpha: {t["panel_alpha"]};
        --blur: {t["blur_px"]};
    }}

    /* Global Styles */
    .stApp {{
        background: linear-gradient(135deg, var(--bg1) 0%, var(--bg2) 50%, var(--bg3) 100%);
        font-family: 'Inter', sans-serif;
        color: var(--text);
    }}

    /* Animated background particles */
    .stApp::before {{
        content: '';
        position: fixed;
        top: 0; left: 0; width: 100%; height: 100%;
        background-image:
            radial-gradient(2px 2px at 20px 30px, var(--accent), transparent),
            radial-gradient(2px 2px at 40px 70px, var(--accent2), transparent),
            radial-gradient(1px 1px at 90px 40px, var(--gold), transparent),
            radial-gradient(1px 1px at 130px 80px, #ff6b6b, transparent);
        animation: sparkle 20s linear infinite;
        pointer-events: none;
        z-index: 0;
    }}
    @keyframes sparkle {{
        0% {{ transform: translateY(0px) rotate(0deg); opacity: 0.7; }}
        50% {{ transform: translateY(-100px) rotate(180deg); opacity: 1; }}
        100% {{ transform: translateY(-200px) rotate(360deg); opacity: 0.7; }}
    }}

    /* Main content styling */
    .main-content {{
        position: relative; z-index: 1;
        backdrop-filter: blur(var(--blur));
        background: rgba(255, 255, 255, var(--panel-alpha));
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 2rem; margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }}

    /* Profile section styling */
    .profile-section {{
        background: linear-gradient(145deg, rgba(255,255,255,0.1), rgba(255,255,255,0.05));
        border-radius: 20px; padding: 2rem;
        border: 1px solid rgba(255,255,255,0.15);
        box-shadow: 0 8px 32px rgba(0,0,0,0.2);
        backdrop-filter: blur(var(--blur));
        position: relative; overflow: hidden;
    }}
    .profile-section::before {{
        content: '';
        position: absolute; top: -50%; left: -50%;
        width: 200%; height: 200%;
        background: conic-gradient(transparent, color-mix(in oklab, var(--accent) 30%, transparent), transparent);
        animation: rotate 10s linear infinite; z-index: -1;
    }}
    @keyframes rotate {{ 0% {{transform: rotate(0deg);}} 100% {{transform: rotate(360deg);}} }}

    /* Typography improvements */
    h1, h2, h3, h4, h5, h6 {{
        background: linear-gradient(135deg, var(--accent), var(--accent2), var(--gold));
        -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent;
        font-weight: 700;
        text-shadow: 0 0 30px color-mix(in oklab, var(--accent) 35%, transparent);
    }}
    .main-title {{ font-size: 3.5rem !important; margin-bottom: .5rem !important; animation: glow 2s ease-in-out infinite alternate; }}
    .subtitle   {{ font-size: 1.8rem !important; color: var(--accent) !important; margin-bottom: 1rem !important; font-weight: 500 !important; }}
    @keyframes glow {{
        from {{ text-shadow: 0 0 20px color-mix(in oklab, var(--accent) 30%, transparent); }}
        to   {{ text-shadow: 0 0 30px color-mix(in oklab, var(--accent) 60%, transparent), 0 0 40px color-mix(in oklab, var(--accent) 30%, transparent); }}
    }}

    /* Description text styling */
    .description {{
        color: var(--muted);
        font-size: 1.1rem; line-height: 1.8; margin-bottom: 2rem;
        padding: 1rem; border-left: 3px solid var(--accent);
        background: color-mix(in oklab, var(--accent) 10%, transparent);
        border-radius: 0 10px 10px 0;
    }}

    /* Contact section styling */
    .contact-section {{
        background: color-mix(in oklab, var(--accent) 15%, transparent);
        border-radius: 15px; padding: 1.5rem; margin: 2rem 0;
        border: 1px solid color-mix(in oklab, var(--accent) 40%, transparent);
        box-shadow: 0 4px 20px color-mix(in oklab, var(--accent) 12%, transparent);
    }}
    .contact-item {{
        display: flex; align-items: center; margin: .8rem 0;
        color: var(--text); font-size: 1rem; transition: all .3s ease;
    }}
    .contact-item:hover {{ color: var(--accent); transform: translateX(10px); }}
    .contact-item i {{ margin-right: 10px; color: var(--accent); width: 20px; }}

    /* Social icons styling */
    .social-container {{
        background: rgba(255,255,255,0.05);
        border-radius: 15px; padding: 1.5rem; text-align: center;
        border: 1px solid rgba(255,255,255,0.1); margin-top: 2rem;
    }}
    .social-icons {{
        display: flex; justify-content: center; gap: 2rem; padding: 1rem 0;
    }}
    .social-icon {{
        display: inline-block; width: 60px; height: 60px; line-height: 60px; text-align: center;
        background: linear-gradient(145deg, color-mix(in oklab, var(--accent) 10%, transparent), color-mix(in oklab, var(--accent2) 10%, transparent));
        border-radius: 50%; border: 2px solid transparent; background-clip: padding-box;
        transition: all .4s ease; position: relative; overflow: hidden;
    }}
    .social-icon::before {{
        content: ''; position: absolute; inset: 0; border-radius: 50%; padding: 2px;
        background: linear-gradient(45deg, var(--accent), var(--accent2), var(--gold));
        -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
        -webkit-mask-composite: exclude; mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
        mask-composite: exclude; opacity: 0; transition: opacity .4s ease;
    }}
    .social-icon:hover::before {{ opacity: 1; }}
    .social-icon:hover {{ transform: translateY(-5px) scale(1.1); box-shadow: 0 10px 30px color-mix(in oklab, var(--accent) 30%, transparent); }}
    .social-icon i {{ color: var(--muted); font-size: 1.5rem; transition: all .3s ease; position: relative; z-index: 1; }}
    .social-icon:hover i {{ color: var(--accent); transform: scale(1.2); }}

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {{
        display: flex;
        flex-wrap: nowrap;            /* không bị xuống hàng, có thể kéo ngang */
        justify-content: space-between;
        gap: 10px;
        width: 100%;
        padding: 10px;
        border-radius: 16px;
        background: rgba(255,255,255,var(--panel-alpha));
        border: 1px solid rgba(255,255,255,0.10);
        backdrop-filter: blur(var(--blur));
        overflow-x: auto;             /* cho phép cuộn ngang khi thiếu chỗ */
        scrollbar-width: none;        /* ẩn scrollbar trên Firefox */
    }}

    .stTabs [data-baseweb="tab"] {{
        flex: 1 1 auto;               /* tab giãn đều theo chiều ngang */
        min-width: 110px;             /* không quá nhỏ */
        height: 44px;
        padding: 0 16px;
        border-radius: 12px;
        color: var(--muted);
        background: transparent;
        border: 1px solid transparent;
        transition: all .25s ease;
        font-weight: 600;
        letter-spacing: .2px;
    }}
    .stTabs [aria-selected="true"] {{
        background: linear-gradient(135deg, color-mix(in oklab, var(--accent) 20%, transparent), color-mix(in oklab, var(--accent2) 10%, transparent)) !important;
        color: var(--accent) !important;
        border: 1px solid color-mix(in oklab, var(--accent) 35%, transparent) !important;
        box-shadow: 0 4px 15px color-mix(in oklab, var(--accent) 20%, transparent);
    }}

    /* Decorative line */
    .decoration-line {{
        height: 2px;
        background: linear-gradient(90deg, transparent, var(--accent), var(--accent2), transparent);
        margin: 2rem 0; border-radius: 1px; animation: pulse-line 3s ease-in-out infinite;
    }}
    @keyframes pulse-line {{
        0%, 100% {{ opacity: .5; transform: scaleX(1); }}
        50% {{ opacity: 1; transform: scaleX(1.02); }}
    }}

    /* Floating elements */
    .floating-element {{
        position: absolute; border-radius: 50%; opacity: 0.1; animation: float 6s ease-in-out infinite;
    }}
    .floating-element:nth-child(1) {{ width: 80px; height: 80px; top: 10%; right: 10%; background: var(--accent); animation-delay: 0s; }}
    .floating-element:nth-child(2) {{ width: 60px; height: 60px; top: 60%; right: 20%; background: var(--accent2); animation-delay: 2s; }}
    .floating-element:nth-child(3) {{ width: 40px; height: 40px; top: 30%; right: 5%; background: var(--gold); animation-delay: 4s; }}
    @keyframes float {{
        0%, 100% {{ transform: translateY(0px) rotate(0deg); }}
        33%      {{ transform: translateY(-20px) rotate(120deg); }}
        66%      {{ transform: translateY(10px) rotate(240deg); }}
    }}

    /* Scrollbar */
    ::-webkit-scrollbar {{ width: 8px; }}
    ::-webkit-scrollbar-track {{ background: rgba(255,255,255,0.1); border-radius: 10px; }}
    ::-webkit-scrollbar-thumb {{ background: linear-gradient(135deg, var(--accent), var(--accent2)); border-radius: 10px; }}
    ::-webkit-scrollbar-thumb:hover {{ background: linear-gradient(135deg, var(--accent2), var(--accent)); }}
    </style>
    """).strip()
