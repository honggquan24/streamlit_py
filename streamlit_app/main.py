import streamlit as st
import about
import experience
import education
import projects_page
import home 
import resume


st.set_page_config(
    page_title="Vo Hong Quan's Portfolio",  # Set the page title
    layout="wide"  # Set the layout to wide) 
)       

# Load Font Awesome (chèn đầu trang để đảm bảo được nhận)
# Font Awesome là thư viện cung cấp hàng nghìn icon vector (GitHub, LinkedIn, Instagram, v.v.).
# st.markdown(..., unsafe_allow_html=True) cho phép Streamlit nhúng trực tiếp HTML
#  (vì mặc định Streamlit chặn HTML để bảo mật).
# Dòng <link rel="stylesheet" ...> là cách nhúng file CSS chứa định nghĩa các icon.
# Bọc trong <head>...</head> là không bắt buộc, nhưng giúp đúng cấu trúc HTML chuẩn.

st.markdown("""
    <head>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    </head>
    """,
    unsafe_allow_html=True
)

cols = st.columns([3, 1, 6])  # Left 30%, Mid 10% and right 60% of the screen
with cols[0]:
    st.title("Vo Hong Quan")
    st.subheader("AI Engineer")
    st.write("This is my personal portfolio website, which showcases my skills and projects in AI engineering." \
    " I hope you can find it useful and inspiring.")
    st.markdown("<br><br>", unsafe_allow_html=True)

    # Contact info (optional: use button or text input)
    st.markdown("##### Contact Me")
    email = st.write("Email: 22134012@student.hcmute.edu.vn")
    phone = st.write("Phone: ", "0363645485")

    # Social icons
    # <div style=...>	Tạo một khối ngang (flex) để chứa icon.
    # display: flex	Sắp các phần tử con nằm ngang (thay vì dọc).
    # justify-content: left	Căn lề trái, có thể đổi sang center, space-between ...
    # gap: 2rem	Khoảng cách giữa các icon.
    # padding: 1.5rem 0	Khoảng cách top-bottom là 1.5 rem.
    # <a href=...>	Gắn link cho từng icon, mở tab mới (target="_blank").
    # <i class="fab fa-github fa-2x">	Dùng icon GitHub (Font Awesome Brands), kích thước gấp đôi.
    # style="color:#8892b0;"	Màu icon hơi xám xanh, giống UI hiện đại.
    # fab = Font Awesome Brands → dùng cho các logo (GitHub, Instagram…)
    # fa-github, fa-linkedin, v.v. = tên riêng từng icon.
    # fa-2x, fa-3x, ... = tăng kích thước icon.
    # st.markdown(..., unsafe_allow_html=True) = kích hoạt HTML thuần bên trong Streamlit.
    st.markdown(
        """
        <div style="display:flex; justify-content:left; gap:2rem; padding:1.5rem 0;">
            <a href="https://github.com/honggquan24" target="_blank">
                <i class="fab fa-github fa-2x" style="color:#8892b0;"></i>
            </a>
            <a href="https://www.linkedin.com/in/vo-hong-quan-b50063373/" target="_blank">
                <i class="fab fa-linkedin fa-2x" style="color:#8892b0;"></i>
            </a>
            <a href="https://www.kaggle.com/honggquan" target="_blank">
                <i class="fab fa-kaggle fa-2x" style="color:#8892b0;"></i>
            </a>
        </div>
        """,
        unsafe_allow_html=True
    )





with cols[2]:
    
    tabs = st.tabs(["Home", "About", "Education",  "Experience", "Projects", "Resume"])

    with tabs[0]:
        home.show_home()  # Call the home page function
        
    with tabs[1]:
        about.show_about()

    with tabs[2]:
        education.show_education()

    with tabs[3]:
        experience.show_experience()

    with tabs[4]:
        projects_page.show_projects()

    with tabs[5]:
        resume.show_resume()
