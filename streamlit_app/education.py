import streamlit as st

# Education page configuration
st.set_page_config(
    page_title="Education",
    layout="wide"
)

def show_education():
    st.title("Education")
    # Dòng 1: Trường và thời gian
    col1, col2 = st.columns([6, 4])
    with col1:
        st.markdown("**HCMC University of Technology and Education (HCMUTE)**")
    with col2:
        st.markdown("*8/2022 – Expected 8/2026*")

    # Dòng 2: Ngành học
    st.markdown("*Bachelor of Engineering – Robotics and Artificial Intelligence*")

    # Dòng 3: GPA
    st.markdown("**GPA:** 3.11 / 4.0")

    # Dòng 4: Môn học liên quan
    st.markdown("**Relevant Coursework:** Machine Vision, Artificial Intelligence," \
                " Artificial Neural Network, Practice of Artificial Intelligence")
    
    st.markdown("---")  # Horizontal separator

