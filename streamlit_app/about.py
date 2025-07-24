import streamlit as st

# About page configuration
st.set_page_config(
    page_title="About",
    layout="wide"
)

def show_about():
    st.title("About Me")
    st.write(
    "I'm a third-year student at the University of Technology and Education (HCM UTE), "
    "majoring in Robotics and Artificial Intelligence. "
    "My favorite work is AI software development and I am eager to learn and grow in this field. "
    "When exploring new technologies, I enjoy diving deep into the details and understanding how they work. "
    "With a strong foundation in machine learning and deep learning, " \
    "I am passionate about applying these skills to solve real-world problems. "
    )

    st.markdown("---")  # Horizontal separator
