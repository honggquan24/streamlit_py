import streamlit as st

# About page configuration
st.set_page_config(
    page_title="About",
    layout="wide"
)

def show_about():
    st.title("About Me")
    st.markdown(
        """
        <div style="text-align: justify; line-height: 1.6;">

        I am a third-year student at the **University of Technology and Education (HCMUTE)**, majoring in **Robotics and Artificial Intelligence**. I enjoy working on **AI software development** and I am eager to keep learning and improving. 
        
        I like exploring new technologies, going into the details, and understanding how things work. With a solid base in **machine learning** and **deep learning**, I am passionate about using these skills to solve real-world problems.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")  # Horizontal separator
