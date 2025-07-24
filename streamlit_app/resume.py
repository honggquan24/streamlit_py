import streamlit as st
import base64

# resume page configuration
st.set_page_config(
    page_title="Resume",
    layout="wide"
)

def show_resume():
    st.title("Resume")
    st.write("You can download my resume from the button below:")

    resume_path = "cv/CV_VoHongQuan.pdf"
     
    with open("cv/CV_VoHongQuan.pdf", "rb") as f:
        pdf_bytes = f.read()
        st.download_button(
            key= "download_cv",
            label="Download Resume",
            data=pdf_bytes,
            file_name="CV_VoHongQuan.pdf",
            mime="application/pdf"
        )

    st.write("---")

 