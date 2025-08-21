import streamlit as st
from pathlib import Path


def show_project():
    with st.container():
        # Header: tiêu đề & mốc thời gian
        c1, c2 = st.columns([3, 1])
        with c1:
            st.subheader("Convolutional Neural Network (CNN) from Scratch")
        with c2:
            st.caption("9/2024 – 10/2024")

        # Mô tả dự án (Markdown chuẩn, rõ ràng)
        st.markdown(
            """
            - Designed and implemented a **pure NumPy CNN** featuring Conv2D, Pooling (Max/Avg), and Fully Connected layers.
            - Engineered the core operations—**forward propagation, backward propagation, and gradient computation**—manually without auto-differentiation.
            - Trained on **MNIST**, **Fashion MNIST** and **CIFAR-10**, achieving competitive accuracy and validating the low-level implementation.
            - Demonstrated deep mastery of CNN fundamentals, backpropagation through spatial dimensions, and optimization.
            """
        )

        # Liên kết
        left, right = st.columns([3, 7])
        with left:
            b1, b2 = st.columns(2)
            with b1:
                st.link_button("GitHub", "http://localhost:8501/CNN")
            with b2:
                try:
                    st.link_button("Demo", "http://localhost:8501/CNN")
                except:
                    # Dùng link bản deploy; nếu chạy local, thay bằng: http://localhost:8501/CNN
                    st.link_button("Demo", "https://vhq-portfolio.streamlit.app/CNN")
