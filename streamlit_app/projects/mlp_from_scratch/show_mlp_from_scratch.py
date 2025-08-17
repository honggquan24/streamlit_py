import streamlit as st
from pathlib import Path

def show_project():
    with st.container():
        # ===== Header (title + time) =====
        cols = st.columns([3, 1])
        with cols[0]:
            st.subheader("Multilayer Perceptron (MLP) from Scratch")
        with cols[1]:
            st.caption("*11/2024 – 12/2024*")

        st.markdown(
            "- Developed a **modular neural network framework** entirely in **NumPy**, covering layers "
            "(*Dense, BatchNorm, Dropout*), activations, loss functions, optimizers, and data pipeline.\n"
            "- Implemented key optimizers (**SGD, Momentum, RMSProp, Adagrad, Adam**) with full state "
            "management and checkpointing.\n"
            "- Supported both **forward/backward propagation** and verified gradients via **gradient checking**.\n"
            "- Designed flexible utilities for **saving/loading models**, enabling reproducible experiments.\n"
            "- Built to provide **transparent, educational insight** into deep learning fundamentals without "
            "high-level libraries."
        )

        col1, col2 = st.columns([3, 7])
        with col1:
            c11, c12 = st.columns(2)
            with c11:
                st.link_button("GitHub", "https://github.com/honggquan24/MLP-from-scratch")
                
            with c12:
                # https://vhq-portfolio.streamlit.app/ or http://localhost:8501/
                try:
                    st.link_button("Demo", "https://localhost:8501/")
                except:
                    st.link_button("Demo", "https://vhq-portfolio.streamlit.app/")

