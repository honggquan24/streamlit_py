import streamlit as st

def show_project():
    with st.container():
        cols = st.columns([3, 1])
        with cols[0]:
            st.subheader("Agri Crop Classification")
        with cols[1]:
            st.caption("*1/2024 – 3/2024*")
        
        st.markdown(
            "- Developed a crop classification model using **TensorFlow** and **Keras**.\n"
            "- Utilized **Convolutional Neural Networks (CNNs)** for image classification tasks.\n"
            "- Achieved high accuracy in identifying different crop types from satellite imagery.\n"
            "- Deployed the model as a web application using **Streamlit** for easy access."
        )
        col1, col2 = st.columns([3, 7])  # tạo layout 3 cột: trái - giữa - phải

        with col1:
            col11, col12 = st.columns(2)  # chia đôi phần giữa

            with col11:
                st.link_button("GitHub", "https://github.com/honggquan24/agri-crop-classification")
            with col12:
                st.link_button("Demo", "https://your-streamlit-demo-url")
