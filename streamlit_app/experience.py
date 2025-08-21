import streamlit as st

# Experience page configuration
st.set_page_config(
    page_title="Experience",
    layout="wide"
)

def show_experience():
    st.title("Experience")

    # First experience
    cols1 = st.columns([3, 1])
    with cols1[0]:
        st.write("**Research Assistant – Robotics Lab, HCMUTE**")
    with cols1[1]:
        st.write("_1/2025 – Present_")

    st.write("- Supported reinforcement learning model training and real-time testing for control systems.")
    st.write("- Assisted in designing reward functions and tuning hyperparameters for reinforcement learning agents.")

    # Second experience
    cols2 = st.columns([3, 1])
    with cols2[0]:
        st.write("**Self Employed**")
    with cols2[1]:
        st.write("_Jun 2023 – Present_")

    st.write("- Code and build something every day: from embedded AI, CV applications to robotics control systems.")
    st.write("- Explore modern AI tools and apply to real-world mini-projects.")

    # st.markdown("---")  # Horizontal separator
