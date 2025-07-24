import streamlit as st

def show_project():
    with st.container():
        cols = st.columns([3, 1])
        with cols[0]:
            st.subheader("RL-Based Rotary Inverted Pendulum Controller")
        with cols[1]:
            st.caption("*2/2025 – 6/2025*")

        st.markdown(
            "- Built a reinforcement learning environment using **Simscape** model of the rotary inverted pendulum.\n"
            "- Implemented **Soft Actor-Critic (SAC)** with MATLAB RL Toolbox for continuous control.\n"
            "- Designed & tuned reward functions:\n"
            "   - **Swing-up Agent**: reach upright in ~2s.\n"
            "   - **Balance Agent**: maintain stable position.\n"
            "- Deployed final trained policy to **ESP32** for real-time execution."
        )

        col1, col2 = st.columns([3, 7])  # tạo layout 3 cột: trái - giữa - phải

        with col1:
            col11, col12 = st.columns(2)  # chia đôi phần giữa

            with col11:
                st.link_button("GitHub", "https://github.com/honggquan24/rotary-pendulum-sac-simscape")
            # with col12:
            #     st.link_button("Demo", "https://your-streamlit-demo-url")
