import streamlit as st
import functools
import projects.rotary_pendulum_sac_simscape.show_rotary_pendulum_sac_simscape as rotary_project
import projects.argi_crop_classification.show_argi_crop_classification as argi_project
import projects.house_price_prediction.show_house_price_prediction as house_price_project
import projects.mlp_from_scratch.show_mlp_from_scratch as mlp_project
import projects.cnn_from_scratch.show_cnn_from_scratch as cnn_project
# Projects page configuration
st.set_page_config(
    page_title="Projects",
    layout="wide"
)

def show_projects():
    st.title("Projects")
    st.markdown("---")
    rotary_project.show_project()  # Show the rotary pendulum project

    st.markdown("---")
    argi_project.show_project()  # Show the argi crop classification project

    st.markdown("---")
    house_price_project.show_project()  # Show the house price prediction project

    st.markdown("---")
    mlp_project.show_project()  # Show the MLP from scratch project

    st.markdown("---")
    cnn_project.show_project()

