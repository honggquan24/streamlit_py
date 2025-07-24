import streamlit as st
import functools
import projects.rotary_pendulum_sac_simscape.show_rotary_pendulum_sac_simscape as rotary_project
import projects.argi_crop_classification.show_argi_crop_classification as argi_project

# Projects page configuration
st.set_page_config(
    page_title="Projects",
    layout="wide"
)

# Dữ liệu dự án


def decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        st.markdown("---")
        func(*args, **kwargs)
    return wrapper

@decorator
def rotary_project_show_project():
    """Show the rotary pendulum project."""
    rotary_project.show_project()

@decorator
def argi_project_show_project():
    """Show the argi crop classification project."""
    argi_project.show_project()

def show_projects():
    st.title("Projects")

    rotary_project_show_project()  # Show the rotary pendulum project

    argi_project_show_project()  # Show the argi crop classification project


