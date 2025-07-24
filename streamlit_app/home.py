import streamlit as st
import about
import experience
import education
import projects
import resume

# Main page configuration
st.set_page_config(
    page_title="Home",  # Set the page title
    layout="wide"  # Set the layout to wide
)

def show_home():
    about.show_about()  # Show the about page content
    education.show_education()  # Show the education page content    
    experience.show_experience()  # Show the experience page content
