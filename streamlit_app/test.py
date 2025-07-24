import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Vo Hong Quan's Portfolio",
    layout="wide"
)

# Initialize session state for navigation
if 'current_section' not in st.session_state:
    st.session_state.current_section = 'About'

# Theme toggle
mode = st.sidebar.radio("Theme mode", ["Light", "Dark"], index=0)

# Color scheme based on theme
if mode == "Dark":
    bg_color = "#0a192f"
    text_color = "#ccd6f6"
    accent_color = "#64ffda"
    nav_bg = "#112240"
else:
    bg_color = "#1577D9"
    text_color = "#212529"
    accent_color = "#0d6efd"
    nav_bg = "#ffffff"

# Inject custom CSS
st.markdown(f"""
    <style>
    body {{ background-color: {bg_color}; color: {text_color}; }}
    .sidebar .sidebar-content {{ background-color: {nav_bg}; }}
    .css-1lcbmhc {{ max-width: 1200px; margin: 0 auto; padding: 2rem 1rem; }}
    </style>
""", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to", ['About', 'Experience', 'Projects'], 
                           index=['About', 'Experience', 'Projects'].index(st.session_state.current_section))
st.session_state.current_section = section

# Hero section function
def show_hero():
    st.title("Vo Hong Quan")
    st.subheader("AI Engineer")

# About section
if st.session_state.current_section == 'About':
    show_hero()
    st.markdown("---")
    st.write(
        "I'm a developer passionate About crafting accessible, pixel-perfect user interfaces "
        "that blend thoughtful design with robust engineering. My favorite work lies at the intersection "
        "of design and development, creating Experiences that not only look great but are meticulously built for performance and usability."
    )
    st.write(
        "Currently, I'm a Senior Front-End Engineer at "
        f"**<span style='color:{accent_color}'>Klaviyo</span>**, specializing in accessibility. "
        "I contribute to the creation and maintenance of UI components that power Klaviyo's frontend, "
        "ensuring our platform meets web accessibility standards and best practices to deliver an inclusive user Experience.",
        unsafe_allow_html=True
    )
    st.write(
        "In the past, I've had the opportunity to develop software across a variety of settings — from "
        f"**advertising agencies** and **large corporations** to **start-ups** and **small digital product studios**. "
        "Additionally, I released a comprehensive video course a few years ago, guiding learners through building "
        "a web app with the Spotify API."
    )
    st.write(
        "In my spare time, I'm usually climbing, reading, hanging out with my wife and two cats, "
        "or running around Hyrule searching for Korok seeds."
    )

# Experience section
elif st.session_state.current_section == 'Experience':
    st.header("Experience")
    jobs = [
        {
            'title': 'Senior Frontend Engineer, Accessibility • Klaviyo',
            'period': '2024 — PRESENT',
            'description': (
                'Build and maintain critical components used to construct Klaviyo\'s frontend, across the whole product. '
                'Work closely with cross-functional teams, including developers, designers, and product managers, '
                'to implement and advocate for best practices in web accessibility.'
            ),
            'tech': ['JavaScript', 'TypeScript', 'React', 'Storybook']
        },
        {
            'title': 'Lead Engineer → Senior Engineer → Engineer • Upstatement',
            'period': '2018 — 2024',
            'description': (
                'Built, styled, and shipped high-quality websites, design systems, mobile apps, and digital Experiences for clients including Harvard Business School, Everytown for Gun Safety, Pratt Institute, and more. '
                'Provided leadership within the engineering department through close collaboration and knowledge sharing.'
            ),
            'tech': ['JavaScript', 'TypeScript', 'HTML & SCSS', 'React', 'Next.js', 'Node.js']
        }
    ]
    for job in jobs:
        st.subheader(job['title'])
        st.caption(job['period'])
        st.write(job['description'])
        st.write("Technologies: " + ", ".join(job['tech']))
        st.markdown("---")

# Projects section
elif st.session_state.current_section == 'Projects':
    st.header("Projects")
    Projects = [
        {
            'title': 'Build a Spotify Connected App',
            'links': ['Android App', 'ScreenTime 2.0'],
            'description': 'A comprehensive video course teaching how to build a web app with Spotify Web API, covering REST APIs, OAuth, Node.js best practices.',
            'tech': ['Cordova', 'Backbone', 'JavaScript']
        },
        {
            'title': 'OctoProfile',
            'links': ['Demo', 'Code'],
            'description': 'A nicer look at your GitHub profile with data visualizations of languages, repos, contributions.',
            'tech': ['Next.js', 'Chart.js', 'GitHub API']
        }
    ]
    for proj in Projects:
        st.subheader(proj['title'])
        st.write(proj['description'])
        st.write("Technologies: " + ", ".join(proj['tech']))
        st.markdown("---")
