# main.py
import streamlit as st

import home
import about
import education
import experience
import projects_page
import resume
import theme  

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="Vo Hong Quan's Portfolio", layout="wide")

# ---------- INJECT CSS ----------
st.markdown(theme.get_css(), unsafe_allow_html=True)

# ---------- LAYOUT: 3 CỘT ----------
left, spacer, right = st.columns([3, 1, 6])

# ===== LEFT: PROFILE (gói tất cả trong 1 markdown để không lệch) =====
profile_html = """
<div class="card profile-section">
  <!-- Floating dots (để nhẹ, dùng box-shadow/gradient) -->
  <div class="floating-element"></div>
  <div class="floating-element"></div>
  <div class="floating-element"></div>

  <h1 class="main-title">Vo Hong Quan</h1>
  <h2 class="subtitle">AI Engineer</h2>

  <div class="decoration-line"></div>

  <div class="card contact-section">
    <h5><i class="fas fa-address-book"></i> Get In Touch</h5>
    <div class="contact-item">
      <i class="fas fa-envelope"></i>
      <a href="mailto:vohongquan6524@gmail.com" style="color:inherit;text-decoration:none">
        22134012@student.hcmute.edu.vn
      </a>
    </div>
    <div class="contact-item">
      <i class="fas fa-phone"></i>
      <a href="tel:+84363645485" style="color:inherit;text-decoration:none">0363645485</a>
    </div>
    <div class="contact-item">
      <i class="fas fa-map-marker-alt"></i>
      Ho Chi Minh City, Vietnam
    </div>
  </div>

  <div class="decoration-line"></div>
  
  <div class="card social-container" style="margin-top:1rem">
    <h5><i class="fas fa-share-alt"></i> Connect With Me</h5>
    <div class="social-icons">
      <a href="https://github.com/honggquan24" target="_blank" class="social-icon" aria-label="GitHub">
        <i class="fab fa-github"></i>
      </a>
      <a href="https://www.linkedin.com/in/vo-hong-quan-b50063373/" target="_blank" class="social-icon" aria-label="LinkedIn">
        <i class="fab fa-linkedin"></i>
      </a>
      <a href="https://www.kaggle.com/honggquan" target="_blank" class="social-icon" aria-label="Kaggle">
        <i class="fab fa-kaggle"></i>
      </a>
    </div>
  </div>
</div>
"""
with left:
    st.markdown(profile_html, unsafe_allow_html=True)

# ===== RIGHT: TABS (không bọc <div> thủ công để tránh ô rỗng) =====
with right:
    tabs = st.tabs(["Home", "About", "Education", "Experience", "Projects", "Resume"])

    with tabs[0]:
        home.show_home()
    with tabs[1]:
        about.show_about()
    with tabs[2]:
        education.show_education()
    with tabs[3]:
        experience.show_experience()
    with tabs[4]:
        projects_page.show_projects()
    with tabs[5]:
        resume.show_resume()
