"""Run with python -m streamlit run participant_experience_review_app.py."""
import streamlit as st
from marathon_absa.participant_experience_review_ui import render_review

st.set_page_config(page_title='Researcher review | SESA', layout='wide')
render_review()
