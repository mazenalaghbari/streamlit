# welcome.py
import streamlit as st

def welcome_page():
    st.header("Welcome Page")
    if not hasattr(st.session_state, 'logged_in') or not st.session_state.logged_in:
        st.warning("Please login first.")
    else:
        st.success("Welcome! You are logged in.")
