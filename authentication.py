# authentication.py
import streamlit as st
from db_connection import create_connection

def authenticate_user(username, password):
    connection = create_connection()
    cursor = connection.cursor()
    query = f"SELECT * FROM users WHERE email='{username}' AND password='{password}'"
    cursor.execute(query)
    user = cursor.fetchone()
    connection.close()
    return user

def login_page():
    st.header("Login")
    username = st.text_input("Username:")
    password = st.text_input("Password:", type="password")
    if st.button("Login"):
        user = authenticate_user(username, password)
        if user:
            st.success("Login Successful!")
            st.session_state.logged_in = True
        else:
            st.error("Invalid Credentials")
