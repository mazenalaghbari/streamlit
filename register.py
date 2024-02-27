# register.py
import streamlit as st
from db_connection import create_connection

def register_page():
    st.header("Register")
    new_name = st.text_input("Name:")
    new_email = st.text_input("Email:")
    new_password = st.text_input("New Password:", type="password")
    
    if st.button("Register"):
        register_user(new_name, new_email, new_password)

def register_user(name, email, password):
    connection = create_connection()
    cursor = connection.cursor()
    
    # Use placeholders to prevent SQL injection
    insert_query = "INSERT INTO users (name, email, password) VALUES (%s, %s, %s)"
    values = (name, email, password)
    
    cursor.execute(insert_query, values)
    connection.commit()
    connection.close()
    
    st.success("Registration Successful!")
