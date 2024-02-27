import streamlit as st
from authentication import authenticate_user, login_page
from register import register_page
from welcome import welcome_page
from index import index_page
from db_connection import create_connection
from config import *
from askme import askme_page
from text_analysis_app import explore_page

def main():
    
    st.title(APP_NAME)

    # Navigation bar
    menu = ["Login", "Register", "Index", "Welcome" , "ask", "explore"]
    if hasattr(st.session_state, 'logged_in') and st.session_state.logged_in:
        menu.append("Logout")
    choice = st.sidebar.selectbox("Select Page", menu)

    if choice == "Login":
        login_page()
    elif choice=="ask":
        askme_page()

    elif choice == "Register":
        register_page()

    elif choice == "Index":
        index_page()
        
    elif choice=="explore":
        explore_page()

    elif choice == "Welcome":
        welcome_page()

    elif choice == "Logout":
        st.session_state.logged_in = False
        st.sidebar.warning("You have been logged out successfully.")

if __name__ == "__main__":
    main()
