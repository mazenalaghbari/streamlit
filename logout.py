import streamlit as st

def logout_page():
    st.header("Logout Page")
    if not hasattr(st.session_state, 'logged_in') or not st.session_state.logged_in:
        st.warning("You are not logged in.")
    else:
        st.success("You have been successfully logged out.")
        st.session_state.logged_in = False  # Reset the login status

# Usage
if __name__ == "__main__":
    logout_page()
