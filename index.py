# index.py
import streamlit as st

def index_page():
    st.header("Index Page")
    # Add Google logo
    st.image("https://logowik.com/content/uploads/images/abc-australian-broadcasting-corporation2950.jpg", caption="", use_column_width=True)

    # Dummy content
    st.write("Here is some dummy content for the index page:")
    items = ["Item 1", "Item 2", "Item 3"]
    st.write(items)

