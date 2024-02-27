# askme.py
import streamlit as st
from mymodel import classify_input
def askme_page():
    st.title("Ask Me Anything Page")

    # User input form
    st.header("User Input")
    sepal_length = st.slider("Sepal Length", 0.0, 10.0, 5.0)
    sepal_width = st.slider("Sepal Width", 0.0, 10.0, 5.0)
    petal_length = st.slider("Petal Length", 0.0, 10.0, 5.0)
    petal_width = st.slider("Petal Width", 0.0, 10.0, 5.0)

    input_data = [[sepal_length, sepal_width, petal_length, petal_width]]

    if st.button("Classify"):
        try:
            result = classify_input(input_data)
            st.success(f"Predicted Species: {result}")
        except ValueError as e:
            st.error(str(e))
