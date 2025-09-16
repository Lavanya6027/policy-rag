import streamlit as st
import requests

# Backend URL (FastAPI)
BACKEND_URL = "http://127.0.0.1:5000"

st.set_page_config(page_title="HR Assistant - Lia", layout="centered")

st.title("🤖 HR Assistant - Lia")

# Chat Section
# -------------------------------
st.header(" Ask a Question")

user_query = st.text_input("Enter your question:")
if st.button("Ask"):
    if user_query.strip():
        with st.spinner("Thinking..."):
            response = requests.post(f"{BACKEND_URL}/chat", json={"query": user_query})
            if response.status_code == 200:
                st.markdown(f"**Answer:** {response.json()['answer']}")
            else:
                st.error(" Failed to fetch answer.")
    else:
        st.warning("Please enter a question.")
