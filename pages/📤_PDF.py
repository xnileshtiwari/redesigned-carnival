import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import streamlit as st
from document_processing import document_chunking_and_uploading_to_vectorstore

# Load custom CSS
def load_custom_css():
    with open('custom.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_custom_css()

# Set page title
st.title('Upload PDF')

# File uploader with form
import tempfile

with st.form("pdf_upload_form", clear_on_submit=True):
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    submit_button = st.form_submit_button("Upload and Process")

    if submit_button:
        if uploaded_file is not None:
            with st.spinner("Processing your PDF..."):
                result = document_chunking_and_uploading_to_vectorstore(uploaded_file)
                st.write(result)  # Display the result instead of 'None'
