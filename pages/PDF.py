import streamlit as st
from document_processing import document_chunking_and_uploading_to_vectorstore

# Load custom CSS
def load_custom_css():
    with open('custom.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_custom_css()

# Set page title
st.title("📤 Upload PDF")


# File uploader with form
import tempfile

with st.form("pdf_upload_form", clear_on_submit=True):
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    submit_button = st.form_submit_button("Upload and Process")

    if submit_button:
        if uploaded_file is not None:
            with st.spinner("Processing your PDF..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_file_path = tmp_file.name

                result = document_chunking_and_uploading_to_vectorstore(tmp_file_path, uploaded_file.name)
                st.success(result)
        else:
            st.error("Please upload a PDF file first")
