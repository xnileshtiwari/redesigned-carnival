import pandas as pd
import streamlit as st
import dotenv
import os
import json
from llm import get_completion
from csv_agent import run_csv_chat_agent
import pandas as pd
import numpy as np


# Load environment variables from a .env file
dotenv.load_dotenv()

# Load custom CSS
# Page configuration
st.set_page_config(
    page_title="Document Chat Assistant",
    page_icon="📊",
    layout="centered"
)

thread_id_1 = "conversation_1"

def load_custom_css(file_path):
    with open(file_path) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Apply custom CSS
load_custom_css('custom.css')

# Function to save feedback to JSON files
def save_feedback(user_input, assistant_response, feedback_type):
    # Create directory if it doesn't exist
    os.makedirs("feedback", exist_ok=True)
    
    # Path to the feedback file
    json_file = f"feedback/{feedback_type}.json"
    
    # Prepare the feedback data
    feedback_data = {
        "user_input": user_input,
        "assistant_response": assistant_response,
        "timestamp": st.session_state.get("current_time", "")
    }
    
    # Load existing feedback data if the file exists
    existing_data = []
    if os.path.exists(json_file):
        try:
            with open(json_file, 'r') as f:
                existing_data = json.load(f)
        except json.JSONDecodeError:
            # If the file is empty or invalid JSON
            existing_data = []
    
    # Append new feedback and save
    existing_data.append(feedback_data)
    with open(json_file, 'w') as f:
        json.dump(existing_data, f, indent=2)

# Hide Streamlit default elements
hide_streamlit_style = """
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}
    [data-testid="stToolbar"] {visibility: hidden !important;}
    [data-testid="stDecoration"] {visibility: hidden !important;}
    .viewerBadge_link__1S137 {display: none !important;}
    .viewerBadge_container__1QSob {display: none !important;}
    .stAttribution {display: none !important;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# App title with emoji
st.title("📚 Document Assistant")

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_mode" not in st.session_state:
    st.session_state.chat_mode = "PDF"

if "selected_csv" not in st.session_state:
    st.session_state.selected_csv = None

if "previous_mode" not in st.session_state:
    st.session_state.previous_mode = "PDF"

if "previous_csv" not in st.session_state:
    st.session_state.previous_csv = None

if "feedback_given" not in st.session_state:
    st.session_state.feedback_given = {}

# Current time for timestamp
from datetime import datetime
st.session_state.current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Hard-coded PDF documents (since they're already embedded in the model)
pdf_documents = [
    "GLS Bergamo National Italy Contract.pdf",
    "Poste Delivery Business Pricing.jpeg",
    "Poste Delivery Business Contract.pdf",
    "UPS International Pricing Contract.png",
    "UPS Standard International Pricing Contract.png",
    "Zoning Fedex International Contract.pdf",
    "UPS Standard Pickup Point international Pricing.png",
    "DHL International Contract.pdf",
    "Postegofresh Contract Italy Terms Conditions.pdf",
    "TNT Contract.pdf",
    "picking-list by product.pdf",
    "GLS Bergamo International Contract.pdf",
    "Fedex International Contract.pdf"

]

# Function to get available CSV documents
def get_csv_documents():
    # Change this path to where your CSV documents are stored
    csv_path = "/home/xnileshtiwari/vscode/new-gemini-upload-file/Updated_CSVs/"
    data_files = []
    for file in os.listdir(csv_path):
        if file.endswith('.csv') or file.endswith('.xlsx'):
            data_files.append((file, os.path.join(csv_path, file)))
    return data_files

# Side panel for mode selection and document selection
with st.sidebar:
    st.title("Settings")
    
    # Mode selection
    st.subheader("Chat Mode")
    
    # Custom mode selector with better styling
    # Create radio buttons with custom styling
    col1, col2 = st.columns(2)
    
    with col1:
        pdf_button = st.button(
            "📄 PDF", 
            key="pdf_button",
            use_container_width=True,
            type="primary" if st.session_state.chat_mode == "PDF" else "secondary"
        )
    
    with col2:
        csv_button = st.button(
            "📊 Data", 
            key="csv_button",
            use_container_width=True,
            type="primary" if st.session_state.chat_mode == "CSV" else "secondary"
        )

    # Handle button clicks
    if pdf_button:
        st.session_state.chat_mode = "PDF"
        st.rerun()
    
    if csv_button:
        st.session_state.chat_mode = "CSV"
        st.rerun()


    # JavaScript for handling mode change
    st.markdown("""
    <script>
    function handleModeChange(mode) {
        const selectElement = document.querySelector('input[name="mode_selector"]');
        if (mode === 'PDF') {
            selectElement.click();
        } else {
            const labels = document.querySelectorAll('.stRadio label');
            if (labels.length > 1) {
                labels[1].click();
            }
        }
    }
    </script>
    """, unsafe_allow_html=True)
    
    # Actual radio buttons (hidden but functional)
    chat_mode = st.radio(
        "Select chat mode",
        options=["PDF", "CSV"],
        index=0 if st.session_state.chat_mode == "PDF" else 1,
        key="mode_selector",
        label_visibility="collapsed"
    )
    
    # Update chat mode in session state
    st.session_state.chat_mode = chat_mode
    
    # Document selection based on mode
    if chat_mode == "PDF":
        st.subheader("Available PDF Documents")
        st.markdown('<div class="pdf-list">', unsafe_allow_html=True)
        for doc in pdf_documents:
            st.markdown(f'<div class="pdf-list-item"><svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line><polyline points="10 9 9 9 8 9"></polyline></svg> {doc}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:  # CSV mode
        st.subheader("Select Data File (CSV/Excel)")
        csv_documents = get_csv_documents()
        if csv_documents:
            csv_names = [doc[0] for doc in csv_documents]
            csv_paths = [doc[1] for doc in csv_documents]
            selected_csv_name = st.selectbox(
                "Choose a data file",
                options=csv_names,
                index=0,
                key="csv_selector"
            )
            selected_index = csv_names.index(selected_csv_name)
            st.session_state.selected_csv = csv_paths[selected_index]
        else:
            st.warning("No data files found.")
            st.session_state.selected_csv = None

    st.divider()
    
    # Clear chat button
    if st.button("Clear Chat", key="clear_chat"):
        st.session_state.messages = []
        st.session_state.feedback_given = {}
        st.rerun()

# Check if mode or CSV selection has changed
if (st.session_state.previous_mode != st.session_state.chat_mode or 
    (st.session_state.chat_mode == "CSV" and 
     st.session_state.previous_csv != st.session_state.selected_csv)):
    # Reset chat
    st.session_state.messages = []
    st.session_state.feedback_given = {}
    st.session_state.previous_mode = st.session_state.chat_mode
    st.session_state.previous_csv = st.session_state.selected_csv

# Predefined sample prompts
pdf_prompts = [
    "📄 What are the payment terms mentioned in the TNT contract?",
    "📑 Summarize the main points in the GLS Bergamo National Italy Contract",
    "🔍 What additional services can be added to the basic shipping service?",
    "⚖️ What are the legal obligations mentioned?"
]

csv_prompts = [
    "📊 What is the total sales amount?",
    "📈 Show me the top 5 customers by revenue",
    "🔢 Calculate the average order value",
    "📉 What's the trend of sales over time?"
]


# Display sample prompts based on mode
st.subheader("Sample Prompts")

if st.session_state.chat_mode == "PDF":
    prompts = pdf_prompts
else:
    prompts = csv_prompts

col1, col2 = st.columns(2)
with col1:
    if st.button(prompts[0]):
        st.session_state.temp_input = prompts[0]
    
    if st.button(prompts[2]):
        st.session_state.temp_input = prompts[2]

with col2:
    if st.button(prompts[1]):
        st.session_state.temp_input = prompts[1]
    
    if st.button(prompts[3]):
        st.session_state.temp_input = prompts[3]

# Chat container - directly show mode info without extra spacing
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

# Display current mode
if st.session_state.chat_mode == "PDF":
    st.info("🔍 Currently in PDF Document Chat Mode")
else:
    if st.session_state.selected_csv:
        csv_name = os.path.basename(st.session_state.selected_csv)
        def column_names(csv_file):
            df = pd.read_csv(csv_file)
            return df.columns.tolist(), df.shape

        column_names, shape = column_names(st.session_state.selected_csv)
        with st.expander("CSV File Information"):
            st.warning(f"Column names: {', '.join(column_names)}", icon="ℹ️")
            st.warning(f"Shape: {shape}", icon="ℹ️")
        st.info(f"📊 Currently Querying: **{csv_name}**")

    else:
        st.warning("⚠️ Please select a CSV file from the sidebar")

# Function to generate a unique ID for each message pair
def get_message_id(idx):
    return f"msg_{idx}"

# Display chat messages with feedback buttons
for idx, message in enumerate(st.session_state.messages):
    msg_id = get_message_id(idx // 2)  # Group user and assistant messages
    
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Add feedback buttons after assistant messages
        if message["role"] == "assistant" and idx > 0:
            # Only show feedback buttons if feedback hasn't been given yet
            if msg_id not in st.session_state.feedback_given:
                # Create columns for the feedback buttons
                feedback_cols = st.columns([0.9, 0.05, 0.05])
                
                # Like button
                with feedback_cols[1]:
                    like_button = st.button("👍", key=f"like_{msg_id}")
                    if like_button:
                        user_message = st.session_state.messages[idx-1]["content"]
                        assistant_message = message["content"]
                        save_feedback(user_message, assistant_message, "liked")
                        st.session_state.feedback_given[msg_id] = "liked"
                        st.rerun()
                
                # Dislike button
                with feedback_cols[2]:
                    dislike_button = st.button("👎", key=f"dislike_{msg_id}")
                    if dislike_button:
                        user_message = st.session_state.messages[idx-1]["content"]
                        assistant_message = message["content"]
                        save_feedback(user_message, assistant_message, "disliked")
                        st.session_state.feedback_given[msg_id] = "disliked"
                        st.rerun()
            else:
                # Show confirmation that feedback was received
                feedback_type = st.session_state.feedback_given[msg_id]
                st.caption(f"Thank you for your feedback! ({feedback_type})")

# User input
user_input = st.chat_input("Ask something about your documents...")

# Use predefined prompt if selected, otherwise use user input
if 'temp_input' in st.session_state and st.session_state.temp_input:
    user_input = st.session_state.temp_input
    st.session_state.temp_input = None  # Clear the temporary input

if user_input:
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Use st.status instead of st.spinner
    with st.status("Processing your request...", expanded=True) as status:
        # Process based on mode
        if st.session_state.chat_mode == "PDF":
            status.update(label="Thinking...", state="running")
            response = get_completion(user_input, thread_id_1)
            status.update(label="✅ Analysis complete", state="complete")
        else:  # CSV mode
            if st.session_state.selected_csv:
                # status.update(label=f"Analyzing CSV data from {os.path.basename(st.session_state.selected_csv)}...", state="running")
                status.update(label=f"Thinking...", state="running")

                thread_id = "conversation_1"

                response = run_csv_chat_agent(st.session_state.selected_csv, user_input, thread_id)
                status.update(label="✅ Analysis complete", state="complete")
            else:
                response = "Please select a CSV file from the sidebar first."
                status.update(label="⚠️ No CSV file selected", state="error")
    

    # Add assistant response to chat
    st.session_state.messages.append({"role": "assistant", "content": response})
    
    with st.chat_message("assistant"):
        st.markdown(response)
    
    st.rerun()

st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="margin-top: 20px; text-align: center; color: #888;">
    <p>Made with ❤️ by Nilesh | Data updated: March 12, 2025</p>
</div>
""", unsafe_allow_html=True)
