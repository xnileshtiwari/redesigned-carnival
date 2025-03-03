# web.py Documentation

## Overview
`web.py` is a Streamlit-based web application that serves as a Social Media Analytics Assistant. It provides a chat interface where users can ask questions about social media analytics data stored in a Neo4j database.

## Dependencies
- `streamlit`: For creating the web interface
- `time`: For timing operations (imported but not explicitly used)
- `random`: For randomization (imported but not explicitly used)
- `main`: Custom module containing the `generate` function that interfaces with the Neo4j database

## Page Configuration
- Title: "AI Chat Assistant"
- Icon: "💬" (chat bubble emoji)
- Layout: Centered
- Page Header: "Social Media Analytics Assistant📈"

## Custom Styling
The application uses custom CSS styling for:
- User and assistant avatars
- Chat message formatting
- Sample prompt buttons
- Layout and visual elements

## Main Components

### Session State Management
- Initializes an empty chat history array (`st.session_state.messages`) if not already present

### Predefined Prompts
The application offers four sample questions as quick-access buttons:
1. "🔍 Which is the most liked post?"
2. "💡 Which post has the most comments?"
3. "🔥 Which post has the most shares?"
4. "📜 Where the post 7176541902119280640 is uploaded?"

### User Interface Layout
- Sample prompts section with a 2x2 grid of buttons
- Chat history display showing previous interactions
- Input box for user questions

### Chat Functionality
- Displays existing chat history from session state
- Accepts user input via chat input field or predefined prompt buttons
- Processes user questions through the `generate` function from `main.py`
- Renders responses with proper formatting, including special handling for tabular data
- Updates the session state with new messages
- Includes error handling for blocked responses

### Response Streaming
- Implements a streaming mechanism for displaying the assistant's response chunk by chunk
- Special handling for tabular data, ensuring tables are properly formatted
- Uses placeholders for real-time updates of the response

## Error Handling
The application catches exceptions during response generation and displays a user-friendly error message.

## UI Refreshing
Uses `st.rerun()` to refresh the UI after receiving user input and generating a response. 