# Redesigned Carnival

This project is a Document Chat Assistant that allows users to interact with PDF and CSV documents using a chat interface. The application leverages Google's Gemini LLM and LlamaIndex for querying and retrieving information.

## Features

- **PDF Document Chat Mode**: Interact with PDF documents to extract and summarize information.
- **CSV Data Chat Mode**: Analyze CSV and Excel files to answer data-related queries.
- **Feedback System**: Users can provide feedback on the responses to improve the system.

## Setup

### Prerequisites

- Python 3.7 or higher
- [Streamlit](https://streamlit.io/)
- [Pandas](https://pandas.pydata.org/)
- [Langchain](https://github.com/hwchase17/langchain)
- [Pinecone](https://www.pinecone.io/)

### Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/xnileshtiwari/redesigned-carnival.git
   cd redesigned-carnival
   ```

2. **Install Required Packages**

   Use the following command to install the necessary Python packages:

   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up Environment Variables**

   Create a `.env` file in the root directory and add your API keys:

   ```plaintext
   GOOGLE_API_KEY=your_google_api_key_here
   PINECONE_API_KEY=your_pinecone_api_key_here
   COHERE_API_KEY=your_cohere_api_key_here
   LANGSMITH_API_KEY=your_langsmith_api_key_here
   

   ```

### Usage

1. **Run the Application**

   Start the Streamlit app by running:

   ```bash
   streamlit run chat_ui.py
   ```

2. **Interact with the App**

   - **PDF Mode**: Select a PDF document from the sidebar and ask questions about its content.
   - **CSV Mode**: Choose a CSV or Excel file and query the data for insights.

3. **Provide Feedback**

   Use the feedback buttons to like or dislike the responses, helping improve the system.

### Troubleshooting

- Ensure your API keys are correctly set in the `.env` file.
- Verify that all required packages are installed.


## License

This project is licensed under the MIT License.

