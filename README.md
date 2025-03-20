# Redesigned Carnival

This project is a Document Chat Assistant that allows users to interact with PDF and CSV documents using a chat interface. The application leverages Google's Gemini LLM and LlamaIndex for querying and retrieving information.

## Features

- **PDF Document Chat Mode**: Interact with PDF documents to extract and summarize information.
- **CSV Data Chat Mode**: Analyze CSV and Excel files to answer data-related queries.
- **Feedback System**: Users can provide feedback on the responses to improve the system.

## Setup

### Prerequisites


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
   streamlit run app.py
   ```

2. **Interact with the App**



### Troubleshooting

- Ensure your API keys are correctly set in the `.env` file.
- Verify that all required packages are installed.


## License

This project is licensed under the MIT License.

