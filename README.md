# LlamaIndex with Google Gemini

This project demonstrates how to use Google's Gemini LLM with LlamaIndex for querying a LlamaCloud index.

## Setup

1. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

2. Set up your API keys:
   - Get a Google API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Add your API keys to the `.env` file:
     ```
     GOOGLE_API_KEY=your_google_api_key_here
     LLAMA_CLOUD_API_KEY=your_llama_cloud_api_key_here
     ```

## Usage

Run the test script:
```bash
python test.py
```

This will:
1. Configure LlamaIndex to use Google's Gemini Pro model
2. Connect to your LlamaCloud index
3. Retrieve information based on your query
4. Print the response

## Troubleshooting

If you encounter errors:

1. Make sure your API keys are correctly set in the `.env` file
2. Ensure you have installed all required packages
3. Check that your Google API key has access to the Gemini Pro model
4. Verify that your LlamaCloud index exists and is accessible with your API key 