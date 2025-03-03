# main.py Documentation

## Overview
`main.py` serves as the backend logic for the Social Media Analytics Assistant. It establishes a connection to a Neo4j graph database and provides a function to generate responses to user queries about social media data using Langchain and Google's Gemini AI.

## Dependencies
- `langchain.graphs.Neo4jGraph`: For interfacing with the Neo4j graph database
- `dotenv`: For loading environment variables
- `langchain.chains.GraphCypherQAChain`: For creating a chain that converts natural language to Cypher queries
- `langchain_google_genai.ChatGoogleGenerativeAI`: For accessing Google's Gemini AI models
- `os`: For accessing environment variables

## Environment Variables
The script utilizes several environment variables:
- `GEMINI_API_KEY`: API key for Google's Gemini AI service
- `NEO4J_URI`: URI endpoint for the Neo4j database
- `NEO4J_PASSWORD`: Password for Neo4j database authentication
- Database name is hardcoded as 'neo4j'
- Username is hardcoded as 'neo4j'

## Main Function

### `generate(user_input)`
This is the core function that processes user queries and returns responses.

**Parameters:**
- `user_input` (str): The natural language query from the user

**Process:**
1. Establishes a connection to the Neo4j graph database using environment variables
2. Initializes the Google Gemini AI model with specific parameters:
   - Model: "gemini-2.0-flash"
   - Temperature: 0 (deterministic outputs)
   - No token limit specified
   - No timeout specified
3. Creates a GraphCypherQAChain that:
   - Takes natural language queries
   - Converts them to Cypher queries using the LLM
   - Executes the queries against the Neo4j database
   - Formats the results into human-readable responses
4. Processes the user input through this chain
5. Returns the result, falling back to different response formats if needed

**Returns:**
- A string response containing the answer to the user's query
- Returns either the 'result' field, 'answer' field, or the entire response object if neither is present

## Security Note
The function uses `allow_dangerous_requests=True` which allows potentially risky Cypher queries to be executed. This is appropriate for a controlled environment but should be carefully monitored in production. 