# Social Media Analytics Assistant - System Architecture

## System Overview

The Social Media Analytics Assistant is a web-based application that allows users to query social media analytics data stored in a Neo4j graph database using natural language. The system leverages Google's Gemini AI to translate user queries into Cypher (Neo4j's query language) and generate human-readable responses.

## Core Components

### 1. Web Interface (`web.py`)
The front-end user interface built with Streamlit that provides:
- An interactive chat interface
- Predefined sample queries
- Real-time streaming responses
- Session state management
- Error handling

### 2. Backend Logic (`main.py`)
The back-end functionality that:
- Connects to the Neo4j graph database
- Interfaces with Google's Gemini AI model
- Processes natural language queries
- Returns formatted responses

### 3. Neo4j Graph Database (External)
Stores the social media analytics data in a graph structure, allowing for complex relationship queries.

### 4. Google Gemini AI (External)
Provides natural language processing capabilities to:
- Understand user questions
- Generate Cypher queries
- Format database results into coherent responses

## Data Flow

1. **User Input**: User submits a question through the Streamlit interface.
2. **Query Processing**: The query is passed to the `generate` function in `main.py`.
3. **LLM Translation**: Google's Gemini AI translates the natural language query into a Cypher query.
4. **Database Query**: The Cypher query is executed against the Neo4j database.
5. **Result Formatting**: The database results are processed and formatted by the LLM.
6. **Response Display**: The formatted response is streamed back to the user interface with special handling for tabular data.

## Deployment Architecture

The application requires:
- A server to host the Streamlit application (`web.py`)
- Access to a Neo4j database instance
- Google Cloud API access for Gemini AI
- Environment variables configuration for secure credential management

## Security Considerations

- The system allows potentially dangerous Cypher queries through the `allow_dangerous_requests=True` setting
- API keys and database credentials are managed through environment variables
- No user authentication is implemented in the current version

## Future Enhancements

Potential improvements to the system architecture could include:
- User authentication and role-based access
- Query caching for improved performance
- More robust error handling and logging
- Database connection pooling
- Response quality monitoring and feedback mechanisms 