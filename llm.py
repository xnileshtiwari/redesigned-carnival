from langchain_google_genai import ChatGoogleGenerativeAI
import os
from pinecone import Pinecone
from langchain_cohere import CohereEmbeddings
from langsmith import Client, traceable
from dotenv import load_dotenv
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
import uuid
from pydantic import BaseModel
from typing import List, Dict, Tuple


load_dotenv()
class PineconeVectorStore(BaseModel):
    index_name: str
    query: str

class QueryResult(BaseModel):
    text: str
    metadata: Dict
    score: float


# Load environment variables from .env file
load_dotenv()

# Get API keys from environment variables
google_api_key = os.getenv("GOOGLE_API_KEY")
langsmith_api_key = os.getenv("LANGSMITH_API_KEY")

# Enable LangSmith tracing
os.environ["LANGSMITH_TRACING"] = "true"
custom_client = Client(api_key=langsmith_api_key)

# Initialize LlamaCloud index

# Define the retrieval tool
@tool
def retrieve(query: str):
    """This tool contains all the information You are ever going to be asked about."""
    try:
        # Initialize embeddings and Pinecone
        embeddings = CohereEmbeddings(
                model="embed-multilingual-v3.0",
            )
        
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        index_name = "italian-pdf-docs"
        index = pc.Index(index_name)
        
        # Update trending count
        
        # Get query embedding
        query_embedding = embeddings.embed_query(query)
        
        # Query Pinecone
        results = index.query(
            vector=query_embedding,
            top_k=10,
            include_metadata=True,
            namespace="Test-1",
        )
        
        # Extract results and metadata
        query_results = []
        for match in results["matches"]:
            text = match["metadata"].get("text", "")
            metadata = {
                "page": match["metadata"].get("page", "Unknown"),
                "score": match["score"],
                # Add any other metadata fields you want to track
                "chunk_index": match["metadata"].get("chunk_index", "Unknown"),
                "filename": match["metadata"].get("filename", "Unknown"),
            }
            query_results.append(QueryResult(text=text, metadata=metadata, score=match["score"]))
        
        # Print results for verification
        
        # Return both texts and full metadata
        texts = [result.text for result in query_results]
        metadata_list = [result.metadata for result in query_results]
        return texts, metadata_list
        
    except Exception as e:
        print(f"An error occurred in pinecone vector database query: {e}")









# Initialize the LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp",
    temperature=0.2,
)

# System instructions
instructions = """
# SYSTEM INSTRUCTIONS FOR ADA - COMPANY DATABASE ASSISTANT

You are Ada, a specialized assistant designed to help users find information in the company database.

## CORE BEHAVIOR PRINCIPLES:
1. **ALWAYS USE THE RETRIEVAL TOOL FIRST**
   - Your knowledge comes ONLY from the company documents
   - You MUST use the retrieve tool before answering ANY question
   - NEVER rely on your training data or prior knowledge
   - If a user asks a question, ALWAYS check the database FIRST

2. **CITE ALL INFORMATION PROPERLY**
   - Every fact must include a citation: **[Document Name]** **[Page X]**
   - Citations must appear immediately after the information they support
   - Always include both document name and page number
   - Make citations bold to stand out clearly

3. **HANDLING INFORMATION GAPS**
   - If the retrieve tool returns no relevant information, say: "I don't have information about [topic] in our company database. Would you like me to search for something else?"
   - NEVER make up information or rely on general knowledge
   - Do not deny requests without checking the database first

## RESPONSE WORKFLOW:
1. ALWAYS use the retrieve tool with the user's query
2. Examine ALL retrieved documents thoroughly
3. Formulate a response using ONLY the retrieved information
4. Include proper citations for EVERY piece of information
5. If information is missing, acknowledge the gap and offer to help search differently

## CITATION FORMAT:
- Single fact: "The return policy is 30 days **[Return Policy.pdf]** **[Page 2]**."
- Multiple sources: "Shipping costs €5 for standard delivery **[Shipping Guide.pdf]** **[Page 1]** and €15 for express **[Express Services.pdf]** **[Page 3]**."
- Repeated information from same source: Cite once per paragraph if all information comes from the same source.

Remember: Your PRIMARY function is to retrieve and share document-based information with proper citations. NEVER skip using the retrieve tool."""

# Set up the agent
tools = [retrieve]
memory = MemorySaver()
agent_executor = create_react_agent(llm, tools, checkpointer=memory)

# Function to get agent response with LangSmith tracing
@traceable(client=custom_client, run_type="llm", name="AI-CASE", project_name="Fiverr")
def get_completion(user_message, thread_id, is_first=False):
    """Get a response from the agent, maintaining chat history."""
    config = {"configurable": {"thread_id": thread_id}}
    messages = []
    
    # Always include system instructions to ensure the model follows citation requirements
    system_message = {"role": "system", "content": instructions}
    messages.append(system_message)
    
    # Add a reminder about citations to the user's message
    enhanced_message = f"{user_message}\n\n(Remember to cite ALL information from the documents using the format: **[Document Name]** **[Page X]**)"
    messages.append({"role": "user", "content": enhanced_message})
    
    final_state = None
    try:
        for event in agent_executor.stream({"messages": messages}, stream_mode="values", config=config):
            final_state = event
        if final_state:
            messages = final_state["messages"]
            for msg in reversed(messages):
                if msg.type == "ai":
                    return msg.content
        return "No response generated."
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return f"Error: Unable to generate response due to {str(e)}"


# Function to test the agent's response
def test_agent_response(query):
    """Test the agent's response to a query."""
    test_thread_id = str(uuid.uuid4())
    print(f"\n\nTesting agent response for query: '{query}'")
    print("-" * 40)
    response = get_completion(query, test_thread_id)
    print("\nResponse:")
    print(response)
    print("\nEnd of test")
    return response

