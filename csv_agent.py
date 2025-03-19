import os
from langsmith import Client, traceable
import pandas as pd
from dotenv import load_dotenv
from typing import TypedDict, List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.tools import PythonAstREPLTool
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
import numpy as np
from termcolor import colored
from prompts import (
    interpret_question_prompt,
    generate_query_prompt,
    format_response_prompt,
)

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables.")

# Initialize the LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=10,
    api_key=GOOGLE_API_KEY,
)
langsmith_api_key = os.getenv("LANGSMITH_API_KEY")
custom_client = Client(api_key=langsmith_api_key)

# Shared MemorySaver instance
memory = MemorySaver()

LANGSMITH_TRACING = True

@traceable(client=custom_client, run_type="llm", name="CSV-Agent", project_name="CSV_TO_CHAT")
def run_csv_chat_agent(file_path: str, user_question: str, thread_id: str) -> str:
    """
    Runs a chat agent that processes a user question against a CSV file, formats the final response,
    and maintains conversation history for context-aware responses.
    """
    # Load the dataframe
    if file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path)
    elif file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    else:
        raise ValueError("Unsupported file format. Use .xlsx or .csv.")

    # Create the Python REPL tool
    tool = PythonAstREPLTool(locals={"df": df})
    df_5_rows = df.head().to_markdown()
    columns = df.columns.tolist()  # Store columns for reuse

    # Prepare dataframe info for prompts
    df_info = f"""
    DataFrame Information:
    - Shape: {df.shape}
    - Columns: {', '.join(columns)}
    - Data Types:
    {df.dtypes.to_string()}
    First 5 rows:
    {df.head().to_markdown()}
    """    

    csv_description = f"This is a csv file"
    # print(colored(f"The csv description is: {csv_description}", "red"))


    # Define node functions
    def interpret_question_node(state: dict) -> dict:
        """Rephrases the user's question based on conversation history."""
        question = state["question"]
        history = state.get("history", [])
        history_str = "\n".join([f"User: {q}\nAssistant: {a}" for q, a in history[-5:]])
        prompt = interpret_question_prompt.format(history_str=history_str, question=question, df_info=df_info, csv_description=csv_description)
        response = llm.invoke(prompt)
        standalone_question = response.content.strip()
        print(colored(f"Standalone question: {standalone_question}", 'green'))
        return {"standalone_question": standalone_question}


    def generate_query_node(state: dict) -> dict:
        """Generates the initial Python query based on the standalone question."""
        standalone_question = state["standalone_question"]
        prompt = generate_query_prompt.format(df_info=df_info, standalone_question=standalone_question, csv_description=csv_description)
        response = llm.invoke(prompt)
        query = response.content.strip()
        print(colored(f"Generated query: {query}", 'blue'))
        return {"query": query}

    def execute_query_node(state: dict) -> dict:
        """Executes the Python query on the DataFrame."""
        query = state["query"]
        try:
            response = tool.run(query)
            return {"response": str(response)}
        except Exception as e:
            return {"response": f"Error executing query: {str(e)}"}


    def format_response_node(state: dict) -> dict:
        """Formats the raw response and updates history."""
        question = state["question"]
        response = state["response"]
        prompt = format_response_prompt.format(question=question, response=response, df_info=df_5_rows, csv_description=csv_description)
        formatted_answer = llm.invoke(prompt).content.strip()
        history = state.get("history", [])
        history.append((question, formatted_answer))
        return {"final_answer": formatted_answer, "history": history}




    # Updated state structure
    class GraphState(TypedDict):
        history: List[tuple]
        question: str
        standalone_question: str
        is_relevant: bool  # New field
        query: str
        response: str
        grade: str
        attempts: int
        final_answer: str

    # Build the workflow graph
    graph = StateGraph(GraphState)
    graph.add_node("interpret_question", interpret_question_node)
    graph.add_node("generate_query", generate_query_node)
    graph.add_node("execute_query", execute_query_node)
    graph.add_node("format_response", format_response_node)

    # Set entry point
    graph.set_entry_point("interpret_question")

    # Define edges
    graph.add_edge("interpret_question", "generate_query")
    graph.add_edge("generate_query", "execute_query")
    graph.add_edge("execute_query", "format_response")
    graph.add_edge("format_response", END)

    # Compile the graph
    app = graph.compile(checkpointer=memory)

    config = {"configurable": {"thread_id": thread_id}}
    initial_state = {"question": user_question, "attempts": 0}
    final_state = app.invoke(initial_state, config=config)
    print(colored(f"Final state: {final_state}", "red"))
    return final_state["final_answer"]

# Example usage
# response = run_csv_chat_agent("/path/to/export22.csv", "Give me all the columns in the dataframe", "123")
# print(colored(response, 'red', attrs=['bold', 'underline']))
