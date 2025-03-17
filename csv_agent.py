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
    grade_response_prompt,
    transform_query_prompt,
    format_response_prompt,
    relevance_check_prompt  # Added new prompt
)

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables.")

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


    def get_csv_info():
    # Check if 'Data_info' column exists in the DataFrame
        if 'Data_info' in df.columns:
            data = df['Data_info'].iloc[0]
            formatted = f"This is CSV file with {data}"
            return df['Data_info'].iloc[0]
        else:
            last_column = df.columns[-1]
            lst_data =  df[last_column].iloc[0]
            formatted = f"This is CSV file with {lst_data}"
            return formatted
    

    csv_description = f"This is a csv file {get_csv_info()}"
    print(colored(f"The csv description is: {csv_description}", "red"))


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

    # New node to check question relevance
    def identify_relevance_node(state: dict) -> dict:
        """Checks if the question can be answered with the available columns."""
        standalone_question = state["standalone_question"]
        columns_str = ', '.join(columns)
        history = state.get("history", [])
        history_str = "\n".join([f"User: {q}\nAssistant: {a}" for q, a in history[-5:]])

        prompt = relevance_check_prompt.format(standalone_question=standalone_question, columns=columns_str, history_str=history_str, csv_description=csv_description)
        response = llm.invoke(prompt).content.strip().lower()
        is_relevant = response == 'yes'
        print(colored(f"Question relevance: {'Yes' if is_relevant else 'No'}", 'yellow'))
        return {"is_relevant": is_relevant}

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

    def grade_response_node(state: dict) -> dict:
        """Grades whether the response answers the standalone question."""
        attempt_number = state["attempts"]
        print(colored(f"Attempt number: {attempt_number}", 'cyan', attrs=['bold']))

        standalone_question = state["standalone_question"]
        response = state["response"]
        prompt = grade_response_prompt.format(standalone_question=standalone_question, response=response, csv_description=csv_description, attempt_number= attempt_number)
        grade_response = llm.invoke(prompt).content.strip().lower()
        print(colored(f"Question: {standalone_question}\nResponse: {response}\nGrade: {grade_response}", 'magenta', attrs=['bold']))
        if grade_response == "yes":
            return {"grade": "yes"}
        else:
            return {"grade": "no", "attempts": state["attempts"] + 1}

    def transform_query_node(state: dict) -> dict:
        """Transforms the query if the previous response was inadequate."""
        standalone_question = state["standalone_question"]
        previous_query = state["query"]
        previous_response = state["response"]
        prompt = transform_query_prompt.format(
            df_info=df_info,
            standalone_question=standalone_question,
            previous_query=previous_query,
            previous_response=previous_response,
            csv_description=csv_description
        )
        response = llm.invoke(prompt)
        new_query = response.content.strip()
        return {"query": new_query}

    def format_response_node(state: dict) -> dict:
        """Formats the raw response and updates history."""
        question = state["question"]
        response = state["response"]
        prompt = format_response_prompt.format(question=question, response=response, df_info=df_5_rows, csv_description=csv_description)
        formatted_answer = llm.invoke(prompt).content.strip()
        history = state.get("history", [])
        history.append((question, formatted_answer))
        return {"final_answer": formatted_answer, "history": history}

    # New node for irrelevant questions
    def set_irrelevant_message_node(state: dict) -> dict:
        """Sets a message when the question is irrelevant and updates history."""
        question = state["question"]
        columns_str = ', '.join(columns)
        final_answer = (
            f"It seems like your question might not be relevant to this dataset. "
            f"The available columns are: {columns_str}. "
            f"Please check if you're querying the correct CSV or rephrase your question."
        )
        history = state.get("history", [])
        history.append((question, final_answer))
        print(colored(f"Final Answer: {final_answer}", 'blue', attrs=['bold']))
        return {"final_answer": final_answer, "history": history}

    def set_sorry_message_node(state: dict) -> dict:
        """Sets a message when max attempts are reached and updates history."""
        question = state["question"]
        columns_str = ', '.join(columns)
        final_answer = (
            f"I'm sorry, but I couldn't find a way to answer your question with the available data. "
            f"The available columns are: {columns_str}. "
            f"Please try rephrasing your question or check if you're querying the correct CSV."
        )
        history = state.get("history", [])
        history.append((question, final_answer))
        print(colored(f"Final Answer: {final_answer}", 'blue', attrs=['bold']))
        return {"final_answer": final_answer, "history": history}

    # Routing functions
    max_attempts = 7



    def route_after_grade(state: dict) -> str:
        """Decides the next step based on grade and attempts."""
        if state["grade"] == "yes":
            attempt_number = state["attempts"]

            return "format_response"
        elif state["attempts"] < max_attempts:
            return "transform_query"
        else:
            return "set_sorry_message"

    def route_after_relevance(state: dict) -> str:
        """Decides the next step based on question relevance."""
        return "generate_query" if state["is_relevant"] else "set_irrelevant_message"

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
    graph.add_node("identify_relevance", identify_relevance_node)  # New node
    graph.add_node("generate_query", generate_query_node)
    graph.add_node("execute_query", execute_query_node)
    graph.add_node("grade_response", grade_response_node)
    graph.add_node("transform_query", transform_query_node)
    graph.add_node("format_response", format_response_node)
    graph.add_node("set_irrelevant_message", set_irrelevant_message_node)  # New node
    graph.add_node("set_sorry_message", set_sorry_message_node)

    # Set entry point
    graph.set_entry_point("interpret_question")

    # Define edges
    graph.add_edge("interpret_question", "identify_relevance")
    graph.add_conditional_edges(
        "identify_relevance",
        route_after_relevance,
        {
            "generate_query": "generate_query",
            "set_irrelevant_message": "set_irrelevant_message",
        }
    )
    graph.add_edge("set_irrelevant_message", END)
    graph.add_edge("generate_query", "execute_query")
    graph.add_edge("execute_query", "grade_response")
    graph.add_conditional_edges(
        "grade_response",
        route_after_grade,
        {
            "format_response": "format_response",
            "transform_query": "transform_query",
            "set_sorry_message": "set_sorry_message",
        }
    )
    graph.add_edge("transform_query", "execute_query")
    graph.add_edge("format_response", END)
    graph.add_edge("set_sorry_message", END)

    # Compile the graph
    app = graph.compile(checkpointer=memory)

    config = {"configurable": {"thread_id": thread_id}}
    initial_state = {"question": user_question, "attempts": 0}
    final_state = app.invoke(initial_state, config=config)
    print(colored(f"Final answer: {final_state['final_answer']}", 'cyan', attrs=['bold']))
    return final_state["final_answer"]

# Example usage
# response = run_csv_chat_agent("/path/to/export22.csv", "Give me all the columns in the dataframe", "123")
# print(colored(response, 'red', attrs=['bold', 'underline']))
