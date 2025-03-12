import os
import pandas as pd
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.tools import PythonAstREPLTool
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from typing import TypedDict, List
from langgraph.checkpoint.memory import MemorySaver

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

# Create a shared MemorySaver instance at the module level
memory = MemorySaver()

def run_csv_chat_agent(file_path: str, user_question: str, thread_id: str) -> str:
    """
    Runs a chat agent that processes a user question against a CSV file, formats the final response,
    and maintains conversation history for context-aware responses.

    Args:
        file_path (str): Path to the CSV or Excel file.
        user_question (str): The question asked by the user.
        thread_id (str): Identifier for the conversation thread to maintain state.

    Returns:
        str: The final formatted answer to the user’s question.
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

    # Prepare dataframe info for prompts
    df_info = f"""
    DataFrame Information:
    - Shape: {df.shape}
    - Columns: {', '.join(df.columns.tolist())}
    - Data Types:
    {df.dtypes.to_string()}
    First 5 rows:
    {df.head().to_markdown()}
    """

    # Define prompts
    interpret_question_prompt = ChatPromptTemplate.from_template("""
    You are an assistant that rephrases user questions based on conversation history to make them standalone and clear for data analysis.

    Here is the conversation history:
    {history_str}

    Current user question: {question}

    Rephrase the current question to be standalone, incorporating any necessary context from the history, such as specific dates, months, product names, or other details that the question might be referring to.
    For example:
    - If the history mentions "sales in January" and the current question is "what about February," rephrase it to "what are the sales in February."
    - If the history has a response like "The total sales are $1000," and the current question is "what is that," rephrase it to "what is the total sales value of $1000 referring to."
    If the question is already clear without additional context, return it unchanged.
    Only provide the rephrased question.
    """)

    generate_query_prompt = ChatPromptTemplate.from_template("""
    You are a data analysis expert with access to a pandas dataframe `df`.

    Here is information about the dataframe:
    {df_info}

    Given the question: {standalone_question}

    Generate a Python code snippet using pandas to answer the question. The DataFrame is named 'df'.
    Only provide the code, no explanation.
    """)

    transform_query_prompt = ChatPromptTemplate.from_template("""
    You are a data analysis expert with access to a pandas dataframe `df`.

    Here is information about the dataframe:
    {df_info}

    Original question: {standalone_question}

    Previous query: {previous_query}

    Previous response: {previous_response}

    The previous response did not adequately answer the question.
    Generate a new Python code snippet using pandas to better answer the question.
    Only provide the code, no explanation.
    """)

    grade_response_prompt = ChatPromptTemplate.from_template("""
    Question: {standalone_question}

    Response: {response}

    Does the response adequately answer the question? Answer 'yes' or 'no'.
    """)

    format_response_prompt = ChatPromptTemplate.from_template("""
    You are a helpful assistant that formats responses based on user questions and data from a CSV file.

    Given the user's question and the raw response from the data, provide a concise and helpful answer.

    If the raw response is a single value, present it clearly with context.
    If it's a list or table, summarize or format it appropriately.
    For complex analyses, provide insights based on the data.
    Note: See the language of dataframe and show the currency values accordingly.

    First 5 rows of the dataframe:
    {df_info}

    User Question: {question}

    Raw Response: {response}

    Formatted Answer:
    """)

    # Define node functions
    def interpret_question_node(state: dict) -> dict:
        """Rephrases the user's question based on conversation history."""
        question = state["question"]
        history = state.get("history", [])
        history_str = "\n".join([f"User: {q}\nAssistant: {a}" for q, a in history[-5:]])  # Limit to last 5
        prompt = interpret_question_prompt.format(history_str=history_str, question=question)
        response = llm.invoke(prompt)
        standalone_question = response.content.strip()
        return {"standalone_question": standalone_question}

    def generate_query_node(state: dict) -> dict:
        """Generates the initial Python query based on the standalone question."""
        standalone_question = state["standalone_question"]
        prompt = generate_query_prompt.format(df_info=df_info, standalone_question=standalone_question)
        response = llm.invoke(prompt)
        query = response.content.strip()
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
        standalone_question = state["standalone_question"]
        response = state["response"]
        prompt = grade_response_prompt.format(standalone_question=standalone_question, response=response)
        grade_response = llm.invoke(prompt).content.strip().lower()
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
            previous_response=previous_response
        )
        response = llm.invoke(prompt)
        new_query = response.content.strip()
        return {"query": new_query}

    def format_response_node(state: dict) -> dict:
        """Formats the raw response and updates history."""
        question = state["question"]
        response = state["response"]
        prompt = format_response_prompt.format(question=question, response=response, df_info=df_5_rows)
        formatted_answer = llm.invoke(prompt).content.strip()
        history = state.get("history", [])
        history.append((question, formatted_answer))
        return {"final_answer": formatted_answer, "history": history}

    def set_sorry_message_node(state: dict) -> dict:
        """Sets a default message when max attempts are reached and updates history."""
        question = state["question"]
        final_answer = "Sorry, I couldn’t find an exact answer."
        history = state.get("history", [])
        history.append((question, final_answer))
        return {"final_answer": final_answer, "history": history}

    # Define routing function
    max_attempts = 3
    def route_after_grade(state: dict) -> str:
        """Decides the next step based on grade and attempts."""
        if state["grade"] == "yes":
            return "format_response"
        elif state["attempts"] < max_attempts:
            return "transform_query"
        else:
            return "set_sorry_message"

    # Define the state structure
    class GraphState(TypedDict):
        history: List[tuple]
        question: str
        standalone_question: str
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
    graph.add_node("grade_response", grade_response_node)
    graph.add_node("transform_query", transform_query_node)
    graph.add_node("format_response", format_response_node)
    graph.add_node("set_sorry_message", set_sorry_message_node)

    # Set entry point
    graph.set_entry_point("interpret_question")

    # Define edges
    graph.add_edge("interpret_question", "generate_query")
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

    # Compile the graph with the shared MemorySaver
    app = graph.compile(checkpointer=memory)

    # Run the graph
    config = {"configurable": {"thread_id": thread_id}}
    initial_state = {"question": user_question, "attempts": 0}
    final_state = app.invoke(initial_state, config=config)
    return final_state["final_answer"]
