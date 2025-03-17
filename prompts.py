from langchain_core.prompts import ChatPromptTemplate

# Existing prompts remain unchanged; adding a new one
interpret_question_prompt = ChatPromptTemplate.from_template("""
You are an assistant who is part of a data analysis team. You receive questions directly from the user and your job is to rephrase that user questions based on conversation history to make them standalone 
and clear for data analyzer the data analyze only know to write codes in python pandas to analyze the csv file.
Here is the conversation history:
{history_str}
Current user question: {question}
                                                             
Here is the information about dataset you are working with: {df_info}
                                                             
Please take your time to think deeply and analyze how exactly user's query can be answered what are the key relationships and rephrase the question and provide the details necessary.


Rephrase the current question to be standalone, incorporating any necessary context from the history, such as specific dates, months, product names, or other details that the question might be referring to.
For example:
- If the history mentions "sales in January" and the current question is "what about February," rephrase it to "what are the sales in February."
- If the history has a response like "The total sales are $1000," and the current question is "what is that," rephrase it to "what is the total sales value of $1000 referring to."
- If the question is complex please provide the steps to analyze that. Like "calculate the price of the following products:
Fiordilatte di Agerola - Mozzarella campana dei Monti Lattari | Ruocco - 800 gr - Formato 800 gr (x5) Filetti di Alici di Cetara Piccanti - Sott'olio e Peperoncino | Nettuno - 420 gr - Formato 420 gr Burro fresco di Agerola - Per colazione | Ruocco - 250 gr - Formato 250 gr
" rephrase it smartly.
If the question is already clear without additional context, return it unchanged.
Only provide the rephrased question.
""")

generate_query_prompt = ChatPromptTemplate.from_template("""
You are a data analysis expert with access to a pandas dataframe `df`.
Here is information about the dataframe:
{df_info}
Given the question: {standalone_question}
                                                         
Please observe the dataframe and question and take your time to think and plan how you are going to get this analysis and then write codes by thinking step by step.
Generate a Python code snippet using pandas to answer the question. The DataFrame is named 'df'.
Only provide the code, no explanation.
""")

grade_response_prompt = ChatPromptTemplate.from_template("""
You are an expert grader working with the team of data analyst. The question is provided by the user and the response is 
provided by your team you need to look to make sure that the response is the correct answer of question.
Question: {standalone_question}
Response: {response}
Please take your time to think and analyze if the response is the relevant answer of question.
Does the response answer the question? Answer 'yes' or 'no'.
""")

transform_query_prompt = ChatPromptTemplate.from_template("""
You are a data analysis expert with access to a pandas dataframe `df`.
Here is information about the dataframe:
{df_info}
Original question: {standalone_question}
Previous query: {previous_query}
Previous response: {previous_response}
The previous response did not adequately answer the question.
Please think deeply and observe why it might now be able to answer previously. And then plan how you are going to fix that. Then write codes by thinking step by step.
Generate a new Python code snippet using pandas to better answer the question.
Only provide the code, no explanation.
""")

format_response_prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant that formats responses based on user questions and data from a CSV file.
Given the user's question and the raw response from the data, provide a concise and helpful answer.
If the raw response is a single value, present it clearly with context.
For complex analyses, provide insights based on the data.
Note: See the language of dataframe and show the currency values accordingly.
First 5 rows of the dataframe:
{df_info}
User Question: {question}
Raw Response: {response}
Formatted Answer:
""")

# New prompt for relevance checking
relevance_check_prompt = ChatPromptTemplate.from_template("""
Given the following question and the list of columns in a DataFrame, and chat history determine if the question can be answered using the data in the DataFrame.

Question: {standalone_question}
Columns: {columns}
Here is the conversation history:
{history_str}
Answer 'yes' if the question can be answered with the available columns or if it has correlation with history, otherwise answer 'no'.
""")
