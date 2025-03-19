from langchain_core.prompts import ChatPromptTemplate
import pandas as pd
import numpy as np

# Existing prompts remain unchanged; adding a new one
interpret_question_prompt = ChatPromptTemplate.from_template("""
You are an assistant who is part of a data analysis team. You can speak english and italian fluently. You receive questions directly from the user and your job is to rephrase that user questions based on conversation history to make them standalone 
and clear for data analyzer the data analyze only know to write codes in python pandas to analyze the csv file.
Here is the conversation history:
{history_str}
Current user question: {question}
                                                             
Here is the information about dataset you are working with: {df_info}

Here is the information about CSV data: {csv_description}                                  
Please take your time to think deeply and analyze how exactly user's query can be answered what are the key relationships and rephrase the question and provide the details necessary.


Rephrase the current question to be standalone, incorporating any necessary context from the history, such as specific dates, months, product names, or other details that the question might be referring to.
For example:
- If the history mentions "sales in January" and the current question is "what about February," rephrase it to "what are the sales in February."
- If the history has a response like "The total sales are $1000," and the current question is "what is that," rephrase it to "what is the total sales value of $1000 referring to."
- If the question is complex please provide the steps to analyze that. Like "calculate the price of the following products:
Fiordilatte di Agerola - Mozzarella campana dei Monti Lattari | Ruocco - 800 gr - Formato 800 gr (x5) Filetti di Alici di Cetara Piccanti - Sott'olio e Peperoncino | Nettuno - 420 gr - Formato 420 gr Burro fresco di Agerola - Per colazione | Ruocco - 250 gr - Formato 250 gr
" rephrase it smartly.
If the question is already clear without additional context, return it unchanged.
Only provide the rephrased question. DO NOT provide details about dataframe.
""")

generate_query_prompt = ChatPromptTemplate.from_template("""
You can speak english and italian fluently.
You are a data analysis expert with access to a pandas dataframe `df`.
Here is information about the dataframe:
{df_info}
Given the question: {standalone_question}
                                                         
Here is the information about CSV data: {csv_description}                                  
                                                         
Please observe the dataframe and question and take your time to think and plan how you are going to get this analysis and then write codes by thinking step by step.
Generate a Python code snippet using pandas to answer the question. The DataFrame is named 'df'.
Only provide the code, no explanation.
""")



format_response_prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant that formats responses based on user questions and data from a CSV file. You can speak english and italian fluently.
Given the user's question and the raw response from the data, provide a concise and helpful answer.
If the raw response is a single value, present it clearly with context.
For complex analyses, provide insights based on the data.
Note: See the language of dataframe and show the currency values accordingly.
First 5 rows of the dataframe:
{df_info}
User Question: {question}
Here is the information about CSV data: {csv_description}                                  
Raw Response: {response}
Formatted Answer:
""")
