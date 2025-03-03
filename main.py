from langchain.graphs import Neo4jGraph
import dotenv
from langchain.chains import GraphCypherQAChain
from langchain_google_genai import ChatGoogleGenerativeAI
import os
dotenv.load_dotenv()


GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = 'neo4j'


def generate(user_input):
    graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD, database=NEO4J_DATABASE)

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0,
        max_tokens=None,
        timeout=None,
        api_key=GOOGLE_API_KEY
    )
    chain = GraphCypherQAChain.from_llm(graph=graph, llm=llm, allow_dangerous_requests=True)
    response = chain(user_input)
    return response.get('result') or response.get('answer') or response


