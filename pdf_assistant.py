import typer
from typing import Optional, List
# independent autonomous ai which will be able to assist for different different task
from phi.assistant import Assistant
# Postgres storage
from agno.storage.agent.postgres import PostgresAgentStorage
# to read the content of the pdf
from agno.knowledge.pdf_url import PDFUrlKnowledgeBase
# vector database
from agno.vectordb.pgvector import PgVector
from agno.storage.postgres import PostgresStorage
from agno.agent import Agent
import os
from dotenv import load_dotenv
import asyncio
from agno.models.groq import Groq
from agno.models.google import Gemini
from phi.embedder.google import GeminiEmbedder

load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"

# # Create a storage backend using the Postgres database
# storage = PostgresStorage(
#     # store sessions in the ai.sessions table
#     table_name="agent_sessions",
#     # db_url: Postgres database URL
#     db_url=db_url,
# )

# # Add storage to the Agent
# agent = Agent(storage=storage)

knowledge_base = PDFUrlKnowledgeBase(
    urls = ["https://agno-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
    vector_db = PgVector(
        db_url = db_url,
        table_name = "recipes3",
        embedder = GeminiEmbedder()
    )
)

# knowledge_base.load()

storage = PostgresAgentStorage(
    table_name="pdf_assistant3",
    db_url=db_url,
)

def pdf_assistant(new: bool = False, user: str = "User"):    
    agent = Agent(
        model = Gemini(id = "gemini-2.0-flash"),
        knowledge = knowledge_base,
        storage=storage,
        search_knowledge = True,
        user_id = user,
        show_tool_calls = True,
        read_chat_history = True
    )
            
    agent.cli_app(markdown = True)
    
if __name__ == "__main__":
    typer.run(pdf_assistant)

    # agent = Agent(
    #     model = Gemini(id = "gemini-2.0-flash"),
    #     knowledge = knowledge_base,
    #     storage=storage,
    #     search_knowledge = True,
    #     user_id = "user",
    #     show_tool_calls = True,
    #     read_chat_history = True
    #     # instructions = [
    #     #     "Search the knowledge base for information"
    #     # ]
    # )

    # agent.knowledge.load()
    
    # agent.print_response("Named all the dishes")

# if __name__ == "__main__":
#     # run only onces
#     asyncio.run(agent.knowledge.aload(recreate = False))

#     asyncio.run(agent.aprint_response("Ask me about something from the knowledge base", markdown = True))Khao Niew Dam Piek Maphrao Awn 