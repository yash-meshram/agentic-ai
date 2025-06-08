import typer
from typing import Optional,List
from phi.agent import Agent
from phi.model.groq import Groq
from phi.assistant import Assistant
from phi.storage.agent.postgres import PgAgentStorage
from phi.knowledge.pdf import PDFUrlKnowledgeBase
from phi.vectordb.pgvector import PgVector2
from phi.embedder.google import GeminiEmbedder

import os
from dotenv import load_dotenv
load_dotenv()


os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Ensure that API keys are set properly
if not os.environ["GROQ_API_KEY"] or not os.environ["GOOGLE_API_KEY"]:
    raise ValueError("Missing API Keys! Check your .env file.")

db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"

knowledge_base=PDFUrlKnowledgeBase(
    urls=["https://phi-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
    vector_db=PgVector2(collection="dish",db_url=db_url,embedder=GeminiEmbedder()),
    chunk=False
)

knowledge_base.load()

storage = PgAgentStorage(
    table_name="pdf_assistant",
    db_url=db_url,
)
def pdf_assistant(new: bool = False, user: str = "user"):
    run_id: Optional[str] = None

    assistant = Agent(
        model=Groq(id="llama-3.3-70b-versatile", embedder=GeminiEmbedder()),
        run_id=run_id,
        user_id=user,
        knowledge_base=knowledge_base,
        storage=storage,
        show_tool_calls=True,
        search_knowledge=True,
        read_chat_history=True,
    )

    if run_id is None:
        run_id = assistant.run_id
        print(f"Started Run: {run_id}\n")
    else:
        print(f"Continuing Run: {run_id}\n")

    assistant.cli_app(markdown=True)

if __name__ == "__main__":
    typer.run(pdf_assistant)