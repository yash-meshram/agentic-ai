from agno.agent import Agent
import agno.api
from agno.models.groq import Groq
from agno.models.google import Gemini
from agno.tools.yfinance import YFinanceTools
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.playground import Playground, serve_playground_app
import agno
from dotenv import load_dotenv
import os

load_dotenv()

agno.api = os.getenv("AGNO_API_KEY")

# Agent 1: Web Search Agent
web_search_agent = Agent(
    name = "Web search agent",
    role = "Search the web for the information",
    model = Groq(id = "llama-3.3-70b-versatile"),
    tools = [DuckDuckGoTools()],
    instructions = ["Always include sources"],
    show_tool_calls = True,
    markdown = True
)

# Agent 2: Financial Agent
# finance_agent = Agent(
#     name = "Finance AI Agent",
#     model = Groq(id = "llama-3.3-70b-versatile"),
#     tools = [YFinanceTools(stock_price = True, analyst_recommendations = True, stock_fundamentals = True, company_news = True)],
#     instructions = ["Use tables to display the data"],
#     show_tool_calls = True,
#     markdown = True
# )
finance_agent=Agent(
    name="Finance AI Agent",
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[
        YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True,
                      company_news=True),
    ],
    instructions=["Use tables to display the data"],
    show_tool_calls=True,
    markdown=True,
)

# building app
app = Playground(agents = [web_search_agent, finance_agent]).get_app()

if __name__ == "__main__":
    serve_playground_app("playground:app", reload = True)