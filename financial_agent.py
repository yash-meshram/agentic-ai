# from phi.agent import Agent
# from phi.model.groq import Groq
from agno.agent import Agent
from agno.models.groq import Groq
from agno.tools.yfinance import YFinanceTools
from agno.tools.duckduckgo import DuckDuckGoTools
from dotenv import load_dotenv

load_dotenv()

# Agent 1: Web Search Agent
web_search_agent = Agent(
    name = "Web search agent",
    role = "Search the web for the information",
    model = Groq(id = "llama-3.3-70b-versatile"),
    tools = [DuckDuckGoTools],
    instructions = ["Always include sources"],
    show_tool_calls = True,
    markdown = True
)

# Agent 2: Financial Agent
finance_agent = Agent(
    name = "Finance AI Agent",
    model = Groq(id = "llama-3.3-70b-versatile"),
    tools = [
        YFinanceTools(
            stock_price = True,
            analyst_recommendations = True,
            stock_fundamentals = True,
            company_news = True
        )
    ],
    instructions = ["Use tables to display the data"],
    show_tool_calls = True,
    markdown = True,
)

# Multi AI Agent
multi_ai_agent = Agent(
    team = [
        web_search_agent,
        finance_agent
    ],
    instructions = [
        "Always include sources",
        "Show data in the table format"
    ],
    show_tool_calls = True,
    markdown = True,
    model = Groq(id = "llama-3.3-70b-versatile")
)

# response
multi_ai_agent.print_response(
    message = "Summarize analyst recommendation and share the latest news for NVIDIA stock. ALso analyze past 2 year data of the stock and give me the prediction of next 5 day from todays also tell me which model you used to predict these values",
    stream = True
)