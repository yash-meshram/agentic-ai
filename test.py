from agno.agent import Agent, RunResponse
from agno.models.groq import Groq

agent = Agent(
    model=Groq(id="llama-3.3-70b-versatile"),
    markdown=True
)

# Print the response in the terminal
agent.print_response("Share a 2 sentence horror story.")