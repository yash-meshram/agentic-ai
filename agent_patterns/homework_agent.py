from agents import Agent, Runner, GuardrailFunctionOutput, InputGuardrail
from pydantic import BaseModel
from langchain_google_genai import GoogleGenerativeAI
import asyncio, nest_asyncio
from dotenv import load_dotenv
import os

load_dotenv("../.env")

# defining model
model = "litellm/gemini/gemini-2.0-flash"
# model = "litellm/groq/llama-3.1-8b-instant"

# Building agents
math_agent = Agent(
    name="Math Agent",
    handoff_description="Specialist agent for Math questions",
    instructions="You provide help with the math problem. Provide your resoning at each step and provide examples.",
    model = model
)

history_agent = Agent(
    name = "History Agent",
    handoff_description="Specialist agent for History questions",
    instructions="You provide help with historical queries. Explain important event and context clearly.",
    model = model
)

# Building guardrails - which will decide if the process should move forward or not
# This will decided if the input provided by user is related to homework or not
class HomeworkOutput(BaseModel):
    is_homework: bool
    reasoning: str

guardrial_agent = Agent(
    name="Guardrail Check Agent",
    instructions="Check if the user is asking out the homework.",
    model = model,
    output_type=HomeworkOutput
)

async def guardrail_homework(ctx, agent, input_data):
    result = await Runner.run(
        starting_agent = guardrial_agent,
        input = input_data,
        context = ctx.context
    )
    final_output = result.final_output_as(HomeworkOutput)
    return GuardrailFunctionOutput(
        output_info = final_output,
        tripwire_triggered = not final_output.is_homework
    )


# Building agent - which will decide, which agent to use
triage_agent = Agent(
    name="Triage Agent",
    instructions="You determine which agent to use based on user homework question.",
    model = model,
    handoffs=[math_agent, history_agent],
    input_guardrails = [
        InputGuardrail(guardrail_function = guardrail_homework)
    ]
)


# running
async def main():
    result = await Runner.run(
        starting_agent=triage_agent,
        input="homwork question: what is quadratic functions"
    )
    print(result)
    print("----------------------------------------")
    print(result.final_output)
    
if __name__ == "__main__":
    nest_asyncio.apply()
    asyncio.run(main())
