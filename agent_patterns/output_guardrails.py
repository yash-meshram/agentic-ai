"""
This example shows how to use output guardrails.

Output guardrails are checks that run on the final output of an agent.
They can be used to do things like:
- Check if the output contains sensitive data
- Check if the output is a valid response to the user's message

In this example, we'll use a (contrived) example where we check if the agent's response contains
a phone number.


input ---> agent ---> response ---> output_guardrails_agent (check for sensitive data) ---> tripped if contains sensitive data (ex.: phone number)
input ---> agent ---> response ---> output_guardrails_agent (check for sensitive data) ---> response
"""

from agents import Agent, Runner, output_guardrail, RunContextWrapper, GuardrailFunctionOutput, OutputGuardrailTripwireTriggered
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import asyncio, json

load_dotenv()

# defining model
model = "litellm/gemini/gemini-2.0-flash"

# building output guardrails 
class MessageOutput(BaseModel):
    resoning: str = Field(description = "Thoughts on how to response to the user's message")
    response: str = Field(description = "The response to the user message")
    user_name: str | None = Field(description = "The name of teh user who sent the message, if known")
    
@output_guardrail
async def check_senstive_data(context: RunContextWrapper, agent: Agent, output: MessageOutput) -> GuardrailFunctionOutput:
    phone_number_in_resoning = "999" in output.resoning
    phone_number_in_response = "999" in output.response
    
    return GuardrailFunctionOutput(
        output_info = {
            "phone_number_in_resoning": phone_number_in_resoning,
            "phone_number_in_response": phone_number_in_response
        },
        tripwire_triggered = phone_number_in_resoning or phone_number_in_response
    )
    
# building agents
assistant_agent = Agent(
    name = "Assistant Agent",
    instructions = "You are a helfuly assistant.",
    output_type = MessageOutput,
    output_guardrails = [check_senstive_data],
    model = model
)

async def main():
    # testing
    test_result = await Runner.run(starting_agent = assistant_agent, input = "WHat is the capital of India?")
    print(f"Passed!\nOutput: {test_result}")
    
    try:
        result = await Runner.run(assistant_agent, "My phone number is 999-8887770. Tell me my phone number and where do I live?")
        
        print(f"Guardrails didnt trip.\nOutput: {json.dumps(result.final_output.model_dump(), indent=4)}")
    
    except OutputGuardrailTripwireTriggered as ex:
        print(f"Guardrails tripped. Info.: {ex.guardrail_result.output.output_info}")
        
if __name__ == "__main__":
    asyncio.run(main())