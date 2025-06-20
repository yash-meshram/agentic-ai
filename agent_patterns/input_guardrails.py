'''
This example shows how to use guardrails.

Guardrails are checks that run in parallel to the agent's execution.
They can be used to do things like:
- Check if input messages are off-topic
- Check that input messages don't violate any policies
- Take over control of the agent's execution if an unexpected input is detected

In this example, we'll setup an input guardrail that trips if the user is asking to do math homework.
If the guardrail trips, we'll respond with a refusal message.


user ---> input ---> math_guardrail_agent ---> guardrail_output

guardrail_output ---> (if tripwired triggered) message like = cants help you with math problem

guardrail_output ---> (if tripwored not triggered) customer_support_agent ---> output ---> user
'''

from agents import (Agent, GuardrailFunctionOutput, InputGuardrailTripwireTriggered, Runner, TResponseInputItem, input_guardrail, RunContextWrapper)
from pydantic import BaseModel
from dotenv import load_dotenv
import asyncio

load_dotenv()

# defining model
model = "litellm/gemini/gemini-2.0-flash"

# building guardrail agent
class MathGuardrailOutput(BaseModel):
    reasoning: str
    is_math_homework: bool

math_guardrail_agent = Agent(
    name = "Math Guardrail Agent",
    instructions = "Check if user is asking you to solve the math problem.",
    model = model,
    output_type = MathGuardrailOutput
)

@input_guardrail
async def math_guardrail(context: RunContextWrapper[None], agent: Agent, input: str | list[TResponseInputItem]) -> GuardrailFunctionOutput:
    result = await Runner.run(
        math_guardrail_agent,
        input,
        context = context.context
    )
    final_output = result.final_output_as(MathGuardrailOutput)
    
    return GuardrailFunctionOutput(
        output_info = final_output,
        tripwire_triggered = final_output.is_math_homework
    )
    

    
async def main():
    # building customer support agent
    customer_support_agent = Agent(
        name = "Customer Support Agent",
        instructions = "You are the customer support agent, you help the user with their queries. The user should not know that you are an AI.",
        input_guardrails = [math_guardrail],
        model = model
    )
    input_data: list[TResponseInputItem] = []
    
    while True:
        user_input = input("Enter your query: ")
        input_data.append(
            {
                'role': 'user',
                'content': user_input
            }
        )
        
        try:
            result = await Runner.run(
                starting_agent = customer_support_agent,
                input = input_data
            )
            print(result.final_output)
            input_data = result.to_input_list()
            
        except InputGuardrailTripwireTriggered:
            message = "Do I look like an math solving machine to you."
            print(message)
            input_data.append(
                {
                    'role': 'assistant',
                    'content': message
                }
            )
    
if __name__ == '__main__':
    asyncio.run(main())