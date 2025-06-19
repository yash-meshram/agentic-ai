'''
Here we run the agent parallelly
'''

from agents import Agent, Runner, trace, ItemHelpers
import asyncio, nest_asyncio
from dotenv import load_dotenv

load_dotenv()

# defining model
model = "litellm/gemini/gemini-2.0-flash"

# defining agent
hindi_agent = Agent(
    name = "Hind Agent",
    instructions = "You translate the user message to hindi.",
    model=model
)

translation_picker = Agent(
    name = "Translation Picker Agent",
    instructions = "You pick and return the best translation from the given options. (NO PREAMBLE)",
    model=model
)

async def main():
    input_message = input("Enter message we translate it to hindi:")
    
    with trace("Parallel translation"):
        result_1, result_2, result_3 = await asyncio.gather(
            Runner.run(hindi_agent, input_message),
            Runner.run(hindi_agent, input_message),
            Runner.run(hindi_agent, input_message)
        )
        
        outputs = [
            ItemHelpers.text_message_outputs(result_1.new_items),
            ItemHelpers.text_message_outputs(result_2.new_items),
            ItemHelpers.text_message_outputs(result_3.new_items)
        ]
        
        translations = "\n\n".join(outputs)
        print(f"\nTranslations:\n{translations}")
        
        best_translation = await Runner.run(
            translation_picker,
            input = f"Input: {input_message} \n\nTranslations:\n{translations}"
        )
        
    print(f"Translation in hindi = {best_translation.final_output}")
    
if __name__ == "__main__":
    asyncio.run(main())