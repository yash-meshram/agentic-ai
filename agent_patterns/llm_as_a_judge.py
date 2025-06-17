'''
input ---> story_outline_generator_agent (generate the story outline based on the input) ---> outline

outliner ---> evaluator_agent (evaluate/judge whether the outline. give feedback and score('pass', 'need_improvement', 'fail'))

if score == 'pass' ---> display outlier
else ---> story_generator_outlier (with user input and feedback)
'''

from agents import Agent, Runner, TResponseInputItem, ItemHelpers, trace
from dataclasses import dataclass
from dotenv import load_dotenv
from typing import Literal
import asyncio, nest_asyncio

load_dotenv()

# defining model
model = "litellm/gemini/gemini-2.0-flash"

# building agents
story_outlier_generator = Agent(
    name = "Story Outlier Generator",
    instructions = (
        "Use users input to generate a short story outline."
        "If the feedback is provided use it to improve the story outline."
        "Your response should always be a complete story outline. All the details should be there in your response. (NO PREAMBLE)"
    ),
    model = model
)

# building judge agent
@dataclass
class EvalationFeedback:
    feedback: str
    score: Literal["pass", "need-improvement", "fail"]

evaluator_agent = Agent(
    name = "Judge Agent",
    instructions = (
        "You evaluate the story outline and judge if its good enough."
        "If it is not good enough you provide feedback on what need to be improved."
        "Never give pass on the first try."
    ),
    output_type = EvalationFeedback,
    model = model
)

async def main():
    # message = input("WHat kind of story you want:")
    message = "give me a story about an indian boy"
    input_items: list[TResponseInputItem] = [{'content': message, 'role': 'user'}]
    
    latest_outline: str | None = None
    
    with trace("LLM as judge"):
        while True:
            print("Generating story outlier...")
            outline = await Runner.run(story_outlier_generator, input_items)
        
            input_items = outline.to_input_list()
            latest_outline = ItemHelpers.text_message_outputs(outline.new_items)
            print("Story outlier generated!")
            
            print("Evaluating genearted story outlier...")
            evaluator_result = await Runner.run(evaluator_agent, input_items)
            result: EvalationFeedback = evaluator_result.final_output
            
            print(f"Evaluation test result: {result.score}")
            if result.score == 'pass':
                break
            
            print("Re-generating story outlier with feedback..")
            
            input_items.append({'content': f"Feedback: {result.feedback}", 'role': 'user'})
        
        print(f"\nStory:\n{latest_outline}")            
    
if __name__ == "__main__":
    nest_asyncio.apply()
    asyncio.run(main())