# input --> agent1 --> agent1 o/p --> agent2 --> agent2 o/p
# o/p of agent1 will be i/p of agent2 ans so on ...
# each step is perform by the agent

# in this example
# user provides the input (story topic)
# input given to agent1 (generate outline)
# agent1 produces story outline (agent2 output)
# story outline passes through check agent
# check agent check if outline is good quality or not and scifi or not
# if not good quality or not scifi, we stop
# if story outline is good and is scifi then, we move forward
# story outline will be given to agent2 (generate story)
# agent2 will produce story


# story_topic ---> generate_outline_agent ---> story_outline
# story_outline ---> check_story_outline_agent ---> good_quality: true/false, scifi: true/false
# if not good_quality or not scifi then --> we stop here
# else
# story_outline ---> generate_story ---> story


from agents import Agent, Runner
from pydantic import BaseModel
import asyncio, nest_asyncio
from dotenv import load_dotenv
import os

load_dotenv("../.env")
load_dotenv()

# defining model
model = "litellm/gemini/gemini-2.0-flash"
# model = "litellm/groq/llama-3.1-8b-instant"

# outline genrator agent
generate_outline_agent = Agent(
    name = "outline Agent",
    instructions = "Generate the outline based on the user input.",
    model = model
)

# output type of outline checker agent
class OutlineCheckerOutput(BaseModel):
    quality: bool
    scifi: bool

# outline checker agent
check_outline_agent = Agent(
    name = "outline Checker Agent",
    instructions = "Read the story outline and judge the quality also determine if it is scifi or not. good quality mean 'true' and if it is related to scifi the 'true'",
    output_type = OutlineCheckerOutput,
    model = model
)

# story generator agent
generate_story_agent = Agent(
    name = "Story Agent",
    instructions = "Genearte the full story based on the outline. Return the 500 word story (NO PREAMBLE)",
    model = model
)

async def chain(input_prompt, attempt = 0):
    # 
    if attempt >= 3:
        print("Not able to produce good quality and scifi story even after multiple attempts.")
        exit(0)

    # generating outline
    outline = await Runner.run(
        starting_agent=generate_outline_agent,
        input=input_prompt
    )
    print("Outline generated")

    # checking the outlier
    outlier_checker_result = await Runner.run(
        starting_agent=check_outline_agent,
        input=outline.final_output
    )

    # checking for quality and scifi content
    assert isinstance(outlier_checker_result.final_output, OutlineCheckerOutput)
    if not outlier_checker_result.final_output.quality or not outlier_checker_result.final_output.scifi:
        return await chain(input_prompt+" Outlier should be well written (good quality) and must be scifi. NO PREAMBLE", attempt = attempt+1)
    
    # generating story
    story = await Runner.run(
        starting_agent=generate_story_agent,
        input=outline.final_output
    )
    print("Story generated")
    return story.final_output


def main():
    input_prompt = "Generate a well written scifi story for me."
    story = asyncio.run(chain(input_prompt))
    print(f"Story:\n{str(story)}")

if __name__ == "__main__":
    # nest_asyncio.apply()
    # asyncio.run(main())
    main()