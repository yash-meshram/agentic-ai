# input ---> orchestrat_agent (will use agent as tool, can call multiple agents at a time parallelly or sequentically)
# orchestrat_agent (after using agents as tool willl contsruct an response) ---> response
# response ---> synthesizer_agent (review response, correctif needed and constract proper response)

'''
# Agent as agent v/s Agent as tool

## Key Differences Summary
Feature             Routing Agent Pattern	                    Agents-as-Tools Pattern
------------------------------------------------------------------------------------------------------
Agent Invocation	Chooses one agent to handle input	        Orchestrator agent calls others as tools
Composition	        One agent per input	                        Multiple agents can be used in one step
Control Flow	    Hard-coded or learned routing logic	        Tool-using agent controls the flow
Use Case Fit	    Best for simple decision-based delegation	Best for complex, multi-step, composable tasks
Agent Autonomy	    Full autonomy per agent	                    Tool agents are used by a controlling agent

## When to Use Which?
Routing Agent Pattern:
If each user query clearly maps to only one agent/task.
If agents should operate independently or asynchronously.

Agents-as-Tools Pattern:
If you need agent collaboration, tool use, or workflow composition.
If you want LLM reasoning to drive tool/agent selection dynamically.
'''


from agents import Agent, Runner, trace, MessageOutputItem, ItemHelpers
import asyncio

# defining model
model = "litellm/gemini/gemini-2.0-flash"

# building agents
spanish_agent = Agent(
    name = "Spanish Translator Agent",
    instructions = "You translate the user message to Spanish",
    handoff_description = "English to Spanish translator",
    model = model
)
hindi_agent = Agent(
    name = "Hindi Translator Agent",
    instructions = "You translate the user message to Hindi",
    handoff_description = "English to Hindi translator",
    model = model
)
tamil_agent = Agent(
    name = "Tamil Translator Agent",
    instructions = "You translate the user message to Tamil",
    handoff_description = "English to Tamil translator",
    model = model
)

orchestrator_agent = Agent(
    name = "Orchestrator Agent",
    instructions = (
        "You are the translator agents. You use the tools given to you to translate."
        "If asked for multiple translations, you call the relevent tools in order."
        "You never translate on your own, ou always use provided tools."
    ),
    tools = [
        spanish_agent.as_tool(
            tool_name = "translate_to_spanish",
            tool_description = "Translate the users message to Spanish"
        ),
        hindi_agent.as_tool(
            tool_name = "translate_to_hindi",
            tool_description = "Translate the users message to Hindi"
        ),
        tamil_agent.as_tool(
            tool_name = "translate_to_tamil",
            tool_description = "Translate the users message to Tamil"
        )
    ],
    model = model
)

synthesizer_agent = Agent(
    name = "Synthesizer Agent",
    instructions = "You inspect translations, correct them if needed, and produce a concatenated user friendly response which include the translation. (NO PREAMBLE)",
    model = model
)

async def main():
    message = input("What you need to translate and to which language?:")
    
    with trace("Orchestrator Evaluator"):
        orchestrator_result = await Runner.run(orchestrator_agent, message)
                
        for item in orchestrator_result.new_items:
            if isinstance(item, MessageOutputItem):
                text = ItemHelpers.text_message_output(item)
                if text:
                    print(f"- translation step:\n{text}")
                    
        synthesizer_output = await Runner.run(synthesizer_agent, orchestrator_result.final_output)
        
    print(f"\nFinal Response:\n{synthesizer_output.final_output}")
    
if __name__ == "__main__":
    asyncio.run(main())