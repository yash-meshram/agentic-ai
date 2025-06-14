# input ---> triage_agent (this agent will choose the which agentto use accordingly)
# 
#               ---> english_agent
#              /
# triage_agent ---> hindi_agent
#              \
#               ---> tamil_agent

from agents import Agent, Runner, TResponseInputItem, RawResponsesStreamEvent, trace
from openai.types.responses import ResponseTextDeltaEvent, ResponseContentPartDoneEvent
import uuid
import asyncio

# building agents
english_agent = Agent(
    name="English Agent",
    instructions="You only speak English"
)

hindi_agent = Agent(
    name="Hindi Agent",
    instructions="You only speak Hindi"
)

tamil_agent = Agent(
    name="Tamil Agent",
    instructions="You only speak Tamil"
)

# Generating Triage agent which will decide which agent to use
triage_agent = Agent(
    name="Triage Agent",
    instructions="Based on the request select the appropriate agent."
)

async def main():
    conversation_id = str(uuid.uuid4().hex[:16])

    message = "Hi, we speak, English, Hindi and Tamil. How can we help you?"
    agent = triage_agent()
    inputs: list[TResponseInputItem] = [{"content": message, "role": "user"}]

    while True:
        with trace("Routing example", group_id = conversation_id):
            result = await Runner.run(
                agent,
                input = inputs
            )

            # Will print response (result) token by token
            async for event in result.stream_event():
                if not isinstance(event, RawResponsesStreamEvent):
                    continue    # ignore other response event but continue the loop
                data = event.data
                if isinstance(data, ResponseTextDeltaEvent):
                    print(data.delta, end="", flush=True)
                    # This is where text streaming happens.
                    # ResponseTextDeltaEvent contains partial chunks of text as the model generates them.
                    # These are printed immediately, without a newline (end="") so it appears as a live typing effect.
                    # flush=True ensures it prints instantly to the terminal.
                if isinstance(data, ResponseContentPartDoneEvent):
                    print("\n")
                    # This will end the response

            inputs = result.to_input_list()
            print("\n")

            user_message = input("Enter a message:")                            # user can enter new message (conversation)
            inputs.append({"content": user_message, "role": "user"})            # adding the new user input message to inputs which will be pass to agent
            agent = result.current_agent                                        # setting current agent as agent (rethink on it)

if __name__ == "__main__":
    asyncio.run(main())