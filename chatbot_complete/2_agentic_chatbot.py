import chainlit as cl
import dotenv
import sniffio
from openai.types.responses import ResponseTextDeltaEvent

dotenv.load_dotenv()

from agents import Runner, SQLiteSession
from nutrition_agent import nutrition_agent

@cl.on_chat_start
async def on_chat_start():
    session = SQLiteSession("conversation_history")
    cl.user_session.set("agentic_session", session)


@cl.on_message
async def on_message(message: cl.Message):
    session = cl.user_session.get("agentic_session")
    msg = cl.Message(content="")

    try:
        result = Runner.run_streamed(
            nutrition_agent,
            message.content,
        )

        async for event in result.stream_events():
            # Stream final message text to screen
            if event.type == "raw_response_event" and isinstance(
                event.data, ResponseTextDeltaEvent
            ):
                await msg.stream_token(token=event.data.delta)

            elif event.type == "raw_response_event":
                item = getattr(event.data, "item", None)
                if (
                    item
                    and getattr(item, "type", None) == "function_call"
                    and len(getattr(item, "arguments", "")) > 0
                ):
                    async with cl.Step(name=item.name, type="tool") as step:
                        step.input = item.arguments

        await msg.update()

    except sniffio.AsyncLibraryNotFoundError:
        # Fallback without streaming if async context detection fails
        result = await Runner.run(nutrition_agent, message.content)
        await cl.Message(content=result.final_output).send()
