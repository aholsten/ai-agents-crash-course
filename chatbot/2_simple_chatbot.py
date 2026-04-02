import asyncio

import chainlit as cl
import dotenv
from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError

dotenv.load_dotenv()

from agents import Runner
from nutrition_agent import nutrition_agent


@cl.on_message
async def on_message(message: cl.Message):
    max_attempts = 4
    backoff_seconds = 1

    for attempt in range(1, max_attempts + 1):
        try:
            result = await Runner.run(nutrition_agent, message.content)
            await cl.Message(content=f"Result: {result.final_output}").send()
            return
        except (InternalServerError, RateLimitError, APIConnectionError, APITimeoutError):
            if attempt == max_attempts:
                await cl.Message(
                    content="OpenAI ist gerade überlastet. Bitte in ein paar Sekunden erneut versuchen."
                ).send()
                return

            await cl.sleep(backoff_seconds)
            backoff_seconds *= 2
    

    