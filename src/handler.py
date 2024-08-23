import asyncio
import runpod
import os
from sg_engine import SGlangEngine, OpenAIRequest

# Initialize the engine
engine = SGlangEngine()
engine.start_server() 
engine.wait_for_server()

print(f" ==== start_server")

async def async_handler(job):
    """Handle the requests asynchronously."""
    job_input = job["input"]
    print(f"JOB_INPUT: {job_input}")
    
    if job_input.get("openai_route"):
        openai_route, openai_input = job_input.get("openai_route"), job_input.get("openai_input")
        async with OpenAIRequest() as openai_request:
            if openai_route == "/v1/chat/completions":
                async for chunk in openai_request.request_chat_completions(**openai_input):
                    yield chunk
            elif openai_route == "/v1/completions":
                async for chunk in openai_request.request_completions(**openai_input):
                    yield chunk
            elif openai_route == "/v1/models":
                models = await openai_request.get_models()
                yield models
    else:
        yield {"error": "Unsupported request type", "details": "Only OpenAI-compatible routes are supported"}

runpod.serverless.start({"handler": async_handler, "return_aggregate_stream": True})

# Ensure the server is shut down when the serverless function is terminated
import atexit
atexit.register(engine.shutdown)
