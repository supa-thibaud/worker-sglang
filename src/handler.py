import asyncio
import runpod
import os
from engine import SGlangEngine, OpenAIRequest

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
        openai_request = OpenAIRequest()
        
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
        async for result in engine.generate(job_input):
            yield result

max_concurrency = int(os.getenv("MAX_CONCURRENCY", 100))
print(f"MAX_CONCURRENCY {max_concurrency}")

runpod.serverless.start({
    "handler": async_handler, 
    "concurrency_modifier": lambda x: max_concurrency, 
    "return_aggregate_stream": True
})