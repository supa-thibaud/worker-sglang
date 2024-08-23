import aiohttp
import asyncio
import json

class OpenAIAsync:
    def __init__(self, base_url="http://0.0.0.0:30000/v1", api_key="EMPTY"):
        self.base_url = base_url
        self.api_key = api_key
        self.session = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.session.close()

    async def request_chat_completions(self, model="default", messages=None, max_tokens=100, stream=False, **kwargs):
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": stream,
            **kwargs
        }
        async with self.session.post(url, json=payload) as response:
            if stream:
                async for line in response.content:
                    if line.strip():
                        yield json.loads(line)
            else:
                yield await response.json()

    async def request_completions(self, model="default", prompt="The capital of France is", max_tokens=100, stream=False, **kwargs):
        url = f"{self.base_url}/completions"
        payload = {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "stream": stream,
            **kwargs
        }
        async with self.session.post(url, json=payload) as response:
            if stream:
                async for line in response.content:
                    if line.strip():
                        yield json.loads(line)
            else:
                yield await response.json()

    async def get_models(self):
        url = f"{self.base_url}/models"
        async with self.session.get(url) as response:
            return await response.json()
