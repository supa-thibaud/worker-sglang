import os
import subprocess
import time
import requests
import asyncio
import aiohttp
from typing import List, Dict, Any, AsyncGenerator
from dataclasses import dataclass

@dataclass
class BatchSize:
    max_size: int
    min_size: int
    growth_factor: float
    current_size: int = None

    def __post_init__(self):
        self.current_size = self.current_size or self.min_size

    def update(self):
        self.current_size = min(int(self.current_size * self.growth_factor), self.max_size)

class SGlangEngine:
    def __init__(self, model="meta-llama/Meta-Llama-3-8B-Instruct", host="0.0.0.0", port=30000):
        self.model = model
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"
        self.process = None
        self.max_concurrency = int(os.getenv("MAX_CONCURRENCY", 100))
        self.default_batch_size = int(os.getenv("DEFAULT_BATCH_SIZE", 32))
        self.batch_size_growth_factor = float(os.getenv("BATCH_SIZE_GROWTH_FACTOR", 1.5))
        self.min_batch_size = int(os.getenv("MIN_BATCH_SIZE", 4))

    def start_server(self):
        command = [
            "python3", "-m", "sglang.launch_server",
            "--host", self.host,
            "--port", str(self.port)
        ]

        # Dictionary of all possible options and their corresponding env var names
        options = {
            'MODEL_PATH': '--model-path',
            'TOKENIZER_PATH': '--tokenizer-path',
            'HOST': '--host',
            'PORT': '--port',
            'ADDITIONAL_PORTS': '--additional-ports',
            'TOKENIZER_MODE': '--tokenizer-mode',
            'LOAD_FORMAT': '--load-format',
            'DTYPE': '--dtype',
            'CONTEXT_LENGTH': '--context-length',
            'QUANTIZATION': '--quantization',
            'SERVED_MODEL_NAME': '--served-model-name',
            'CHAT_TEMPLATE': '--chat-template',
            'MEM_FRACTION_STATIC': '--mem-fraction-static',
            'MAX_RUNNING_REQUESTS': '--max-running-requests',
            'MAX_NUM_REQS': '--max-num-reqs',
            'MAX_TOTAL_TOKENS': '--max-total-tokens',
            'CHUNKED_PREFILL_SIZE': '--chunked-prefill-size',
            'MAX_PREFILL_TOKENS': '--max-prefill-tokens',
            'SCHEDULE_POLICY': '--schedule-policy',
            'SCHEDULE_CONSERVATIVENESS': '--schedule-conservativeness',
            'TENSOR_PARALLEL_SIZE': '--tensor-parallel-size',
            'STREAM_INTERVAL': '--stream-interval',
            'RANDOM_SEED': '--random-seed',
            'LOG_LEVEL': '--log-level',
            'LOG_LEVEL_HTTP': '--log-level-http',
            'API_KEY': '--api-key',
            'FILE_STORAGE_PTH': '--file-storage-pth',
            'DATA_PARALLEL_SIZE': '--data-parallel-size',
            'LOAD_BALANCE_METHOD': '--load-balance-method',
        }

        # Boolean flags
        boolean_flags = [
            'SKIP_TOKENIZER_INIT', 'TRUST_REMOTE_CODE', 'LOG_REQUESTS',
            'SHOW_TIME_COST', 'DISABLE_FLASHINFER', 'DISABLE_FLASHINFER_SAMPLING',
            'DISABLE_RADIX_CACHE', 'DISABLE_REGEX_JUMP_FORWARD', 'DISABLE_CUDA_GRAPH',
            'DISABLE_DISK_CACHE', 'ENABLE_TORCH_COMPILE', 'ENABLE_P2P_CHECK',
            'ENABLE_MLA', 'ATTENTION_REDUCE_IN_FP32', 'EFFICIENT_WEIGHT_LOAD'
        ]

        # Add options from environment variables only if they are set
        for env_var, option in options.items():
            value = os.getenv(env_var)
            if value is not None and value != "":
                command.extend([option, value])

        # Add boolean flags only if they are set to true
        for flag in boolean_flags:
            if os.getenv(flag, '').lower() in ('true', '1', 'yes'):
                command.append(f"--{flag.lower().replace('_', '-')}")

        print("LAUNCH SERVER COMMAND:")
        print(command)
        self.process = subprocess.Popen(command, stdout=None, stderr=None)
        print(f"Server started with PID: {self.process.pid}")

    def wait_for_server(self, timeout=300, interval=5):
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"{self.base_url}/v1/models")
                if response.status_code == 200:
                    print("Server is ready!")
                    return True
            except requests.RequestException:
                pass
            time.sleep(interval)
        raise TimeoutError("Server failed to start within the timeout period.")

    def shutdown(self):
        if self.process:
            self.process.terminate()
            self.process.wait() 
            print("Server shut down.")

    async def generate_batch(self, inputs: List[Dict[str, Any]], batch_size: BatchSize) -> AsyncGenerator[List[Dict[str, Any]], None]:
        async with aiohttp.ClientSession() as session:
            while inputs:
                current_batch = inputs[:batch_size.current_size]
                inputs = inputs[batch_size.current_size:]

                tasks = [self.generate_single(session, input_data) for input_data in current_batch]
                batch_results = await asyncio.gather(*tasks)

                yield batch_results
                batch_size.update()

    async def generate_single(self, session: aiohttp.ClientSession, input_data: Dict[str, Any]) -> Dict[str, Any]:
        generate_url = f"{self.base_url}/generate"
        headers = {"Content-Type": "application/json"}
        generate_data = {
            "text": input_data.get("prompt", ""),
            "sampling_params": input_data.get("sampling_params", {})
        }

        async with session.post(generate_url, json=generate_data, headers=headers) as response:
            if response.status == 200:
                return await response.json()
            else:
                return {"error": f"Generate request failed with status code {response.status}", "details": await response.text()}

    async def generate(self, job_input: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
        batch_size = BatchSize(
            max_size=self.default_batch_size,
            min_size=self.min_batch_size,
            growth_factor=self.batch_size_growth_factor
        )

        if isinstance(job_input, list):
            async for batch in self.generate_batch(job_input, batch_size):
                yield batch
        else:
            async for batch in self.generate_batch([job_input], batch_size):
                yield batch[0]

class OpenAIRequest:
    def __init__(self, base_url="http://0.0.0.0:30000/v1", api_key="EMPTY"):
        self.base_url = base_url
        self.api_key = api_key

    async def request_chat_completions(self, model="default", messages=None, max_tokens=100, stream=False, frequency_penalty=0.0, n=1, stop=None, temperature=1.0, top_p=1.0):
        if messages is None:
            messages = [
                {"role": "system", "content": "You are a helpful AI assistant"},
                {"role": "user", "content": "List 3 countries and their capitals."},
            ]
        
        url = f"{self.base_url}/chat/completions"
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
        data = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": stream,
            "frequency_penalty": frequency_penalty,
            "n": n,
            "stop": stop,
            "temperature": temperature,
            "top_p": top_p
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=data, headers=headers) as response:
                if stream:
                    async for line in response.content:
                        if line:
                            yield json.loads(line.decode('utf-8').strip('data: '))
                else:
                    yield await response.json()

    async def request_completions(self, model="default", prompt="The capital of France is", max_tokens=100, stream=False, frequency_penalty=0.0, n=1, stop=None, temperature=1.0, top_p=1.0):
        url = f"{self.base_url}/completions"
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
        data = {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "stream": stream,
            "frequency_penalty": frequency_penalty,
            "n": n,
            "stop": stop,
            "temperature": temperature,
            "top_p": top_p
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=data, headers=headers) as response:
                if stream:
                    async for line in response.content:
                        if line:
                            yield json.loads(line.decode('utf-8').strip('data: '))
                else:
                    yield await response.json()

    async def get_models(self):
        url = f"{self.base_url}/models"
        headers = {"Authorization": f"Bearer {self.api_key}"}

        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers) as response:
                return await response.json()