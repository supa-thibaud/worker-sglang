import subprocess
import time
import requests
import asyncio
import os
from openai_async import OpenAIAsync  # Assuming you save the OpenAIAsync class in a file named openai_async.py

class SGlangEngine:
    def __init__(self, model="meta-llama/Meta-Llama-3-8B-Instruct", host="0.0.0.0", port=30000):
        self.model = model
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"
        self.process = None
        
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

class OpenAIRequest:
    def __init__(self, base_url="http://0.0.0.0:30000/v1", api_key="EMPTY"):
        self.base_url = base_url
        self.api_key = api_key
        self.client = None

    async def __aenter__(self):
        self.client = OpenAIAsync(base_url=self.base_url, api_key=self.api_key)
        await self.client.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.client.__aexit__(exc_type, exc, tb)
    
    async def request_chat_completions(self, model="default", messages=None, max_tokens=100, stream=False, **kwargs):
        async for chunk in self.client.request_chat_completions(model, messages, max_tokens, stream, **kwargs):
            yield chunk
    
    async def request_completions(self, model="default", prompt="The capital of France is", max_tokens=100, stream=False, **kwargs):
        async for chunk in self.client.request_completions(model, prompt, max_tokens, stream, **kwargs):
            yield chunk
    
    async def get_models(self):
        return await self.client.get_models()
