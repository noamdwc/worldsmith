"""Owner-controlled inference settings; never read from the researcher workspace."""

MODEL = "Qwen/Qwen3-4B-Instruct-2507"
MODEL_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
VLLM_REQUIREMENT = "vllm==0.10.2"
OPENAI_REQUIREMENT = "openai==1.99.9"
# vLLM 0.10.2 predates Transformers 5; pin the model loader as well.
TRANSFORMERS_REQUIREMENT = "transformers==4.55.2"
BASE_URL = "http://127.0.0.1:8000/v1"
API_KEY = "worldsmith-local"


def server_command(python: str) -> list[str]:
    return [
        python, "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL, "--revision", MODEL_REVISION,
        "--tokenizer-revision", MODEL_REVISION,
        "--host", "127.0.0.1", "--port", "8000", "--api-key", API_KEY,
        "--enable-auto-tool-choice", "--tool-call-parser", "hermes",
        # The pinned tokenizer supplies the tool-aware chat template. Bound KV
        # cache and concurrency for Colab; do not allocate the native 256K context.
        "--dtype", "half", "--max-model-len", "8192",
        "--max-num-seqs", "1", "--gpu-memory-utilization", "0.6",
        "--generation-config", "vllm", "--seed", "0",
    ]
