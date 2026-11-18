"""Use the standard SDK directly; preserve its responses and error types."""

from openai import OpenAI

from .config import API_KEY, BASE_URL


def create_client(*, timeout: float = 60) -> OpenAI:
    return OpenAI(base_url=BASE_URL, api_key=API_KEY, timeout=timeout, max_retries=0)
