"""Azure OpenAI clients via the v1 API (no api-version needed). Settings come from .env."""

import os
from functools import cache

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from .config import PROJECT_ROOT

load_dotenv(PROJECT_ROOT / ".env")


def _base_url() -> str:
    return os.environ["AZURE_OPENAI_ENDPOINT"].rstrip("/") + "/openai/v1/"


def embedding_deployment() -> str:
    return os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"]


@cache
def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=embedding_deployment(),
        base_url=_base_url(),
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        check_embedding_ctx_length=False,  # send raw text, not tiktoken ids
        max_retries=8,                     # back off on 429 rate limits
    )


@cache
def get_chat(fast: bool = False) -> ChatOpenAI:
    """fast=True uses AZURE_OPENAI_FAST_DEPLOYMENT, falling back to the chat deployment."""
    deployment = os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"]
    if fast:
        deployment = os.environ.get("AZURE_OPENAI_FAST_DEPLOYMENT") or deployment
    # reasoning models (gpt-5*, o*) reject a custom temperature
    extra = {} if deployment.startswith(("gpt-5", "gpt-6", "o")) else {"temperature": 0}
    return ChatOpenAI(model=deployment, base_url=_base_url(),
                      api_key=os.environ["AZURE_OPENAI_API_KEY"], max_retries=6, **extra)
