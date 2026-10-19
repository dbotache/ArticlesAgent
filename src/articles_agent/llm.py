"""Azure OpenAI clients via the v1 API (no api-version needed). Settings come from .env."""

import os
from functools import cache

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from .config import APP_HOME

load_dotenv(APP_HOME / ".env")  # shared by all projects


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


# ------------------------------------------------------------ settings for the setup page

SETTINGS = {
    "AZURE_OPENAI_ENDPOINT": "Endpoint (https://<resource>.openai.azure.com/)",
    "AZURE_OPENAI_API_KEY": "API key",
    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT": "Embedding deployment",
    "AZURE_OPENAI_CHAT_DEPLOYMENT": "Chat deployment (answers)",
    "AZURE_OPENAI_FAST_DEPLOYMENT": "Fast deployment (query rewriting; optional)",
}
REQUIRED = [k for k in SETTINGS if k != "AZURE_OPENAI_FAST_DEPLOYMENT"]


def missing_settings() -> list[str]:
    return [k for k in REQUIRED if not os.environ.get(k)]


def save_settings(values: dict[str, str]) -> None:
    """Write the given variables into .env (other lines are kept) and apply them now."""
    env = APP_HOME / ".env"
    lines = env.read_text(encoding="utf-8").splitlines() if env.exists() else []
    values = {k: v.strip() for k, v in values.items()}
    out, done = [], set()
    for line in lines:
        key = line.split("=", 1)[0].strip()
        if key in values:
            out.append(f"{key}={values[key]}")
            done.add(key)
        else:
            out.append(line)
    out += [f"{k}={v}" for k, v in values.items() if k not in done]
    env.write_text("\n".join(out) + "\n", encoding="utf-8")
    env.chmod(0o600)
    os.environ.update(values)
    get_embeddings.cache_clear()
    get_chat.cache_clear()


def check_connection() -> list[tuple[str, bool, str]]:
    """Try one tiny call per deployment: [(label, ok, message)]."""
    checks = []
    try:
        dim = len(get_embeddings().embed_query("test"))
        checks.append((f"Embeddings ({embedding_deployment()})", True, f"{dim} dimensions"))
    except Exception as exc:
        checks.append((f"Embeddings ({os.environ.get('AZURE_OPENAI_EMBEDDING_DEPLOYMENT')})", False, str(exc)[:300]))
    for fast in (False, True):
        llm = get_chat(fast=fast)
        try:
            llm.invoke("Reply with OK")
            checks.append((f"{'Fast' if fast else 'Chat'} model ({llm.model_name})", True, "responds"))
        except Exception as exc:
            checks.append((f"{'Fast' if fast else 'Chat'} model ({llm.model_name})", False, str(exc)[:300]))
    return checks
