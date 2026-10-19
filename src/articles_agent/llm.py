"""Model providers: chat models (freely switchable) and embedding models (pinned per project).

Settings live in the app home's .env (shared by all projects):

    CHAT_PROVIDER / CHAT_MODEL           answers                 azure | openai | anthropic | compatible
    FAST_PROVIDER / FAST_MODEL           query rewriting, grading (optional: defaults to the chat model)
    EMBEDDING_PROVIDER / EMBEDDING_MODEL default for *new* projects: azure | openai | compatible
    AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY
    OPENAI_API_KEY
    ANTHROPIC_API_KEY
    COMPATIBLE_BASE_URL, COMPATIBLE_API_KEY   any OpenAI-compatible server (Ollama, LM Studio, vLLM, …)

The pre-provider settings (AZURE_OPENAI_CHAT/FAST/EMBEDDING_DEPLOYMENT) still work: without
CHAT_PROVIDER they mean "Azure with these deployments".

Chat models can be changed at any time. An embedding model can't: a project's index only works
with the model that built it, so each project pins its embedding model (projects.py), and
changing it means re-embedding that project.
"""

import os
from dataclasses import asdict, dataclass
from functools import cache

from dotenv import load_dotenv

from .config import APP_HOME

load_dotenv(APP_HOME / ".env")  # shared by all projects


@dataclass(frozen=True)
class Provider:
    name: str
    credentials: tuple[str, ...]        # required settings
    optional: tuple[str, ...] = ()
    chat_models: tuple[str, ...] = ()   # suggestions; any model name can be typed in
    embedding_models: tuple[str, ...] = ()
    model_hint: str = "Model name"


PROVIDERS = {
    "azure": Provider("Azure OpenAI", ("AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY"),
                      chat_models=("gpt-5.4", "gpt-5.4-mini", "gpt-4.1-mini", "gpt-4.1"),
                      embedding_models=("text-embedding-3-small", "text-embedding-3-large"),
                      model_hint="Deployment name (as created in Azure AI Foundry)"),
    "openai": Provider("OpenAI", ("OPENAI_API_KEY",),
                       chat_models=("gpt-5.4", "gpt-5.4-mini", "gpt-4.1-mini"),
                       embedding_models=("text-embedding-3-small", "text-embedding-3-large")),
    "anthropic": Provider("Anthropic (Claude)", ("ANTHROPIC_API_KEY",),
                          chat_models=("claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5")),
    "compatible": Provider("OpenAI-compatible (Ollama, LM Studio, vLLM, OpenRouter, …)",
                           ("COMPATIBLE_BASE_URL",), optional=("COMPATIBLE_API_KEY",),
                           embedding_models=("nomic-embed-text", "bge-m3", "mxbai-embed-large"),
                           model_hint="Model name as the server lists it"),
}
CHAT_PROVIDERS = list(PROVIDERS)
EMBEDDING_PROVIDERS = [k for k, p in PROVIDERS.items() if p.embedding_models or k == "compatible"]

CREDENTIAL_LABELS = {
    "AZURE_OPENAI_ENDPOINT": "Azure endpoint (https://<resource>.openai.azure.com/)",
    "AZURE_OPENAI_API_KEY": "Azure API key",
    "OPENAI_API_KEY": "OpenAI API key",
    "ANTHROPIC_API_KEY": "Anthropic API key",
    "COMPATIBLE_BASE_URL": "OpenAI-compatible base URL (e.g. http://localhost:11434/v1)",
    "COMPATIBLE_API_KEY": "OpenAI-compatible API key (optional)",
}
SECRET_KEYS = {"AZURE_OPENAI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "COMPATIBLE_API_KEY"}


# ------------------------------------------------------------ model specs

@dataclass(frozen=True)
class ModelSpec:
    provider: str
    model: str

    @property
    def label(self) -> str:
        return f"{self.provider}:{self.model}"


@dataclass(frozen=True)
class EmbeddingSpec:
    provider: str
    model: str
    base_url: str = ""  # only for "compatible"

    @property
    def identity(self) -> str:
        """What an index / the embedding cache is keyed by. Azure keeps the bare deployment name,
        as indexes built before providers existed did."""
        return self.model if self.provider == "azure" else f"{self.provider}:{self.model}"

    @property
    def label(self) -> str:
        return f"{self.provider}:{self.model}"

    def to_dict(self) -> dict:
        return asdict(self)


def _env(key: str) -> str:
    return os.environ.get(key, "").strip()


def chat_spec(fast: bool = False) -> ModelSpec:
    if _env("CHAT_PROVIDER"):
        main = ModelSpec(_env("CHAT_PROVIDER"), _env("CHAT_MODEL"))
        if fast and _env("FAST_MODEL"):
            return ModelSpec(_env("FAST_PROVIDER") or main.provider, _env("FAST_MODEL"))
        return main
    # pre-provider settings: Azure deployments
    model = _env("AZURE_OPENAI_CHAT_DEPLOYMENT")
    if fast and _env("AZURE_OPENAI_FAST_DEPLOYMENT"):
        model = _env("AZURE_OPENAI_FAST_DEPLOYMENT")
    return ModelSpec("azure", model)


def default_embedding_spec() -> EmbeddingSpec:
    """The embedding model a *new* project's database is built with."""
    if _env("EMBEDDING_PROVIDER"):
        provider = _env("EMBEDDING_PROVIDER")
        return EmbeddingSpec(provider, _env("EMBEDDING_MODEL"),
                             _env("COMPATIBLE_BASE_URL") if provider == "compatible" else "")
    return EmbeddingSpec("azure", _env("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"))


def embedding_deployment() -> str:
    """Identity of the default embedding model (kept for older callers)."""
    return default_embedding_spec().identity


def missing_settings(extra_embeddings: tuple[EmbeddingSpec, ...] = ()) -> list[str]:
    """Settings still needed for the configured chat/fast/default-embedding models (and any
    project-pinned embedding models passed in)."""
    missing = []
    specs = [chat_spec(), chat_spec(fast=True), default_embedding_spec(), *extra_embeddings]
    for spec in specs:
        if not spec.model:
            key = "CHAT_MODEL" if isinstance(spec, ModelSpec) else "EMBEDDING_MODEL"
            if not _env("CHAT_PROVIDER") and not _env("EMBEDDING_PROVIDER"):  # legacy names
                key = "AZURE_OPENAI_CHAT_DEPLOYMENT" if isinstance(spec, ModelSpec) else \
                    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT"
            missing.append(key)
        provider = PROVIDERS.get(spec.provider)
        if provider is None:
            missing.append(f"unknown provider '{spec.provider}'")
            continue
        missing += [k for k in provider.credentials if not _env(k)]
    return list(dict.fromkeys(missing))


# ------------------------------------------------------------ clients

def _azure_base_url() -> str:
    return _env("AZURE_OPENAI_ENDPOINT").rstrip("/") + "/openai/v1/"


def _base_url() -> str:  # kept for older callers / tests
    return _azure_base_url()


def _openai_kwargs(provider: str) -> dict:
    if provider == "azure":
        return {"base_url": _azure_base_url(), "api_key": _env("AZURE_OPENAI_API_KEY")}
    if provider == "openai":
        return {"api_key": _env("OPENAI_API_KEY")}
    if provider == "compatible":  # local servers often need no key, but the client wants one
        return {"base_url": _env("COMPATIBLE_BASE_URL"), "api_key": _env("COMPATIBLE_API_KEY") or "none"}
    raise ValueError(f"{provider} has no OpenAI-style API")


@cache
def _embeddings_for(spec: EmbeddingSpec):
    from langchain_openai import OpenAIEmbeddings
    kwargs = _openai_kwargs(spec.provider)
    if spec.provider == "compatible" and spec.base_url:
        kwargs["base_url"] = spec.base_url
    return OpenAIEmbeddings(model=spec.model, check_embedding_ctx_length=False,  # send raw text
                            max_retries=8, **kwargs)                            # back off on 429


def get_embeddings(spec: EmbeddingSpec | None = None):
    return _embeddings_for(spec or default_embedding_spec())


get_embeddings.cache_clear = _embeddings_for.cache_clear


@cache
def _chat_for(spec: ModelSpec):
    if spec.provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        # current Claude models fix their own sampling (a custom temperature is rejected)
        return ChatAnthropic(model=spec.model, api_key=_env("ANTHROPIC_API_KEY"),
                             max_tokens=16000, max_retries=6)
    from langchain_openai import ChatOpenAI
    # reasoning models (gpt-5*, o*) reject a custom temperature
    reasoning = spec.model.startswith(("gpt-5", "gpt-6", "o1", "o3", "o4"))
    extra = {} if reasoning else {"temperature": 0}
    return ChatOpenAI(model=spec.model, max_retries=6, **_openai_kwargs(spec.provider), **extra)


def get_chat(fast: bool = False):
    """The chat model for answers, or (fast=True) for query rewriting and grading."""
    return _chat_for(chat_spec(fast))


get_chat.cache_clear = _chat_for.cache_clear


def structured(llm, schema, fast: bool = False):
    """`llm` constrained to return `schema`, in the way the provider supports best:
    OpenAI/Azure strict JSON schema; Claude's native structured outputs (not forced tool use,
    which current Claude models reject); plain JSON schema for compatible servers
    (override with STRUCTURED_OUTPUT_METHOD=function_calling|json_mode if a server needs it)."""
    provider = chat_spec(fast).provider
    if provider in ("azure", "openai"):
        return llm.with_structured_output(schema, method="json_schema", strict=True)
    if provider == "anthropic":
        return llm.with_structured_output(schema, method="json_schema")
    return llm.with_structured_output(schema, method=_env("STRUCTURED_OUTPUT_METHOD") or "json_schema")


def models_used(embedding: EmbeddingSpec | None = None) -> dict:
    """For provenance in answers: which models produced it."""
    out = {"chat": chat_spec().label, "fast": chat_spec(fast=True).label}
    if embedding:
        out["embedding"] = embedding.label
    return out


# ------------------------------------------------------------ settings for the app

def save_settings(values: dict[str, str]) -> None:
    """Write the given variables into .env (other lines are kept) and apply them now.
    An empty value removes the variable."""
    env = APP_HOME / ".env"
    lines = env.read_text(encoding="utf-8").splitlines() if env.exists() else []
    values = {k: v.strip() for k, v in values.items()}
    out, done = [], set()
    for line in lines:
        key = line.split("=", 1)[0].strip()
        if key in values:
            done.add(key)
            if values[key]:
                out.append(f"{key}={values[key]}")
        else:
            out.append(line)
    out += [f"{k}={v}" for k, v in values.items() if k not in done and v]
    env.write_text("\n".join(out) + "\n", encoding="utf-8")
    env.chmod(0o600)
    for k, v in values.items():
        if v:
            os.environ[k] = v
        else:
            os.environ.pop(k, None)
    get_embeddings.cache_clear()
    get_chat.cache_clear()


def check_connection(embeddings: tuple[EmbeddingSpec, ...] = ()) -> list[tuple[str, bool, str]]:
    """One tiny call per configured model: [(label, ok, message)]."""
    checks = []
    for spec in dict.fromkeys((default_embedding_spec(), *embeddings)):
        label = f"Embeddings ({spec.label})"
        try:
            dim = len(get_embeddings(spec).embed_query("test"))
            checks.append((label, True, f"{dim} dimensions"))
        except Exception as exc:
            checks.append((label, False, str(exc)[:300]))
    for fast in (False, True):
        spec = chat_spec(fast)
        label = f"{'Fast' if fast else 'Chat'} model ({spec.label})"
        try:
            get_chat(fast=fast).invoke("Reply with OK")
            checks.append((label, True, "responds"))
        except Exception as exc:
            checks.append((label, False, str(exc)[:300]))
    return checks
