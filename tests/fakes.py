"""Offline stand-ins for the Azure OpenAI models.

FakeEmbeddings: deterministic bag-of-words hashing, so texts sharing words are similar.
FakeChat: answers the pipeline's structured-output calls from a `responder(schema, input)`.
"""

import hashlib
import math
import re

from langchain_core.messages import AIMessage

DIM = 64


class FakeEmbeddings:
    def __init__(self):
        self.calls = 0

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * DIM
        for word in re.findall(r"\w+", text.lower()):
            if len(word) > 2:
                vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % DIM] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


SOURCE_BLOCK = re.compile(r"\[S(\d+)\] [^\n]*\n(.*?)(?=\n\n---\n\n\[S\d+\]|\Z)", re.S)
INVENTED_QUOTE = "This sentence was invented by the model and appears in no source at all."


def sources_in(prompt: str) -> dict[int, str]:
    return {int(n): text for n, text in SOURCE_BLOCK.findall(prompt)}


def first_sentence(text: str) -> str:
    plain = re.sub(r"[*#_]", "", text).strip()
    return re.split(r"(?<=[.!?])\s", " ".join(plain.split()))[0]


def default_responder(schema, payload):
    """Plausible answers: rewrite → two queries, grade → first 3 sources, answer → one real
    quote from S1, one real quote from S2 and one invented quote (must be dropped)."""
    from articles_agent.graph import Answer, Finding, Queries, Relevant

    if schema is Queries:
        return Queries(queries=["convolutional neural network surrogate topology optimization",
                                "Topologieoptimierung neuronale Netze"])
    if schema is Relevant:
        ids = sorted(sources_in(str(payload)))[:3]
        return Relevant(source_ids=[f"S{i}" for i in ids])
    if schema is Answer:
        prompt = payload[-1].content
        src = sources_in(prompt)
        findings = [Finding(claim=f"Claim from S{i}", quote=first_sentence(src[i]), source_id=f"S{i}")
                    for i in sorted(src)[:2]]
        findings.append(Finding(claim="Hallucinated", quote=INVENTED_QUOTE, source_id="S1"))
        return Answer(summary="CNN surrogates speed up the optimization [S1][S2].", findings=findings)
    raise AssertionError(f"unexpected schema {schema}")


class FakeChat:
    model_name = "fake-chat"

    def __init__(self, responder=default_responder):
        self.responder = responder
        self.calls: list[tuple[str, object]] = []

    def with_structured_output(self, schema, **_):
        chat = self

        class Structured:
            def invoke(self, payload):
                chat.calls.append((schema.__name__, payload))
                return chat.responder(schema, payload)

        return Structured()

    def invoke(self, _payload):
        return AIMessage(content="OK")
