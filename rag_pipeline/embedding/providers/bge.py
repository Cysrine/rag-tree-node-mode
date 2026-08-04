"""Default provider: local BGE via sentence-transformers.

Requires '.[embed-local]' (sentence-transformers + torch). The model is loaded
lazily on first use. BGE recommends a query instruction prefix for retrieval,
applied in ``embed_query`` only. Exposes ``count_tokens`` (the model's own
tokenizer) so the chunker can budget precisely instead of estimating.
"""

from __future__ import annotations

from collections.abc import Sequence

_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class BGEEmbedder:
    def __init__(self, model: str = "BAAI/bge-large-en-v1.5", dim: int = 1024) -> None:
        self._model_name = model
        self._dim = dim
        self._model = None  # lazy

    @property
    def dim(self) -> int:
        return self._dim

    def _ensure_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # lazy/heavy

            self._model = SentenceTransformer(self._model_name)
        return self._model

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        model = self._ensure_model()
        vectors = model.encode(
            list(texts), normalize_embeddings=True, convert_to_numpy=True, batch_size=32
        )
        return [v.tolist() for v in vectors]

    def embed_query(self, text: str) -> list[float]:
        model = self._ensure_model()
        vector = model.encode(
            [_QUERY_INSTRUCTION + text], normalize_embeddings=True, convert_to_numpy=True
        )[0]
        return vector.tolist()

    def count_tokens(self, text: str) -> int:
        """Exact token count via the model tokenizer (special tokens excluded)."""
        model = self._ensure_model()
        return len(model.tokenizer.tokenize(text))
