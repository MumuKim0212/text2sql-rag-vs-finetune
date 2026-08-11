"""Few-shot example retrieval: the Spider *train* questions nearest to a dev question.

The index is built from the train split only. Spider's train and dev DB schemas
are disjoint, so a retrieved example can never carry a dev schema or gold query
into the prompt (see the data-leakage rules in project-text2sql-brief.md).
"""

import numpy as np
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_embedder(model_name: str = EMBEDDING_MODEL) -> SentenceTransformer:
    """Load the sentence embedder shared by retrieval and schema linking.

    Pinned to CPU: the index is ~8.6k short questions, so a GPU buys nothing and
    this keeps the step runnable on the machine that holds the Spider data.
    """
    return SentenceTransformer(model_name, device="cpu")


def encode(embedder: SentenceTransformer, texts: list[str]) -> np.ndarray:
    """Embed `texts` as L2-normalised rows, so a dot product is cosine similarity."""
    return embedder.encode(texts, normalize_embeddings=True, convert_to_numpy=True, batch_size=64)


def top_k(query_embeddings: np.ndarray, index_embeddings: np.ndarray, k: int) -> np.ndarray:
    """Return, per query row, the indices of its `k` most similar index rows (best first)."""
    k = min(k, index_embeddings.shape[0])
    similarity = query_embeddings @ index_embeddings.T
    # argpartition finds the k best in linear time; argsort then orders just those k.
    partitioned = np.argpartition(-similarity, kth=k - 1, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(similarity, partitioned, axis=1), axis=1)
    return np.take_along_axis(partitioned, order, axis=1)
