"""
Thin wrapper around the sentence-transformers model so every module
shares one loaded copy and the same normalisation.
"""

import numpy as np

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

_model = None


def get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed(texts, batch_size=64):
    """Unit-normalised embeddings, so dot product == cosine similarity."""
    vecs = get_model().encode(
        list(texts), batch_size=batch_size, normalize_embeddings=True,
        show_progress_bar=len(texts) > 500,
    )
    return np.asarray(vecs, dtype=np.float32)
