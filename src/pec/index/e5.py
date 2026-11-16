"""E5 query encoding via pyterrier-dr.

Only the true query text is ever encoded here (prefix "query: "), for plotting where the
information need sits in exp01 and for the optional distillation loss. Document vectors
always come from the FAISS index, so everything stays in the index's space.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def e5_model(model_name: str, batch_size: int = 32, device: str | None = None):
    import pyterrier_dr

    return pyterrier_dr.E5(model_name, batch_size=batch_size, device=device)


def encode_topics(topics: pd.DataFrame, model_name: str, **kwargs) -> dict[str, np.ndarray]:
    """qid -> L2-normalised E5 query vector."""
    encoder = e5_model(model_name, **kwargs).query_encoder()
    out = encoder(topics[["qid", "query"]])
    return {str(q): np.asarray(v, dtype=np.float32) for q, v in zip(out["qid"], out["query_vec"])}
