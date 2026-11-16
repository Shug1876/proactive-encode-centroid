"""Synthetic fixtures: a small clustered corpus in an in-memory FAISS index.

Each toy query has a direction in space; perfectly relevant documents sit close to it,
highly and partially relevant ones progressively further away, and non-relevant ones are
background noise. The index lives only in memory for the duration of the tests.
"""
import faiss
import numpy as np
import pandas as pd
import pytest

from pec.index.faiss_store import DocnoMap, FaissDocStore

DIM = 16


def _unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


@pytest.fixture(scope="session")
def toy():
    rng = np.random.default_rng(0)
    vecs, qrels, topics = [], [], []
    spread = {3: 0.15, 2: 0.3, 1: 0.5, 0: 0.9}
    counts = {3: 6, 2: 8, 1: 10, 0: 10}
    for q in range(4):
        centre = _unit(rng.normal(size=DIM))
        topics.append({"qid": f"q{q}", "query": f"toy query number {q}"})
        for grade, count in counts.items():
            for _ in range(count):
                docno = f"d{len(vecs)}"
                vecs.append(_unit(centre + spread[grade] * rng.normal(size=DIM)))
                qrels.append({"qid": f"q{q}", "docno": docno, "label": grade})
    for _ in range(200):
        vecs.append(_unit(rng.normal(size=DIM)))
    vecs = np.asarray(vecs, dtype=np.float32)
    index = faiss.IndexFlatIP(DIM)
    index.add(vecs)
    docnos = [f"d{i}" for i in range(len(vecs))]
    store = FaissDocStore(index, DocnoMap(docnos=docnos))
    return {"store": store, "vecs": vecs, "docnos": docnos, "qrels": pd.DataFrame(qrels),
            "topics": pd.DataFrame(topics)}
