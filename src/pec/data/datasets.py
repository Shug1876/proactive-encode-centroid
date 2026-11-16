"""Dataset loading through PyTerrier.

TREC-DL19, TREC-DL20 and TREC-COVID come from PyTerrier's ir_datasets integration. A
dataset that is not in ir_datasets can instead point at qrels and topics files, which are
read with PyTerrier's TREC readers.
"""
from __future__ import annotations

import pandas as pd

from pec.config import require_path
from pec.data.qrels import Relevance, normalise_qrels


def load_dataset(cfg: dict, key: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(topics[qid, query], qrels[qid, docno, label]) for a dataset key in paths.yaml."""
    import pyterrier as pt

    entry = cfg["datasets"][key]
    if "irds" in entry:
        ds = pt.get_dataset(entry["irds"])
        # datasets with several query fields (e.g. beir/trec-covid: text, query, narrative)
        # need the field named, otherwise PyTerrier returns no "query" column
        topics, qrels = ds.get_topics(variant=entry.get("topics_variant")), ds.get_qrels()
    else:
        qrels = pt.io.read_qrels(str(require_path(entry["qrels"], f"{key} qrels")))
        topics = pt.io.read_topics(str(require_path(entry["topics"], f"{key} topics")),
                                   format=entry.get("topics_format", "singleline"))
    topics = topics[["qid", "query"]].astype({"qid": str})
    qrels = normalise_qrels(qrels)
    topics = topics[topics["qid"].isin(set(qrels["qid"]))].reset_index(drop=True)
    return topics, qrels


def relevance(cfg: dict, key: str) -> Relevance:
    """The dataset's relevance scheme (context grade, target grade, evaluation views)."""
    return Relevance.from_config(cfg["datasets"][key].get("relevance"))


def text_loader(cfg: dict, key: str):
    """PyTerrier transformer adding a `text` column for docnos, from the dataset corpus.
    Several `text_fields` (e.g. [title, text]) are joined with a space."""
    import pyterrier as pt

    entry = cfg["datasets"][key]
    fields = list(entry.get("text_fields", ["text"]))
    getter = pt.text.get_text(pt.get_dataset(entry.get("corpus", entry.get("irds"))), fields)
    if fields == ["text"]:
        return getter
    join = lambda df: df.assign(text=df[fields].fillna("").astype(str).agg(" ".join, axis=1).str.strip())
    return getter >> pt.apply.generic(join)
