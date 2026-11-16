"""Text retrievers of proactive-icl-query-pair (src/data.py, get_retr), used in
Experiment 1 to evaluate the text queries of the query formulation baselines.

- bm25:    BM25 over a PISA index
- monot5:  bm25 % 50 >> passage text >> MonoT5ReRanker() (re-ranks BM25's top 50)
- colbert: TCT-ColBERT (castorini/tct_colbert-v2-hnp-msmarco) over a dense index (a FlexIndex
           with its numpy retriever for MS MARCO, a NumpyIndex for TREC-COVID)
- splade:  SPLADE (naver/splade-cocondenser-ensembledistil) query encoder over a quantized
           PISA index built without stemming

The models, the index kinds and the re-ranking depth are the ones that repository used, and
the index locations are its paths (configs/paths.yaml, `text_retrievers.<index key>`, so
DL19 and DL20 share the MS MARCO entries). Two things differ, both for Experiment 1's
evaluation: every retriever returns `num_results` documents (that repository cut ColBERT
and SPLADE at its --retr_k, 10, and evaluated nDCG@10 only), searching deep enough that
`num_results` remain after the context passages are removed (ResidualTextRetriever); and
ColBERT and SPLADE do not fetch passage text after retrieval, which does not change their
rankings.
"""
from __future__ import annotations

import pandas as pd
import pyterrier as pt

from pec.config import require_path

RETRIEVERS = ("bm25", "monot5", "colbert", "splade")
TCT_COLBERT_MODEL = 'castorini/tct_colbert-v2-hnp-msmarco'
SPLADE_MODEL = 'naver/splade-cocondenser-ensembledistil'
MONOT5_DEPTH = 50  # BM25 documents re-ranked by monoT5 (retr % 50)


def _pisa_index(path, what: str, **kwargs):
    """A PISA index from a local path, or a HuggingFace artifact given as "hf:<repo>"
    (proactive-icl-query-pair loads TREC-COVID's BM25 index as pyterrier/trec-covid.pisa)."""
    import pyterrier_pisa as ptp

    path = str(path)
    if path.startswith("hf:"):
        return pt.Artifact.from_hf(path[3:])
    return ptp.PisaIndex(str(require_path(path, what)), **kwargs)


def get_retr(retr_name: str, entry: dict, corpus: str, num_results: int = 1000,
             monot5_depth: int = MONOT5_DEPTH) -> pt.Transformer:
    """[qid, query] -> ranked documents, as proactive-icl-query-pair's get_retr builds them.

    `entry` is the dataset index's `text_retrievers` entry in paths.yaml; `corpus` the
    ir_datasets corpus the passage texts come from (monoT5)."""
    if retr_name not in RETRIEVERS:
        raise ValueError(f"Unknown retriever: {retr_name}; choose from {RETRIEVERS}")
    if retr_name in ("bm25", "monot5"):
        retr = _pisa_index(entry["bm25"], "BM25 PISA index").bm25(num_results=num_results)
        if retr_name == "monot5":
            from pyterrier_t5 import MonoT5ReRanker

            retr = retr % monot5_depth
            monoT5 = MonoT5ReRanker()
            retr = retr >> pt.text.get_text(pt.get_dataset(corpus), ['text']) >> monoT5
        return retr
    if retr_name == "colbert":
        import pyterrier_dr as ptdr

        model = ptdr.TctColBert(TCT_COLBERT_MODEL)
        dense = entry["colbert"]
        if "flex" in dense:
            index = ptdr.FlexIndex(str(require_path(dense["flex"], "TCT-ColBERT FlexIndex"))).np_retriever(
                num_results=num_results)
        else:
            index = ptdr.NumpyIndex(str(require_path(dense["numpy"], "TCT-ColBERT NumpyIndex")), num_results=num_results)
        return model >> index
    import pyterrier_splade

    splade = pyterrier_splade.Splade(SPLADE_MODEL)
    index = _pisa_index(entry["splade"], "SPLADE PISA index", stemmer='none')
    return splade.query_encoder() >> index.quantized(num_results=num_results)


class ResidualTextRetriever(pt.Transformer):
    """[qid, query, seed_docnos] -> the retriever's top `num_results` documents for the
    query text, context passages (seed_docnos) removed, re-ranked from 0."""

    def __init__(self, retriever, num_results: int = 1000):
        self.retriever, self.num_results = retriever, num_results

    def transform(self, inp: pd.DataFrame) -> pd.DataFrame:
        pt.validate.columns(inp, includes=["qid", "query", "seed_docnos"])
        res = self.retriever(inp[["qid", "query"]].reset_index(drop=True))
        if len(res) == 0:
            return pd.DataFrame(columns=["qid", "docno", "score", "rank"])
        exclude = {(q, str(d)) for q, seeds in zip(inp["qid"], inp["seed_docnos"]) for d in seeds}
        res = res[[(q, str(d)) not in exclude for q, d in zip(res["qid"], res["docno"])]]
        res = res[["qid", "docno", "score"]].astype({"docno": str})
        res = res.sort_values(["qid", "score"], ascending=[True, False])
        res = res.groupby("qid", sort=False).head(self.num_results).reset_index(drop=True)
        return pt.model.add_ranks(res)


def build_retrievers(cfg: dict, dataset: str, names, num_results: int = 1000, depth: int | None = None,
                     monot5_depth: int = MONOT5_DEPTH) -> dict[str, ResidualTextRetriever]:
    """name -> ResidualTextRetriever for the dataset, each searching `depth` documents
    (enough for `num_results` after removing the context) and keeping `num_results`."""
    entry = cfg["datasets"][dataset]
    index_key = entry["index"]
    paths = (cfg.get("text_retrievers") or {}).get(index_key)
    if paths is None:
        raise KeyError(f"no text_retrievers.{index_key} entry in configs/paths.yaml (needed for {dataset})")
    corpus = entry.get("corpus", entry.get("irds"))
    depth = max(depth or num_results, num_results)
    return {name: ResidualTextRetriever(get_retr(name, paths, corpus, depth, monot5_depth), num_results)
            for name in names}
