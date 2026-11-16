"""PyTerrier transformers for context-driven retrieval.

Input to every pipeline is a context topics frame [qid, seed_docnos] (see
pec.data.seeds.seed_topics). Encoders add a `query_vec` column in the E5 document space;
retrievers turn it into a ranked result frame. Context documents are removed from every
ranking (residual ranking), since the user already has them.

    centroid  = CentroidEncoder(store) >> FaissRetriever(store)
    seed_rrf  = SeedFusionRetriever(store, fusion="rrf")
    oracle    = FixedQueryVectors(query_vecs) >> FaissRetriever(store)
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
import pyterrier as pt


def _normalise(v: np.ndarray) -> np.ndarray:
    return v / max(float(np.linalg.norm(v)), 1e-12)


def _context_rows(store, seed_docnos) -> np.ndarray:
    rows = store.rows(list(seed_docnos))
    return rows[rows >= 0]


class CentroidEncoder(pt.Transformer):
    """query_vec = normalised mean of the context vectors (Rocchio-style, no training)."""

    def __init__(self, store):
        self.store = store

    def transform(self, inp: pd.DataFrame) -> pd.DataFrame:
        pt.validate.columns(inp, includes=["qid", "seed_docnos"])
        vecs = [_normalise(self.store.vectors(_context_rows(self.store, s)).mean(0)) for s in inp["seed_docnos"]]
        return inp.assign(query_vec=vecs)


class TrueQueryEncoder(pt.Transformer):
    """Oracle reference: E5 encoding of the real query text (unknown to every baseline)."""

    def __init__(self, topics: pd.DataFrame, model_name: str, **kwargs):
        from pec.index.e5 import e5_model

        self.topics = topics[["qid", "query"]]
        self.encoder = e5_model(model_name, **kwargs).query_encoder()

    def transform(self, inp: pd.DataFrame) -> pd.DataFrame:
        pt.validate.columns(inp, includes=["qid"])
        enc = self.encoder(inp[["qid"]].merge(self.topics, on="qid"))
        return inp.merge(enc[["qid", "query_vec"]], on="qid")


class FixedQueryVectors(pt.Transformer):
    """query_vec from a precomputed qid -> vector mapping (e.g. cached E5 query vectors)."""

    def __init__(self, vectors: dict[str, np.ndarray]):
        self.vectors = vectors

    def transform(self, inp: pd.DataFrame) -> pd.DataFrame:
        pt.validate.columns(inp, includes=["qid"])
        inp = inp[inp["qid"].isin(self.vectors.keys())]
        return inp.assign(query_vec=[self.vectors[q] for q in inp["qid"]])


def _results_frame(records: list[dict], carry: pd.DataFrame) -> pd.DataFrame:
    res = pd.DataFrame.from_records(records, columns=["qid", "docno", "score"])
    res = res.merge(carry.drop_duplicates(subset="qid"), on="qid", how="left")
    res = res.sort_values(["qid", "score"], ascending=[True, False]).reset_index(drop=True)
    return pt.model.add_ranks(res)


class FaissRetriever(pt.Transformer):
    """query_vec -> top `num_results` documents by cosine, context documents removed."""

    def __init__(self, store, num_results: int = 1000):
        self.store, self.num_results = store, num_results

    def transform(self, inp: pd.DataFrame) -> pd.DataFrame:
        pt.validate.columns(inp, includes=["qid", "query_vec"])
        has_seeds = "seed_docnos" in inp.columns
        extra = max((len(s) for s in inp["seed_docnos"]), default=0) if has_seeds else 0
        sims, rows = self.store.search(np.stack(inp["query_vec"].tolist()), self.num_results + extra)
        records = []
        for i, qid in enumerate(inp["qid"]):
            exclude = set(inp["seed_docnos"].iloc[i]) if has_seeds else set()
            ok = rows[i] >= 0
            kept = 0
            for docno, score in zip(self.store.docnos(rows[i][ok]), sims[i][ok].tolist()):
                if docno in exclude:
                    continue
                records.append({"qid": qid, "docno": docno, "score": score})
                kept += 1
                if kept == self.num_results:
                    break
        return _results_frame(records, inp.drop(columns=["docno", "score", "rank"], errors="ignore"))


class SeedFusionRetriever(pt.Transformer):
    """Baseline: search with every context document separately and fuse the lists.

    - fusion="max": a document scores the highest cosine it reached against any single
      context document (rewards one strong match).
    - fusion="rrf": reciprocal rank fusion, sum over context documents of
      1 / (rrf_k + rank), rank counted after removing the context documents from each
      list (rewards documents that many context documents agree on). A list that does
      not contain the document contributes nothing, so `depth` bounds what can score.
    """

    def __init__(self, store, num_results: int = 1000, fusion: str = "max", rrf_k: int = 60, depth: int | None = None):
        if fusion not in ("max", "rrf"):
            raise ValueError(f"fusion must be 'max' or 'rrf', not {fusion!r}")
        self.store, self.num_results, self.fusion, self.rrf_k = store, num_results, fusion, rrf_k
        self.depth = depth or num_results

    def transform(self, inp: pd.DataFrame) -> pd.DataFrame:
        pt.validate.columns(inp, includes=["qid", "seed_docnos"])
        records = []
        for qid, seeds in zip(inp["qid"], inp["seed_docnos"]):
            rows = _context_rows(self.store, seeds)
            context = set(rows.tolist())
            sims, found = self.store.search(self.store.vectors(rows), self.depth + len(rows))
            fused: dict[int, float] = {}
            for list_rows, list_sims in zip(found, sims):
                keep = [(r, s) for r, s in zip(list_rows.tolist(), list_sims.tolist()) if r >= 0 and r not in context]
                for rank, (r, s) in enumerate(keep[: self.depth], start=1):
                    if self.fusion == "max":
                        fused[r] = max(fused.get(r, -np.inf), s)
                    else:
                        fused[r] = fused.get(r, 0.0) + 1.0 / (self.rrf_k + rank)
            top = sorted(fused.items(), key=lambda kv: -kv[1])[: self.num_results]
            for docno, (_, s) in zip(self.store.docnos([r for r, _ in top]), top):
                records.append({"qid": qid, "docno": docno, "score": s})
        return _results_frame(records, inp.drop(columns=["docno", "score", "rank"], errors="ignore"))


class DocVectors(pt.Transformer):
    """Adds each result's stored E5 vector as `doc_vec`, for vector re-rankers such as MMR."""

    def __init__(self, store):
        self.store = store

    def transform(self, inp: pd.DataFrame) -> pd.DataFrame:
        pt.validate.result_frame(inp)
        return inp.assign(doc_vec=list(self.store.vectors(self.store.rows(inp["docno"].tolist()))))


def seed_mmr(store, num_results: int = 1000, Lambda: float = 0.7, depth: int | None = None,
             pool: int = 2000) -> pt.Transformer:
    """Multi-query MMR: every context document acts as a query.

    Candidates are the union of the context documents' top-`depth` neighbours (context
    removed), with relevance = the highest cosine to any context document (as in max seed
    fusion). The `pool` most relevant candidates are re-ranked by pyterrier-dr's MmrScorer,
    which picks next the document maximising
        Lambda * relevance - (1 - Lambda) * max cosine to the documents already picked,
    and the top `num_results` are kept. Lambda = 1 would be plain max seed fusion.
    """
    if not 0.0 <= Lambda < 1.0:
        raise ValueError(f"Lambda must be in [0, 1), not {Lambda}; Lambda=1 is SeedFusionRetriever(fusion='max')")
    from pyterrier_dr import MmrScorer

    candidates = SeedFusionRetriever(store, max(pool, num_results), fusion="max", depth=depth or num_results)
    return (candidates >> DocVectors(store) >> MmrScorer(Lambda=Lambda)) % num_results


class InverseContextMMR(pt.Transformer):
    """Inverse context MMR (seed_inv_mmr@Lambda): every context document acts as a query.

    Candidates are the union of the context documents' top-`depth` neighbours (context
    removed), keeping the `pool` with the highest cosine to any context document. For a
    candidate d, q is the context document d is most similar to (as in seed_mmr), and

        S(d) = -Lambda * S_R(d, q) + (1 - Lambda) * S_c(d, C)

    with S_R(d, q) = cos(d, q) and S_c(d, C) the mean cosine between d and the other
    context documents (0 when the context is q alone). Unlike MMR, C is fixed rather than
    the documents picked so far, so every candidate is scored once and the pool sorted.
    """

    def __init__(self, store, num_results: int = 1000, Lambda: float = 0.7, depth: int | None = None,
                 pool: int = 2000):
        if not 0.0 <= Lambda <= 1.0:
            raise ValueError(f"Lambda must be in [0, 1], not {Lambda}")
        self.store, self.num_results, self.Lambda = store, num_results, Lambda
        self.candidates = SeedFusionRetriever(store, max(pool, num_results), fusion="max", depth=depth or num_results)

    def transform(self, inp: pd.DataFrame) -> pd.DataFrame:
        pt.validate.columns(inp, includes=["qid", "seed_docnos"])
        cands = self.candidates(inp[["qid", "seed_docnos"]])
        records = []
        for qid, seeds in zip(inp["qid"], inp["seed_docnos"]):
            c = cands[cands["qid"] == qid]
            if c.empty:
                continue
            context = self.store.vectors(_context_rows(self.store, seeds))  # (k, dim)
            sims = self.store.vectors(self.store.rows(c["docno"].tolist())) @ context.T  # (m, k)
            s_r = sims.max(axis=1)
            k = sims.shape[1]
            s_c = (sims.sum(axis=1) - s_r) / (k - 1) if k > 1 else np.zeros(len(s_r))
            scores = -self.Lambda * s_r + (1.0 - self.Lambda) * s_c
            order = np.argsort(-scores, kind="stable")[: self.num_results]
            records += [{"qid": qid, "docno": d, "score": float(sc)}
                        for d, sc in zip(c["docno"].to_numpy()[order], scores[order])]
        return _results_frame(records, inp.drop(columns=["docno", "score", "rank"], errors="ignore"))


class ContextPassageQueries(pt.Transformer):
    """[qid, seed_docnos] -> adds `queries`: the text of each context passage, in
    seed_docnos order (empty texts dropped), each to be used as a query."""

    def __init__(self, texts):
        self.texts = texts

    def transform(self, inp: pd.DataFrame) -> pd.DataFrame:
        pt.validate.columns(inp, includes=["qid", "seed_docnos"])
        get = lambda d: self.texts.get(d) if hasattr(self.texts, "get") else self.texts[d]
        queries = [tuple(t for t in (get(d) for d in seeds) if isinstance(t, str) and t.strip())
                   for seeds in inp["seed_docnos"]]
        return inp.assign(queries=queries)


def seed_bm25_rrf(texts, bm25, num_results: int = 1000, rrf_k: int = 60) -> pt.Transformer:
    """bm25_DaaQ: each context passage's text is a BM25 query; `bm25` is a
    pec.text_retrievers.ResidualTextRetriever (top-K per query, context removed), and the
    per-passage rankings are fused by reciprocal rank fusion (pyterrier-alpha rr_fusion,
    one input per passage), as the multi-query query formulation methods are."""
    from pec.query_formulation import MultiQueryRetriever

    # the cutoff also applies to one-passage contexts, whose BM25 ranking is not fused
    return (ContextPassageQueries(texts) >> MultiQueryRetriever(bm25, num_results, rrf_k)) % num_results


def sparse_centroid(passages: list[str], analyse, terms: int | None = 100) -> dict[str, float]:
    """The context's centroid in a sparse term space: each passage's term counts divided
    by its length (in index terms), averaged over the passages; the `terms` highest weights
    are kept (None keeps every term) and scaled so the largest is 1. `analyse` maps a text
    to the index's terms (tokenised, stopwords removed, stemmed)."""
    vectors = []
    for text in passages:
        toks = analyse(text) if isinstance(text, str) else []
        if toks:
            vectors.append({t: c / len(toks) for t, c in Counter(toks).items()})
    if not vectors:
        return {}
    mean: dict[str, float] = {}
    for v in vectors:
        for t, w in v.items():
            mean[t] = mean.get(t, 0.0) + w / len(vectors)
    ranked = sorted(mean.items(), key=lambda kv: (-kv[1], kv[0]))
    if terms is not None:
        ranked = ranked[:terms]
    top = ranked[0][1]
    return {t: w / top for t, w in ranked}


class SparseCentroidRetriever(pt.Transformer):
    """bm25_centroid: the context centroid in BM25's term space (pec.pipeline.sparse_centroid
    of the context passages' texts), searched as one weighted query: `retriever` maps
    [qid, query_toks] to a ranking (PISA BM25 with query_weighted=True), so a document
    scores sum_t w_t * BM25(t, d), the centroid's dot product with the document's BM25
    vector. Context documents are removed; `num_results` are kept."""

    def __init__(self, texts, analyse, retriever, num_results: int = 1000, terms: int | None = 100):
        self.texts, self.analyse, self.retriever = texts, analyse, retriever
        self.num_results, self.terms = num_results, terms

    def query_toks(self, seeds) -> dict[str, float]:
        get = lambda d: self.texts.get(d) if hasattr(self.texts, "get") else self.texts[d]
        return sparse_centroid([get(d) for d in seeds], self.analyse, self.terms)

    def transform(self, inp: pd.DataFrame) -> pd.DataFrame:
        pt.validate.columns(inp, includes=["qid", "seed_docnos"])
        queries = pd.DataFrame({"qid": inp["qid"].tolist(),
                                "query_toks": [self.query_toks(s) for s in inp["seed_docnos"]]})
        queries = queries[queries["query_toks"].map(len) > 0].reset_index(drop=True)
        if queries.empty:
            return pd.DataFrame(columns=["qid", "docno", "score", "rank"])
        res = self.retriever(queries)
        exclude = {(q, str(d)) for q, seeds in zip(inp["qid"], inp["seed_docnos"]) for d in seeds}
        res = res[[(q, str(d)) not in exclude for q, d in zip(res["qid"], res["docno"])]]
        res = res[["qid", "docno", "score"]].astype({"docno": str}).sort_values(["qid", "score"], ascending=[True, False])
        res = res.groupby("qid", sort=False).head(self.num_results).reset_index(drop=True)
        return pt.model.add_ranks(res)
