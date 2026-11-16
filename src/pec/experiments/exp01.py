"""Experiment 1: zero-training baselines against the true query, swept over context sizes.

Runs per dataset (TREC-DL19, TREC-DL20, TREC-COVID, ...). Each dataset's relevance scheme
says which grade is the context (DL and COVID: 1) and which is the target (DL: 3, COVID:
2). Every system gets contexts of m partially relevant passages (m in context_sizes, with
`repeats` nested random draws per m) and is scored against the target documents, with
the context removed from every ranking:

- E5 baselines: the context centroid, per-document fusion by RRF (seed_rrf), multi-query
  MMR (seed_mmr@lambda), inverse context MMR (seed_inv_mmr@lambda)
- BM25 baselines: each context passage as a BM25 query fused by RRF (bm25_DaaQ), and the
  context centroid in BM25's term space (bm25_centroid), optionally without BM25's term
  weighting (*_nobm25)
- the text query formulation baselines of proactive-icl-query-pair (zero-shot and static
  ICL LLM queries, sliding window, doc2query), through E5 and that repository's text
  retrievers
- the true query (the Oracle), with E5 and with every text retriever

Outputs go to outputs/exp01_<dataset>/baseline/; with several datasets,
outputs/exp01_comparison/ gathers every dataset's best m.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from pec.analysis.plots import SERIES, apply_style, lines, save
from pec.analysis.oracle import ABLATION_SUFFIX, pinned_value, pinned_values, warn_unused_pins
from pec.data.qrels import DL_RELEVANCE, GRADES, Relevance, grade_sets

log = logging.getLogger(__name__)
PARTS = ("baseline",)


def _json_default(o):
    return float(o) if isinstance(o, (np.floating, np.integer)) else str(o)


@dataclass
class DatasetContext:
    topics: pd.DataFrame
    qrels: pd.DataFrame
    store: object
    rows_by_grade: dict[str, dict[int, np.ndarray]]
    labels: dict[str, dict[int, int]]
    query_vecs: dict[str, np.ndarray] | None = None
    texts: dict[str, str] | None = None
    missing_docnos: int = 0
    index_key: str = "msmarco_v1"
    relevance: Relevance = DL_RELEVANCE
    dataset: str = "dl19"
    _vec_cache: dict[int, np.ndarray] = field(default_factory=dict, repr=False)

    def vecs(self, rows) -> np.ndarray:
        rows = np.asarray(rows, dtype=np.int64)
        missing = [r for r in rows.tolist() if r not in self._vec_cache]
        if missing:
            for r, v in zip(missing, self.store.vectors(np.asarray(missing, dtype=np.int64))):
                self._vec_cache[r] = v
        if len(rows) == 0:
            return np.zeros((0, self.store.dim), dtype=np.float32)
        return np.stack([self._vec_cache[r] for r in rows.tolist()])

    def judged(self, qid: str) -> tuple[np.ndarray, np.ndarray]:
        by_grade = self.rows_by_grade[qid]
        rows = np.concatenate([by_grade[g] for g in GRADES])
        grades = np.concatenate([np.full(len(by_grade[g]), g) for g in GRADES])
        return rows, grades

    def query_text(self, qid: str) -> str | None:
        hit = self.topics.loc[self.topics["qid"] == qid, "query"]
        return str(hit.iloc[0]) if len(hit) else None


def build_context(store, topics: pd.DataFrame, qrels: pd.DataFrame, query_vecs=None, texts=None,
                  index_key: str = "msmarco_v1", relevance: Relevance = DL_RELEVANCE,
                  dataset: str = "dl19") -> DatasetContext:
    rows_by_grade, labels, missing = {}, {}, 0
    for qid, by_grade in grade_sets(qrels).items():
        rows_by_grade[qid] = {}
        for g in GRADES:
            rows = store.rows(by_grade[g])
            missing += int((rows < 0).sum())
            rows_by_grade[qid][g] = rows[rows >= 0]
        labels[qid] = {int(r): g for g in GRADES for r in rows_by_grade[qid][g].tolist()}
    if missing:
        log.warning("%s: %d judged docnos are not in the index and are ignored", dataset, missing)
    return DatasetContext(topics, qrels, store, rows_by_grade, labels, query_vecs, texts, missing, index_key,
                          relevance, dataset)


def load_context(cfg: dict, dataset: str | None = None, section: str = "exp01") -> DatasetContext:
    """The dataset's index, topics, qrels and (with <section>.encode_queries) E5 query
    vectors; `section` is the experiment's config section (exp01, exp02)."""
    from pec.data.datasets import load_dataset, relevance
    from pec.index.faiss_store import FaissDocStore

    ec = cfg[section]
    dataset = dataset or ec.get("dataset", "dl19")
    index_key = cfg["datasets"][dataset]["index"]
    store = FaissDocStore.from_config(cfg, index_key)
    topics, qrels = load_dataset(cfg, dataset)
    query_vecs = None
    if ec.get("encode_queries", True):
        from pec.index.e5 import encode_topics

        query_vecs = encode_topics(topics, cfg["e5_model"])
    return build_context(store, topics, qrels, query_vecs, index_key=index_key,
                         relevance=relevance(cfg, dataset), dataset=dataset)


# ------------------------------------------------------------------------ baseline

def baseline_pipelines(store, c: dict, query_vecs: dict | None, reference: str = "true_query", texts=None,
                       query_encoder=None, qf_cache_dir: Path | None = None, text_retrievers: dict | None = None,
                       topics: pd.DataFrame | None = None, bm25_retriever=None,
                       bm25_centroid=None, bm25_ablation: dict | None = None) -> tuple[dict, dict]:
    """Zero-training systems for the context-size sweep, and their figure groups
    (pec.evaluate.figure_groups: main, mmr, inv_mmr, llm, text, qf_<retriever>).

    - c["other_baselines"] (default true): the embedding baselines (centroid, seed_rrf,
      MMR, inverse context MMR) and BM25 + RRF; false runs only the query
      formulation methods (and the true query, every share's reference).
    - c["inv_mmr"]["lambdas"]: seed_inv_mmr@<lambda> (pec.pipeline.InverseContextMMR).
    - c["bm25_rrf"]["enabled"]: bm25_DaaQ, each context passage's `texts` searched with
      `bm25_retriever` (a ResidualTextRetriever over the BM25 index) and fused by RRF.
    - c["bm25_centroid"]["enabled"]: bm25_centroid, the context centroid in BM25's term
      space searched as one weighted BM25 query (pec.pipeline.SparseCentroidRetriever);
      `bm25_centroid` is (analyse, weighted BM25 retriever), see bm25_centroid_parts.
      bm25_centroid.terms (default 100) prunes the centroid; all_terms keeps every term.
    - c["bm25_binary_ablation"]: the same two methods without BM25's term weighting, as
      separate runs bm25_DaaQ_nobm25 / bm25_centroid_nobm25: `bm25_ablation` holds
      their retrievers (BM25 with k1 = 0, see pisa_bm25).
    - With bm25_DaaQ or bm25_centroid, the true query text is also searched with BM25
      (true_query@bm25): their Oracle in the figures.

    The query formulation baselines (c["query_formulation"]["methods"], limited to
    query_formulation.groups: llm = zero_shot / static_icl, text = sliding_window /
    doc2query) need the context passages' `texts` (docno -> text), a `query_encoder` for
    their E5 runs and, for the text retrievers in c["query_formulation"]["retrievers"],
    `text_retrievers` (name -> pec.text_retrievers.ResidualTextRetriever). With `topics`
    and query_formulation.true_query, the true query text also goes through every text
    retriever (true_query@<retriever>, its upper bound on that retriever)."""
    from pec.evaluate import figure_groups
    from pec.pipeline import (CentroidEncoder, FaissRetriever, FixedQueryVectors, InverseContextMMR, SeedFusionRetriever,
                              SparseCentroidRetriever, seed_bm25_rrf, seed_mmr)
    from pec.query_formulation import query_formulation_pipelines

    k = c["num_results"]
    pipes = {}
    if c.get("other_baselines", True):
        pipes.update({"centroid": CentroidEncoder(store) >> FaissRetriever(store, k),
                      "seed_rrf": SeedFusionRetriever(store, k, fusion="rrf", rrf_k=c.get("rrf_k", 60))})
        mmr = c.get("mmr") or {}
        for lam in mmr.get("lambdas", []):
            pipes[f"seed_mmr@{lam:g}"] = seed_mmr(store, k, Lambda=float(lam), pool=mmr.get("pool", 2000))
        inv = c.get("inv_mmr") or {}
        for lam in inv.get("lambdas", []):
            pipes[f"seed_inv_mmr@{lam:g}"] = InverseContextMMR(store, k, Lambda=float(lam), pool=inv.get("pool", 2000))
        bm25 = c.get("bm25_rrf") or {}
        if bm25.get("enabled"):
            if texts is None or bm25_retriever is None:
                raise ValueError("bm25_DaaQ needs the context passages' texts and a BM25 retriever")
            pipes["bm25_DaaQ"] = seed_bm25_rrf(texts, bm25_retriever, k, bm25.get("rrf_k", 60))
        bc = c.get("bm25_centroid") or {}
        if bc.get("enabled"):
            if texts is None or bm25_centroid is None:
                raise ValueError("bm25_centroid needs the context passages' texts and the BM25 index's analyser "
                                 "and weighted retriever")
            analyse, weighted = bm25_centroid
            pipes["bm25_centroid"] = SparseCentroidRetriever(texts, analyse, weighted, k, bm25_centroid_terms(bc))
        if c.get("bm25_binary_ablation") and (bm25.get("enabled") or bc.get("enabled")):
            ab = bm25_ablation or {}
            if bm25.get("enabled"):
                if "bm25_DaaQ" not in ab:
                    raise ValueError("the bm25_DaaQ ablation needs a BM25 retriever with k1 = 0")
                pipes["bm25_DaaQ" + ABLATION_SUFFIX] = seed_bm25_rrf(texts, ab["bm25_DaaQ"], k,
                                                                          bm25.get("rrf_k", 60))
            if bc.get("enabled"):
                if "bm25_centroid" not in ab:
                    raise ValueError("the bm25_centroid ablation needs a weighted BM25 retriever with k1 = 0")
                analyse, weighted = ab["bm25_centroid"]
                pipes["bm25_centroid" + ABLATION_SUFFIX] = SparseCentroidRetriever(texts, analyse, weighted, k,
                                                                                   bm25_centroid_terms(bc))
    qf = c.get("query_formulation") or {}
    methods = qf_methods(qf)
    if methods:
        if texts is None or ("e5" in qf_retrievers(qf) and query_encoder is None):
            raise ValueError("the query formulation baselines need the context passages' texts and a query encoder")
        pipes.update(query_formulation_pipelines(store, {**qf, "methods": methods}, texts, query_encoder, k,
                                                 qf_cache_dir, text_retrievers=text_retrievers))
    if query_vecs:
        pipes[reference] = FixedQueryVectors(query_vecs) >> FaissRetriever(store, k)
    text_names = [r for r in qf_retrievers(qf) if r != "e5"] if methods else []
    oracle_retrievers = {r: text_retrievers[r] for r in text_names} if qf.get("true_query", True) else {}
    if bm25_retriever is not None and any(n in pipes for n in ("bm25_DaaQ", "bm25_centroid")):
        oracle_retrievers.setdefault("bm25", bm25_retriever)  # the BM25 methods' Oracle
    if oracle_retrievers and topics is not None:
        import pyterrier as pt

        add_query = pt.apply.generic(lambda df: df.merge(topics[["qid", "query"]], on="qid"))
        for r, retriever in oracle_retrievers.items():
            pipes[f"{reference}@{r}"] = add_query >> retriever
    return pipes, figure_groups(pipes, reference)


QF_GROUPS = ("llm", "text")
BM25_CENTROID_TERMS = 100  # terms kept in the BM25 centroid unless bm25_centroid.all_terms


def bm25_centroid_terms(b: dict) -> int | None:
    """Terms kept in the BM25 centroid: None (every term) with bm25_centroid.all_terms,
    else bm25_centroid.terms (default 100)."""
    if b.get("all_terms"):
        return None
    terms = int(b.get("terms") or BM25_CENTROID_TERMS)
    if terms < 1:
        raise ValueError(f"bm25_centroid.terms must be at least 1, not {terms}")
    return terms


def pisa_bm25(cfg: dict, dataset: str, depth: int, k1: float | None = None, weighted: bool = False):
    """BM25 over the dataset's PISA index (text_retrievers.<index>.bm25), `depth` deep:
    PISA's defaults (k1 0.9, b 0.4), or with k1 = 0 binary term matching (a matching term
    scores its idf, whatever its frequency in the document or the document's length).
    `weighted` takes [qid, query_toks] weighted queries. Returns (index, retriever)."""
    from pec.text_retrievers import _pisa_index

    entry = cfg["datasets"][dataset]
    paths = (cfg.get("text_retrievers") or {}).get(entry["index"]) or {}
    if "bm25" not in paths:
        raise KeyError(f"no text_retrievers.{entry['index']}.bm25 in configs/paths.yaml")
    index = _pisa_index(paths["bm25"], "BM25 PISA index")
    kwargs = {"num_results": depth, "query_weighted": True if weighted else None}
    if k1 is not None:
        kwargs["k1"] = k1
    return index, index.bm25(**kwargs)


def bm25_binary_text_retriever(cfg: dict, dataset: str, num_results: int, depth: int):
    """bm25_DaaQ's ablation retriever: [qid, query, seed_docnos] -> BM25 with k1 = 0
    (binary term matching), context removed."""
    from pec.text_retrievers import ResidualTextRetriever

    return ResidualTextRetriever(pisa_bm25(cfg, dataset, depth, k1=0.0)[1], num_results)


def bm25_centroid_parts(cfg: dict, dataset: str, depth: int, k1: float | None = None):
    """(analyse, retriever) for bm25_centroid over the dataset's BM25 PISA index
    (text_retrievers.<index>.bm25 in paths.yaml): `analyse` turns a text into the index's
    terms (PISA's tokeniser, the index's stopwords removed, then its stemmer), and
    `retriever` is BM25 taking weighted [qid, query_toks] queries, `depth` documents deep
    (k1 = 0: the binary-matching ablation). Nothing is indexed: the existing index is only
    read."""
    import pyterrier_pisa as ptp
    from pyterrier_pisa.stopwords import _STOPWORDS

    index, retriever = pisa_bm25(cfg, dataset, depth, k1=k1, weighted=True)
    stops = index.stops
    if isinstance(stops, (list, tuple, set)):
        stopset = {str(w) for w in stops}
    else:
        name = ptp.PisaStopwords(stops).value
        stopset = set() if name == "none" else set(_STOPWORDS[name])

    def analyse(text: str) -> list[str]:
        words = [w for w in ptp.tokenize(text) if w not in stopset]  # unstemmed, so stopwords match
        return index.tokenize(" ".join(words)) if words else []

    return analyse, retriever


def qf_methods(qf: dict) -> list[str]:
    """Query formulation methods to run: query_formulation.methods within
    query_formulation.groups ("llm": zero_shot, static_icl; "text": sliding_window,
    doc2query; default both)."""
    from pec.query_formulation import LLM_METHODS

    groups = set(qf.get("groups") or QF_GROUPS)
    unknown = groups - set(QF_GROUPS)
    if unknown:
        raise ValueError(f"unknown query formulation groups {sorted(unknown)}; choose from {QF_GROUPS}")
    return [m for m in (qf.get("methods") or []) if ("llm" if m in LLM_METHODS else "text") in groups]


def qf_run_names(qf: dict, reference: str = "true_query") -> list[str]:
    """Run names of the query formulation baselines (and their true_query@<retriever>)."""
    from pec.query_formulation import run_name

    methods = qf_methods(qf)
    names = [run_name(m, r) for m in methods for r in qf_retrievers(qf)]
    if methods and qf.get("true_query", True):
        names += [f"{reference}@{r}" for r in qf_retrievers(qf) if r != "e5"]
    return names


def qf_retrievers(qf: dict) -> list[str]:
    """Retrievers the query formulation baselines' queries are evaluated with."""
    return list(qf.get("retrievers") or ["e5"])


def _load_texts(ctx: DatasetContext, cfg: dict | None) -> dict[str, str] | None:
    """Texts of the judged documents (the dataset's text_fields), loaded once."""
    if ctx.texts is not None or cfg is None:
        return ctx.texts
    try:
        from pec.data.datasets import text_loader

        docnos = sorted(set(ctx.qrels["docno"]))
        frame = text_loader(cfg, ctx.dataset)(pd.DataFrame({"qid": "0", "docno": docnos}))
        ctx.texts = dict(zip(frame["docno"], frame["text"]))
    except Exception as e:  # corpus not downloadable here
        log.warning("document text unavailable (%s)", e)
    return ctx.texts


def _qf_texts(ctx: DatasetContext, c: dict, cfg: dict | None) -> dict[str, str] | None:
    """Texts of the context passages for the query formulation baselines: the dataset's
    text fields (as _load_texts), or c["query_formulation"]["text_fields"] when set."""
    fields = (c.get("query_formulation") or {}).get("text_fields")
    if not fields:
        return _load_texts(ctx, cfg)
    if cfg is None:
        return None
    from pec.config import deep_merge
    from pec.data.datasets import text_loader

    docnos = sorted(set(ctx.qrels.loc[ctx.qrels["label"] == ctx.relevance.context, "docno"]))
    loader = text_loader(deep_merge(cfg, {"datasets": {ctx.dataset: {"text_fields": list(fields)}}}), ctx.dataset)
    frame = loader(pd.DataFrame({"qid": "0", "docno": docnos}))
    return dict(zip(frame["docno"], frame["text"]))


def run_baseline(ctx: DatasetContext, c: dict, out: Path, random_seed: int = 0, cfg: dict | None = None,
                 query_encoder=None, text_retrievers: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Zero-training baselines swept over context sizes. Returns (per-query metrics, best k).

    Results already stored in baseline/ are reused (pec.evaluate.cached_sweep): a run that
    has results for a context size and repeat is not run again unless c["rerun"] is true
    or lists it, and runs made in separate invocations accumulate in one set of tables and
    figures. The query formulation baselines see the same context draws as every other
    baseline, limited to query_formulation.context_sizes. Their queries are E5-encoded with
    cfg["e5_model"] (or `query_encoder`) and run through the text retrievers of
    proactive-icl-query-pair (cfg["text_retrievers"], or `text_retrievers`); their outputs
    are kept in baseline/query_formulation/<method>.jsonl, which reruns reuse."""
    from pec.data.seeds import context_draws, size_label
    from pec.evaluate import cached_sweep, figure_groups, metric_names, relative_to_reference, write_context_sweep

    rel = ctx.relevance
    out = out / "baseline"
    out.mkdir(parents=True, exist_ok=True)
    reference = c.get("reference", "true_query")
    metric_cols = metric_names(c["recall_ks"])
    k = c["num_results"]
    contexts = context_draws(ctx.qrels, c["context_sizes"], c["repeats"], random_seed,
                             require_grade=rel.target, context_grade=rel.context)
    max_context = max((len(s) for s in contexts["seed_docnos"]), default=0)
    qf = c.get("query_formulation") or {}
    methods = qf_methods(qf)
    bm25_cfg = c.get("bm25_rrf") or {}
    use_bm25 = bool(c.get("other_baselines", True) and bm25_cfg.get("enabled"))
    use_bm25_centroid = bool(c.get("other_baselines", True) and (c.get("bm25_centroid") or {}).get("enabled"))
    texts = qf_dir = bm25_retriever = None
    if methods or use_bm25 or use_bm25_centroid:
        texts = _qf_texts(ctx, c, cfg)
        if texts is None:
            raise RuntimeError(f"{ctx.dataset}: the query formulation, BM25 + RRF and BM25 centroid baselines need "
                               "the passage texts, which could not be loaded")
    if methods:
        if query_encoder is None and "e5" in qf_retrievers(qf):
            from pec.index.e5 import e5_model

            query_encoder = e5_model((cfg or {}).get("e5_model", "intfloat/e5-base-v2")).query_encoder()
        text_names = [r for r in qf_retrievers(qf) if r != "e5"]
        if text_names and text_retrievers is None:
            from pec.text_retrievers import MONOT5_DEPTH, build_retrievers

            if cfg is None:
                raise ValueError("the text retrievers need the config (text_retrievers in paths.yaml)")
            # deep enough that num_results documents remain once the context is removed
            text_retrievers = build_retrievers(cfg, ctx.dataset, text_names, k, k + max_context,
                                               qf.get("monot5_depth", MONOT5_DEPTH))
        qf_dir = out / "query_formulation"
    if use_bm25 or use_bm25_centroid:  # bm25_DaaQ, and the BM25 methods' Oracle (true_query@bm25)
        top_k = int(bm25_cfg.get("depth", k))
        if text_retrievers and "bm25" in text_retrievers and top_k == k:
            bm25_retriever = text_retrievers["bm25"]
        else:
            from pec.text_retrievers import build_retrievers

            if cfg is None:
                raise ValueError("the BM25 baselines need the config (text_retrievers.<index>.bm25 in paths.yaml)")
            bm25_retriever = build_retrievers(cfg, ctx.dataset, ["bm25"], top_k, top_k + max_context)["bm25"]
    bm25_centroid = None
    if use_bm25_centroid:
        if cfg is None:
            raise ValueError("bm25_centroid needs the config (text_retrievers.<index>.bm25 in paths.yaml)")
        bm25_centroid = bm25_centroid_parts(cfg, ctx.dataset, k + max_context)
    bm25_ablation = {}
    if c.get("bm25_binary_ablation") and (use_bm25 or use_bm25_centroid):
        if cfg is None:
            raise ValueError("the BM25 ablation needs the config (text_retrievers.<index>.bm25 in paths.yaml)")
        if use_bm25:
            top_k = int(bm25_cfg.get("depth", k))
            bm25_ablation["bm25_DaaQ"] = bm25_binary_text_retriever(cfg, ctx.dataset, top_k, top_k + max_context)
        if use_bm25_centroid:
            bm25_ablation["bm25_centroid"] = bm25_centroid_parts(cfg, ctx.dataset, k + max_context, k1=0.0)
    pipes, _ = baseline_pipelines(ctx.store, c, ctx.query_vecs, reference, texts, query_encoder, qf_dir,
                                  text_retrievers, ctx.topics, bm25_retriever, bm25_centroid, bm25_ablation)
    qf_sizes = qf.get("context_sizes")
    run_sizes = {n: {size_label(x) for x in qf_sizes} for n in qf_run_names(qf, reference)} if qf_sizes else None
    settings = {"dataset": ctx.dataset, "random_seed": random_seed, "num_results": k,
                "recall_ks": [int(x) for x in c["recall_ks"]],
                "views": {v: [int(g) for g in grades] for v, grades in rel.views.items()},
                "context_grade": rel.context, "target_grade": rel.target}
    metrics, perquery, executed = cached_sweep(pipes, contexts, ctx.qrels, c["recall_ks"], out, settings, rel,
                                               run_sizes, c.get("rerun", False))
    log.info("%s baseline: ran %s; %d runs with results in total", ctx.dataset, executed or "nothing new",
             metrics["name"].nunique())
    perquery = relative_to_reference(perquery, metric_cols, reference)
    perquery.to_csv(out / "metrics_perquery.csv", index=False)
    groups = figure_groups(set(metrics["name"]), reference)
    pins = pinned_values(cfg, ctx.dataset)
    if c.get("pin_oracle", True):
        warn_unused_pins(pins, metrics, ctx.dataset, reference)
    _, best = write_context_sweep(metrics, metric_cols, out, groups.pop("main"), reference, figure_groups=groups,
                                  style=figure_style(c, pins))
    return perquery, best


def figure_style(c: dict, pinned: dict | None = None):
    """Experiment 1's figure style (baseline section): the Oracle called "Oracle", the
    context size "m", with fixed_oracle (default) every m compared with the Oracle at
    m = oracle_context_size (default 5), a flat line, and with pin_oracle (default) the
    `pinned` Oracle scores (configs/oracle_values.yaml) used wherever they are given."""
    from pec.data.seeds import size_label
    from pec.evaluate import SweepFigureStyle

    size = size_label(c.get("oracle_context_size", 5)) if c.get("fixed_oracle", True) else None
    pins = (pinned or None) if c.get("pin_oracle", True) else None
    return SweepFigureStyle(oracle_size=size, label=c.get("oracle_label", "Oracle"), size_symbol="m", pinned=pins)


# ------------------------------------------------------------------------- driver

def run_parts(ctx: DatasetContext, cfg: dict, out: Path, parts=PARTS) -> dict:
    """Run the parts (only "baseline") for one dataset; figures have titles only with exp01.plot_titles, and
    the result figures use the fixed y ranges of cfg["y_limits"] (configs/plot_results.yaml)."""
    from pec.analysis.plots import figure_titles, y_limits

    with figure_titles(cfg["exp01"].get("plot_titles", False)), y_limits(cfg.get("y_limits")):
        return _run_parts(ctx, cfg, out, parts)


def _run_parts(ctx: DatasetContext, cfg: dict, out: Path, parts=PARTS) -> dict:
    ec = cfg["exp01"]
    apply_style()
    summary: dict = {"dataset": ctx.dataset, "queries": len(ctx.rows_by_grade),
                     "judged_docnos_missing_from_index": ctx.missing_docnos,
                     "context_grade": ctx.relevance.context, "target_grade": ctx.relevance.target}
    for part in parts:
        log.info("exp01 %s part: %s", ctx.dataset, part)
        if part == "baseline":
            _, best = run_baseline(ctx, ec["baseline"], out, ec.get("random_seed", 0), cfg)
            centroid = best[(best["name"] == "centroid") & (best["metric"] == "nDCG@10")]
            summary["baseline_best_k_centroid_ndcg10"] = {
                r["view"]: {"best_k": r["best_k"], "pct_true_query": r["best_pct"], "pct_true_query_at_all": r["all_pct"]}
                for _, r in centroid.iterrows()}
        else:
            raise ValueError(f"unknown exp01 part {part!r}; choose from {PARTS}")
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=_json_default)
    return summary


def dataset_config(cfg: dict, dataset: str) -> dict:
    """The config with exp01.overrides.<dataset> merged into exp01 (e.g. TREC-COVID's
    context sizes without "all")."""
    from pec.config import deep_merge

    ec = cfg["exp01"]
    merged = deep_merge(cfg, {"exp01": (ec.get("overrides") or {}).get(dataset, {})})
    merged["exp01"]["dataset"] = dataset
    return merged


def write_comparison(cfg: dict, datasets: list[str], out: Path) -> pd.DataFrame | None:
    """outputs/exp01_comparison: every dataset's best k in one table, and the centroid's
    nDCG@10 share of the Oracle against m on each dataset's target view (the Oracle at
    m = 5 unless baseline.fixed_oracle is false, as in the baseline figures)."""
    import matplotlib.pyplot as plt

    from pec.data.datasets import relevance
    from pec.data.seeds import ALL, ordered_sizes
    from pec.evaluate import drop_retired

    ec = cfg["exp01"]
    best, curves = [], {}
    for ds in datasets:
        base = Path(cfg.get("output_dir", "outputs")) / ec["output_subdir"].format(dataset=ds) / "baseline"
        if not (base / "best_context_size.csv").exists():
            continue
        b = drop_retired(pd.read_csv(base / "best_context_size.csv", dtype={"best_k": str}))
        best.append(b.assign(dataset=ds, target_view=b["view"] == relevance(cfg, ds).target_view))
        s = pd.read_csv(base / "summary_by_context_size.csv", dtype={"context_size": str})
        s = s[(s["view"] == relevance(cfg, ds).target_view) & (s["context_size"] != ALL)]
        target = relevance(cfg, ds).target_view
        style = figure_style(dataset_config(cfg, ds)["exp01"]["baseline"], pinned_values(cfg, ds))
        oracle = s[(s["name"] == "true_query") & (s["context_size"] == style.oracle_size)]
        pin = pinned_value(style.pinned, "nDCG@10", "true_query", target)
        s = s[s["name"] == "centroid"].copy()
        ref = pin if pin is not None else (float(oracle["nDCG@10_mean"].iloc[0])
                                           if style.oracle_size and len(oracle) else None)
        if ref is not None:  # every m against one Oracle value: pinned, or at the fixed m
            s["nDCG@10_pct_true_query_mean"] = 100.0 * s["nDCG@10_mean"] / ref
            s["nDCG@10_pct_true_query_std"] = 100.0 * s["nDCG@10_std"] / ref
        curves[ds] = s.assign(oracle_context_size=style.oracle_size if ref is not None and pin is None else None,
                              oracle_ndcg10=ref, oracle_pinned=pin is not None)
    if not best:
        return None
    out.mkdir(parents=True, exist_ok=True)
    table = pd.concat(best, ignore_index=True)
    table.insert(0, "dataset", table.pop("dataset"))
    table.to_csv(out / "best_context_size.csv", index=False)

    apply_style()
    sizes = ordered_sizes(pd.concat(curves.values())["context_size"])
    pos = {s: i for i, s in enumerate(sizes)}
    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    xs, ys = {}, {}
    for i, (ds, s) in enumerate(list(curves.items())[: len(SERIES)]):
        s = s.sort_values("context_size", key=lambda c: c.map(pos))
        x = s["context_size"].map(pos).to_numpy(dtype=float)
        y, std = s["nDCG@10_pct_true_query_mean"].to_numpy(), s["nDCG@10_pct_true_query_std"].fillna(0).to_numpy()
        ax.fill_between(x, y - std, y + std, color=SERIES[i], alpha=0.15, linewidth=0)
        label = f"{ds} ({relevance(cfg, ds).target_view})"
        xs[label], ys[label] = x, y
    style = figure_style(ec["baseline"])
    lines(ax, xs, ys, log2_x=False, xlabel=style.xlabel(), ylabel=f"% of {style.label}", ykind="pct")
    ax.axhline(100, color="#898781", linewidth=1, linestyle="--", label=style.label)
    ax.legend(fontsize=7)
    ax.set_xticks(range(len(sizes)), sizes)
    ax.set_title(f"Centroid nDCG@10 on each dataset's target documents{style.note()}")
    save(fig, out / "centroid_ndcg10_by_dataset.png")
    pd.concat([s.assign(dataset=ds) for ds, s in curves.items()], ignore_index=True).to_csv(
        out / "centroid_ndcg10_by_dataset.csv", index=False)
    return table


def run_exp01(cfg: dict, parts=None, datasets=None) -> dict:
    """Run the baseline sweep for every dataset in exp01.datasets (or `datasets`)."""
    from pec.config import output_dir

    ec = cfg["exp01"]
    datasets = list(datasets or ec.get("datasets") or [ec.get("dataset", "dl19")])
    summaries = {}
    for ds in datasets:
        dcfg = dataset_config(cfg, ds)
        out = output_dir(dcfg, dcfg["exp01"]["output_subdir"].format(dataset=ds))
        summaries[ds] = run_parts(load_context(dcfg, ds), dcfg, out, parts or dcfg["exp01"].get("parts", PARTS))
    if len(datasets) > 1 and "baseline" in (parts or ec.get("parts", PARTS)):
        from pec.analysis.plots import figure_titles, y_limits

        with figure_titles(ec.get("plot_titles", False)), y_limits(cfg.get("y_limits")):
            write_comparison(cfg, datasets, Path(cfg.get("output_dir", "outputs")) / "exp01_comparison")
    return summaries
