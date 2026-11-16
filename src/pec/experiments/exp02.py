"""Experiment 2: context-based retrieval on "realistic" contexts that mix judged
non-relevant documents with partially relevant ones.

A user's context is rarely all partially relevant: some of what they looked at is off
target. Every context here has a fixed size m (5) and a share α of non-relevant
documents, α in {0, 0.2, 0.4, 0.6, 0.8} (exp02.alphas): α = 0 is a context of partially
relevant documents only, α = 0.8 has 4 non-relevant and 1 partially relevant one.
Non-relevant documents are only the query's own judged grade-0 documents (pooled from
real systems, so on-topic looking but not relevant), never documents sampled from the
collection. Partially relevant means the dataset's context grade (TREC DL and
TREC-COVID: grade 1).

Only queries with enough documents of both kinds for every α are used, so every α is
scored on the same queries; contexts are nested across α (pec.data.seeds.noise_draws),
and the α = 0 context is Experiment 1's m = 5 context. Context documents of both kinds
are removed from every ranking. Runs (exp02.runs): the E5 context centroid, and
Experiment 1's BM25 methods bm25_DaaQ and bm25_centroid. Each is compared with its
own retriever's Oracle (the "Oracle" of the figures): the E5 true query (true_query) or
the true query text with BM25 (true_query@bm25), run on the same contexts.

Outputs per dataset under outputs/exp02_<dataset>/: contexts.csv, metrics_by_alpha.csv,
metrics_perquery.csv, summary_by_alpha.csv (% columns: of each run's own Oracle),
noise_sweep_<view>.png and summary.json; with several datasets, outputs/exp02_comparison/
holds the centroid's nDCG@10 share of the Oracle against α on each dataset's target view.
Figures have no titles unless exp02.plot_titles is set. Stored results are reused unless
exp02.rerun is set (pec.evaluate.cached_sweep).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from pec.analysis.plots import MUTED, SERIES, apply_style, oracle_lines_shown, save, set_ylim, y_limits
from pec.data.qrels import view_tag
from pec.experiments.exp01 import DatasetContext, _json_default, load_context

log = logging.getLogger(__name__)
BY = "alpha"
RUNS = ("centroid", "bm25_DaaQ", "bm25_centroid")  # the runs exp02.runs can choose from
REFERENCE_LABEL = "Oracle"  # how the true-query reference is named in the figures


def alpha_xlabel(size: int) -> str:
    return f"α (share of non-relevant documents), m = {size}"


def pct_ylabel(metric: str, reference_label: str = REFERENCE_LABEL) -> str:
    return f"{metric} (% of {reference_label})"


def noise_pipelines(store, num_results: int, query_vecs: dict | None, reference: str = "true_query",
                    runs=("centroid",), texts=None, bm25_retriever=None, bm25_centroid=None,
                    topics: pd.DataFrame | None = None, rrf_k: int = 60, terms: int | None = 100) -> dict:
    """The runs (centroid, bm25_DaaQ, bm25_centroid) and their Oracles: the E5 true
    query, and with a BM25 run the true query text searched with BM25 (true_query@bm25).
    The BM25 runs need the context passages' `texts`, `bm25_retriever` (a
    ResidualTextRetriever) and `bm25_centroid` ((analyse, weighted retriever), see
    pec.experiments.exp01.bm25_centroid_parts)."""
    import pyterrier as pt

    from pec.pipeline import CentroidEncoder, FaissRetriever, FixedQueryVectors, SparseCentroidRetriever, seed_bm25_rrf

    unknown = set(runs) - set(RUNS)
    if unknown:
        raise ValueError(f"unknown exp02 runs {sorted(unknown)}; choose from {RUNS}")
    pipes = {}
    if "centroid" in runs:
        pipes["centroid"] = CentroidEncoder(store) >> FaissRetriever(store, num_results)
    if "bm25_DaaQ" in runs:
        if texts is None or bm25_retriever is None:
            raise ValueError("bm25_DaaQ needs the context passages' texts and a BM25 retriever")
        pipes["bm25_DaaQ"] = seed_bm25_rrf(texts, bm25_retriever, num_results, rrf_k)
    if "bm25_centroid" in runs:
        if texts is None or bm25_centroid is None:
            raise ValueError("bm25_centroid needs the context passages' texts and the BM25 index's analyser and "
                             "weighted retriever")
        analyse, weighted = bm25_centroid
        pipes["bm25_centroid"] = SparseCentroidRetriever(texts, analyse, weighted, num_results, terms)
    if query_vecs:
        pipes[reference] = FixedQueryVectors(query_vecs) >> FaissRetriever(store, num_results)
    if bm25_retriever is not None and topics is not None and {"bm25_DaaQ", "bm25_centroid"} & set(runs):
        add_query = pt.apply.generic(lambda df: df.merge(topics[["qid", "query"]], on="qid"))
        pipes[f"{reference}@bm25"] = add_query >> bm25_retriever
    return pipes


def load_texts(cfg: dict, dataset: str, docnos) -> dict[str, str]:
    """docno -> text (the dataset's text_fields) for the context passages of every α,
    non-relevant ones included."""
    from pec.data.datasets import text_loader

    docnos = sorted({str(d) for d in docnos})
    frame = text_loader(cfg, dataset)(pd.DataFrame({"qid": "0", "docno": docnos}))
    return dict(zip(frame["docno"].astype(str), frame["text"]))


def _indexed_qrels(ctx: DatasetContext) -> pd.DataFrame:
    """The qrels without judged documents missing from the index, so a context always has
    exactly the α it is labelled with."""
    if not ctx.missing_docnos:
        return ctx.qrels
    rows = ctx.store.rows(ctx.qrels["docno"].tolist())
    log.warning("%s: %d judged documents are not in the index and are never drawn into a context",
                ctx.dataset, int((rows < 0).sum()))
    return ctx.qrels[rows >= 0]


def summarise_by_alpha(metrics: pd.DataFrame, metric_cols, alphas, reference: str = "true_query") -> pd.DataFrame:
    """Mean and standard deviation over repeats per (run, view, α), in `alphas` order."""
    cols = list(metric_cols) + [f"{m}_pct_{reference}" for m in metric_cols if f"{m}_pct_{reference}" in metrics]
    grouped = metrics.groupby(["name", "view", BY])
    out = grouped[cols].agg(["mean", "std"])
    out.columns = [f"{c}_{stat}" for c, stat in out.columns]
    out["n_queries"] = grouped["n_queries"].mean()
    out["repeats"] = grouped["repeat"].nunique()
    out = out.reset_index()
    order = {a: i for i, a in enumerate(alphas)}
    out = out[out[BY].isin(order)]
    return out.sort_values(["name", "view", BY], key=lambda c: c.map(order) if c.name == BY else c).reset_index(drop=True)


def plot_noise_sweep(summary: pd.DataFrame, metric_cols, alphas, out_prefix: Path, size: int,
                     reference: str = "true_query", reference_label: str = REFERENCE_LABEL,
                     titles: bool = False, runs=RUNS, pinned: dict | None = None) -> None:
    """One figure per view, one panel per metric, against α; mean over repeats with a +-1
    std band. Top row: the runs' scores and the Oracles' (E5 dashed, BM25 dotted; a flat
    line at the pinned value where `pinned` gives one). Bottom row: each run as a share of
    its own retriever's Oracle (the dashed 100% line). Titles only with `titles`."""
    import matplotlib.pyplot as plt

    from pec.analysis.oracle import display_name, oracle_label, oracle_line_label, oracle_of, pinned_value

    apply_style()
    pos = {a: i for i, a in enumerate(alphas)}
    for view, sv in summary.groupby("view"):
        names = [n for n in runs if n in set(sv["name"])]
        oracles = [o for o in dict.fromkeys(oracle_of(n, reference) for n in names)
                   if o in set(sv["name"]) or any(pinned_value(pinned, m, o, view) is not None for m in metric_cols)]
        pct_label = oracle_label({oracle_of(n, reference) for n in names}, reference, reference_label)
        fig, axes = plt.subplots(2, len(metric_cols), figsize=(3.6 * len(metric_cols), 6.4), squeeze=False)
        for j, m in enumerate(metric_cols):
            for row, col, ylabel in ((0, m, m), (1, f"{m}_pct_{reference}", pct_ylabel(m, pct_label))):
                ax = axes[row, j]
                if f"{col}_mean" not in sv:
                    ax.axis("off")
                    continue
                for i, name in enumerate(names):
                    s = sv[sv["name"] == name].sort_values(BY, key=lambda c: c.map(pos))
                    x = s[BY].map(pos).to_numpy(dtype=float)
                    y, std = s[f"{col}_mean"].to_numpy(), s[f"{col}_std"].fillna(0).to_numpy()
                    ax.fill_between(x, y - std, y + std, color=SERIES[i], alpha=0.15, linewidth=0)
                    ax.plot(x, y, color=SERIES[i], marker="o", label=display_name(name))
                if row == 0:
                    for o, dash in zip(oracles if oracle_lines_shown() else [], ("--", ":", "-.")):
                        line_label = oracle_line_label(o, len(oracles) > 1, reference, reference_label)
                        pin = pinned_value(pinned, m, o, view)
                        if pin is not None:  # the pinned Oracle: one flat line
                            ax.axhline(pin, color=MUTED, linewidth=1.2, linestyle=dash, label=line_label)
                            continue
                        r = sv[sv["name"] == o].sort_values(BY, key=lambda c: c.map(pos))
                        ax.plot(r[BY].map(pos).to_numpy(dtype=float), r[f"{m}_mean"].to_numpy(), color=MUTED,
                                linewidth=1.2, linestyle=dash, label=line_label)
                elif oracle_lines_shown():
                    ax.axhline(100, color=MUTED, linewidth=1, linestyle="--", label=pct_label)
                ax.set_xticks(range(len(alphas)), alphas)
                ax.set_xlabel(alpha_xlabel(size))
                ax.set_ylabel(ylabel)
                set_ylim(ax, "raw" if row == 0 else "pct")
                if titles:
                    ax.set_title(m)
                ax.legend(loc="best", fontsize=6.5)
        if titles:
            fig.suptitle(f"Non-relevant documents in the context ({view})")
        save(fig, Path(f"{out_prefix}_{view_tag(view)}.png"))


def run_noise(ctx: DatasetContext, c: dict, out: Path, random_seed: int = 0, cfg: dict | None = None) -> dict:
    """The runs in exp02.runs (centroid, bm25_DaaQ, bm25_centroid) and their Oracles
    (the E5 true query; with a BM25 run, the true query text with BM25) on every α (share
    of non-relevant documents). The BM25 runs need `cfg` (the passage texts and
    text_retrievers.<index>.bm25). Every run's % values are against its own retriever's
    Oracle on the same α and draw. Returns the dataset's summary."""
    from pec.analysis.oracle import pct_of_oracle, pinned_values, warn_unused_pins
    from pec.data.seeds import alpha_label, noise_draws
    from pec.evaluate import cached_sweep, metric_names

    rel = ctx.relevance
    out.mkdir(parents=True, exist_ok=True)
    reference = c.get("reference", "true_query")
    metric_cols = metric_names(c["recall_ks"])
    size, alphas = int(c["context_size"]), [alpha_label(a) for a in c["alphas"]]
    contexts = noise_draws(_indexed_qrels(ctx), size, alphas, c["repeats"], random_seed,
                           require_grade=rel.target, context_grade=rel.context)
    n_queries = contexts["qid"].nunique()
    log.info("%s: %d of %d queries have enough judged non-relevant and partially relevant documents for every α",
             ctx.dataset, n_queries, len(ctx.rows_by_grade))
    if contexts.empty:
        raise RuntimeError(f"{ctx.dataset}: no query has enough judged grade-0 and grade-{rel.context} documents "
                           f"for {size}-document contexts with alphas {alphas}")
    join = lambda docs: " ".join(docs)
    contexts.assign(n_nonrelevant=contexts["noise_docnos"].map(len), n_partial=contexts["context_docnos"].map(len),
                    **{col: contexts[col].map(join) for col in ("seed_docnos", "noise_docnos", "context_docnos")}
                    ).to_csv(out / "contexts.csv", index=False)

    settings = {"dataset": ctx.dataset, "random_seed": random_seed, "num_results": c["num_results"],
                "recall_ks": [int(x) for x in c["recall_ks"]], "context_size": size, "alphas": alphas,
                "views": {v: [int(g) for g in grades] for v, grades in rel.views.items()},
                "context_grade": rel.context, "target_grade": rel.target, "noise_grade": 0}
    runs = list(c.get("runs") or ["centroid"])
    texts = bm25_retriever = bm25_centroid = None
    if {"bm25_DaaQ", "bm25_centroid"} & set(runs):
        from pec.experiments.exp01 import bm25_centroid_parts
        from pec.text_retrievers import build_retrievers

        if cfg is None:
            raise ValueError("the BM25 runs need the config (passage texts, text_retrievers.<index>.bm25)")
        texts = load_texts(cfg, ctx.dataset, {d for s in contexts["seed_docnos"] for d in s})
        k = c["num_results"]
        bm25_retriever = build_retrievers(cfg, ctx.dataset, ["bm25"], k, k + size)["bm25"]
        if "bm25_centroid" in runs:
            bm25_centroid = bm25_centroid_parts(cfg, ctx.dataset, k + size)
    bc = c.get("bm25_centroid") or {}
    terms = None if bc.get("all_terms") else int(bc.get("terms", 100))
    pipes = noise_pipelines(ctx.store, c["num_results"], ctx.query_vecs, reference, runs, texts, bm25_retriever,
                            bm25_centroid, ctx.topics, int(c.get("rrf_k", 60)), terms)
    metrics, perquery, executed = cached_sweep(pipes, contexts, ctx.qrels, c["recall_ks"], out, settings, rel,
                                               rerun=c.get("rerun", False), by=BY)
    log.info("%s exp02: ran %s", ctx.dataset, executed or "nothing new")
    strip = lambda df: df[[col for col in df.columns if "_pct_" not in col]]
    metrics = pct_of_oracle(strip(metrics[metrics[BY].isin(alphas)]), metric_cols, BY, reference)
    perquery = pct_of_oracle(strip(perquery[perquery[BY].isin(alphas)]), metric_cols, BY, reference)
    metrics.to_csv(out / "metrics_by_alpha.csv", index=False)
    perquery.to_csv(out / "metrics_perquery.csv", index=False)
    summary = summarise_by_alpha(metrics, metric_cols, alphas, reference)
    summary.to_csv(out / "summary_by_alpha.csv", index=False)
    # the figures divide by the pinned Oracle scores where given; the tables keep the computed Oracle
    pins = pinned_values(cfg, ctx.dataset) if c.get("pin_oracle", True) else {}
    plotted = summary
    if pins:
        warn_unused_pins(pins, metrics, ctx.dataset, reference)
        pinned_draws = pct_of_oracle(strip(metrics), metric_cols, BY, reference, pinned=pins)
        plotted = summarise_by_alpha(pinned_draws, metric_cols, alphas, reference)
    with y_limits((cfg or {}).get("y_limits")):  # the shared fixed y ranges (configs/plot_results.yaml)
        plot_noise_sweep(plotted, metric_cols, alphas, out / "noise_sweep", size, reference,
                         c.get("reference_label", REFERENCE_LABEL), bool(c.get("plot_titles", False)), runs,
                         pins or None)

    result = {"dataset": ctx.dataset, "queries": len(ctx.rows_by_grade), "queries_used": int(n_queries),
              "context_size": size, "alphas": alphas, "context_grade": rel.context, "target_grade": rel.target,
              "noise_grade": 0, "runs": runs}
    pct = f"nDCG@10_pct_{reference}_mean"
    for name in runs:
        for view, sv in summary[summary["name"] == name].groupby("view"):
            result[f"{name}_ndcg10_{view}"] = {r[BY]: {"mean": r["nDCG@10_mean"], "pct_oracle": r.get(pct, np.nan)}
                                               for _, r in sv.iterrows()}
    with open(out / "summary.json", "w") as f:
        json.dump(result, f, indent=2, default=_json_default)
    return result


def dataset_config(cfg: dict, dataset: str) -> dict:
    """The config with exp02.overrides.<dataset> merged into exp02."""
    from pec.config import deep_merge

    ec = cfg["exp02"]
    merged = deep_merge(cfg, {"exp02": (ec.get("overrides") or {}).get(dataset, {})})
    merged["exp02"]["dataset"] = dataset
    return merged


def write_comparison(cfg: dict, datasets: list[str], out: Path) -> pd.DataFrame | None:
    """outputs/exp02_comparison: the centroid's nDCG@10 share of the Oracle (true query)
    against α, one line per dataset (on its target view); with exp02.pin_oracle the
    pinned E5 Oracle score (configs/oracle_values.yaml) where one is given."""
    import matplotlib.pyplot as plt

    from pec.analysis.oracle import pinned_value, pinned_values
    from pec.data.datasets import relevance
    from pec.data.seeds import alpha_label

    ec = cfg["exp02"]
    alphas = [alpha_label(a) for a in ec["alphas"]]
    label = ec.get("reference_label", REFERENCE_LABEL)
    pos = {a: i for i, a in enumerate(alphas)}
    curves = {}
    for ds in datasets:
        path = Path(cfg.get("output_dir", "outputs")) / ec["output_subdir"].format(dataset=ds) / "summary_by_alpha.csv"
        if not path.exists():
            continue
        s = pd.read_csv(path, dtype={BY: str})
        target = relevance(cfg, ds).target_view
        s = s[(s["name"] == "centroid") & (s["view"] == target) & s[BY].isin(pos)].assign(dataset=ds)
        pin = pinned_value(pinned_values(cfg, ds), "nDCG@10", "true_query", target) \
            if ec.get("pin_oracle", True) else None
        if pin is not None:  # against the pinned Oracle score
            s = s.assign(**{"nDCG@10_pct_true_query_mean": 100.0 * s["nDCG@10_mean"] / pin,
                            "nDCG@10_pct_true_query_std": 100.0 * s["nDCG@10_std"] / pin})
        curves[ds] = s.assign(oracle_ndcg10_pinned=pin)
    curves = {ds: s for ds, s in curves.items() if len(s)}
    if not curves:
        return None
    out.mkdir(parents=True, exist_ok=True)
    table = pd.concat(curves.values(), ignore_index=True)
    table.insert(0, "dataset", table.pop("dataset"))
    table.to_csv(out / "centroid_ndcg10_by_dataset.csv", index=False)

    apply_style()
    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    for i, (ds, s) in enumerate(list(curves.items())[: len(SERIES)]):
        s = s.sort_values(BY, key=lambda col: col.map(pos))
        x = s[BY].map(pos).to_numpy(dtype=float)
        y, std = s["nDCG@10_pct_true_query_mean"].to_numpy(), s["nDCG@10_pct_true_query_std"].fillna(0).to_numpy()
        ax.fill_between(x, y - std, y + std, color=SERIES[i], alpha=0.15, linewidth=0)
        ax.plot(x, y, color=SERIES[i], marker="o", label=f"{ds} ({relevance(cfg, ds).target_view})")
    if oracle_lines_shown():
        ax.axhline(100, color=MUTED, linewidth=1, linestyle="--", label=label)
    ax.set_xticks(range(len(alphas)), alphas)
    ax.set_xlabel(alpha_xlabel(int(ec["context_size"])))
    ax.set_ylabel(pct_ylabel("nDCG@10", label))
    with y_limits(cfg.get("y_limits")):
        set_ylim(ax, "pct")
    ax.legend(loc="best", fontsize=7)
    if ec.get("plot_titles", False):
        ax.set_title("Centroid nDCG@10 with non-relevant context documents")
    save(fig, out / "centroid_ndcg10_by_dataset.png")
    return table


def run_exp02(cfg: dict, datasets=None) -> dict:
    """Experiment 2 for every dataset in exp02.datasets (or `datasets`)."""
    from pec.config import output_dir

    ec = cfg["exp02"]
    datasets = list(datasets or ec.get("datasets") or ["dl19"])
    summaries = {}
    for ds in datasets:
        dcfg = dataset_config(cfg, ds)
        out = output_dir(dcfg, dcfg["exp02"]["output_subdir"].format(dataset=ds))
        summaries[ds] = run_noise(load_context(dcfg, ds, section="exp02"), dcfg["exp02"], out,
                                  dcfg["exp02"].get("random_seed", 0), dcfg)
    if len(datasets) > 1:
        write_comparison(cfg, datasets, Path(cfg.get("output_dir", "outputs")) / "exp02_comparison")
    return summaries
