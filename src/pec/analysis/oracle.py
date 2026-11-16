"""The Oracle (the true query) that each run is compared with in the figures.

Every run is compared with the Oracle of its own retriever: E5 runs (the centroid,
seed_rrf, the MMRs, the query formulation methods' E5 runs) with the E5 true
query `true_query`; runs on a text retriever with the true query text on that retriever,
`true_query@<retriever>` (bm25_DaaQ, bm25_centroid and their ablations: BM25;
`<method>@<retriever>`: that retriever).

`pct_of_oracle` adds `<metric>_pct_<reference>` to a per-draw table as 100 * run /
Oracle, either against the Oracle on the same view, condition and repeat, or, with
`fixed_at` (e.g. context size "5"), against the Oracle's mean over repeats at that one
condition, so every condition is compared with the same, flat Oracle. Pinned values
(configs/oracle_values.yaml, `pinned_values`) replace the computed Oracle wherever they
are given (metric, dataset, Oracle run, view).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

BM25_RUNS = ("bm25_DaaQ", "bm25_centroid")
ABLATION_SUFFIX = "_nobm25"  # BM25 without term-frequency / length weighting (k1 = 0)
RETRIEVER_NAMES = {"bm25": "BM25", "monot5": "monoT5", "colbert": "ColBERT", "splade": "SPLADE"}
ORACLE_LABEL = "Oracle"
# How runs are named in figure legends (and titles) where that differs from the run name in
# the tables and configs; "<run>@<retriever>" follows its run. Only the figures change: the
# stored tables, configs and command-line options keep the run names on the left.
DISPLAY_NAMES = {
    "centroid": "E5_centroid",
    "static_icl": "QF-Fixed",
    "zero_shot": "QF-Zero",
    "sliding_window": "QF-SW",
    "doc2query": "QF-D2Q",
}


def display_name(name: str) -> str:
    """A run's name in the figures (DISPLAY_NAMES): e.g. static_icl -> QF-Fixed,
    doc2query@bm25 -> QF-D2Q@bm25; every other label unchanged."""
    name = str(name)
    base, sep, rest = name.partition("@")
    return DISPLAY_NAMES.get(base, base) + sep + rest


def oracle_of(name: str, reference: str = "true_query") -> str:
    """The Oracle run `name` is compared with."""
    from pec.query_formulation import METHODS

    if name == reference or name.startswith(f"{reference}@"):
        return name
    base = name[: -len(ABLATION_SUFFIX)] if name.endswith(ABLATION_SUFFIX) else name
    if base in BM25_RUNS:
        return f"{reference}@bm25"
    if "@" in name:
        method, retriever = name.split("@", 1)
        if method in METHODS:
            return f"{reference}@{retriever}"
    return reference


def retriever_of_oracle(oracle: str, reference: str = "true_query") -> str:
    """"E5" for the E5 Oracle, else the retriever's display name ("BM25", ...)."""
    if oracle == reference or "@" not in oracle:
        return "E5"
    r = oracle.split("@", 1)[1]
    return RETRIEVER_NAMES.get(r, r)


def oracle_label(oracles, reference: str = "true_query", label: str = ORACLE_LABEL) -> str:
    """Legend / axis name of the Oracle line(s) for runs compared with `oracles`:
    "Oracle" for the E5 Oracle, "Oracle (BM25)" for another retriever's, "Oracle (own
    retriever)" when the runs use several."""
    oracles = set(oracles)
    if len(oracles) == 1:
        (only,) = oracles
        return label if only == reference else f"{label} ({retriever_of_oracle(only, reference)})"
    return f"{label} (own retriever)"


def oracle_line_label(oracle: str, several: bool, reference: str = "true_query",
                      label: str = ORACLE_LABEL) -> str:
    """Label of one raw-score Oracle line: "Oracle", or "Oracle (E5)" / "Oracle (BM25)"
    when several Oracles are drawn."""
    return f"{label} ({retriever_of_oracle(oracle, reference)})" if several else oracle_label([oracle], reference, label)


def pinned_values(cfg: dict | None, dataset: str) -> dict:
    """{metric: {oracle: {view: value}}} for one dataset, from cfg["oracle_values"]
    (metric -> dataset -> Oracle run -> view -> score)."""
    pins = (cfg or {}).get("oracle_values") or {}
    out = {}
    for metric, by_dataset in pins.items():
        by_oracle = (by_dataset or {}).get(dataset)
        if by_oracle:
            out[str(metric)] = {str(o): {str(v): float(x) for v, x in (views or {}).items()}
                                for o, views in by_oracle.items()}
    return out


def pinned_value(pinned: dict | None, metric: str, oracle: str, view: str) -> float | None:
    """The pinned score of one Oracle, metric and view, or None."""
    value = ((pinned or {}).get(metric) or {}).get(oracle, {}).get(view)
    return None if value is None else float(value)


def warn_unused_pins(pinned: dict | None, draws: pd.DataFrame, dataset: str, reference: str = "true_query") -> None:
    """Log pinned values whose metric, Oracle run or view never occur in `draws` (a typo
    there silently falls back to the computed Oracle)."""
    if not pinned or draws.empty:
        return
    views, names = set(draws["view"]), set(draws["name"]) | {oracle_of(n, reference) for n in draws["name"]}
    for metric, by_oracle in pinned.items():
        for oracle, by_view in by_oracle.items():
            for view in by_view:
                if metric not in draws.columns or oracle not in names or view not in views:
                    log.warning("%s: pinned Oracle value %s / %s / %s matches no stored result", dataset, metric,
                                oracle, view)


def oracle_values(draws: pd.DataFrame, metric_cols, by: str, fixed_at=None) -> pd.DataFrame:
    """Oracle scores per draw ([name, view, by, repeat (, qid)] + metrics), or with
    `fixed_at` their mean over repeats at that condition ([name, view (, qid)] + metrics)."""
    cols = [m for m in metric_cols if m in draws]
    q = ["qid"] if "qid" in draws.columns else []
    if fixed_at is None:
        return draws[["name", "view", by, "repeat"] + q + cols]
    at = draws[draws[by].astype(str) == str(fixed_at)]
    return at.groupby(["name", "view"] + q, as_index=False)[cols].mean()


def pct_of_oracle(draws: pd.DataFrame, metric_cols, by: str = "context_size", reference: str = "true_query",
                  fixed_at=None, pinned: dict | None = None) -> pd.DataFrame:
    """`draws` with <metric>_pct_<reference> = 100 * metric / the run's own Oracle (see
    oracle_of), on the same view, condition and repeat, or with `fixed_at` the Oracle's
    mean at that condition; `pinned` ({metric: {oracle: {view: value}}}, pinned_values)
    replaces the Oracle's score wherever it gives one. Runs whose Oracle has neither a
    score nor a pinned value get NaN (logged once)."""
    cols = [m for m in metric_cols if m in draws]
    out = draws.copy()
    if out.empty:
        return out
    out["_oracle"] = out["name"].map(lambda n: oracle_of(n, reference))
    names = set(out["name"])
    oracles = set(out["_oracle"])
    pinned_oracles = {o for by_oracle in (pinned or {}).values() for o in by_oracle}
    missing = sorted(o for o in oracles if o not in names and o not in pinned_oracles)
    if missing:
        log.warning("no scores for the Oracle(s) %s: the runs compared with them get no %% values", missing)
    keys = (["view"] if fixed_at is not None else ["view", by, "repeat"]) + (["qid"] if "qid" in out.columns else [])
    if fixed_at is not None and not (out[by].astype(str) == str(fixed_at)).any():
        log.warning("no results at %s = %s for the fixed Oracle; comparing each %s with its own Oracle instead",
                    by, fixed_at, by)
        return pct_of_oracle(draws, metric_cols, by, reference, None, pinned)
    ref = oracle_values(out[out["name"].isin(oracles)], cols, by, fixed_at).rename(columns={"name": "_oracle"})
    if fixed_at is None:
        ref = ref.astype({by: str})
        out[by] = out[by].astype(str)
    ref = ref.rename(columns={m: f"{m}__oracle" for m in cols})
    merged = out.merge(ref, on=["_oracle"] + keys, how="left")
    for m, by_oracle in (pinned or {}).items():
        if m not in cols:
            continue
        for oracle, by_view in by_oracle.items():
            for view, value in by_view.items():
                hit = (merged["_oracle"] == oracle) & (merged["view"] == view)
                merged.loc[hit, f"{m}__oracle"] = float(value)
    for m in cols:
        merged[f"{m}_pct_{reference}"] = 100.0 * merged[m] / merged[f"{m}__oracle"].replace(0, np.nan)
    return merged.drop(columns=["_oracle"] + [f"{m}__oracle" for m in cols])
