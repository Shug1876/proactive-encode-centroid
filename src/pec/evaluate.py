"""Evaluation with PyTerrier.

Every run is evaluated twice with pt.Experiment, once per qrels view:
- "grade>=2": nDCG@10 graded over highly + perfectly relevant documents, Recall@k of both
- "grade=3":  nDCG@10 and Recall@k of perfectly relevant documents only
Context documents are removed from every ranking before evaluation.

Contexts are swept over sizes k (pec.data.seeds.context_draws). Because a query with
fewer than k grade-1 documents is left out at that k, the query set changes with k, so
runs are compared through their share of the true-query upper bound measured on the same
queries and contexts ("<metric>_pct_true_query"), and the best k is chosen on that share.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pyterrier as pt
from ir_measures import R, nDCG

from pec.data.qrels import DL_RELEVANCE, Relevance, qrels_view, view_tag
from pec.data.seeds import ALL, ordered_sizes

log = logging.getLogger(__name__)
DRAW_KEYS = ["view", "context_size", "repeat"]


def measures(recall_ks) -> list:
    return [nDCG @ 10] + [R @ k for k in recall_ks]


def metric_names(recall_ks) -> list[str]:
    return [str(m) for m in measures(recall_ks)]


def evaluate_runs(runs: dict[str, pd.DataFrame], topics: pd.DataFrame, qrels: pd.DataFrame, recall_ks,
                  perquery: bool = False, relevance: Relevance = DL_RELEVANCE) -> pd.DataFrame:
    """Long table: one row per (run, view) with nDCG@10 and R@k columns (or per query).
    The views come from the dataset's relevance scheme."""
    frames = []
    for view in relevance.view_names:
        res = pt.Experiment(list(runs.values()), topics[["qid"]], qrels_view(qrels, view, relevance), measures(recall_ks),
                            names=list(runs), perquery=perquery, validate="ignore")
        if perquery:
            res = res.pivot_table(index=["name", "qid"], columns="measure", values="value").reset_index()
            res.columns.name = None
        res.insert(1, "view", view)
        frames.append(res)
    return pd.concat(frames, ignore_index=True)


def sweep_context_sizes(pipes: dict, contexts: pd.DataFrame, qrels: pd.DataFrame, recall_ks,
                        perquery: bool = False, relevance: Relevance = DL_RELEVANCE,
                        run_sizes: dict[str, set[str] | None] | None = None,
                        skip: set[tuple[str, str, int]] | None = None, by: str = "context_size"
                        ) -> tuple[pd.DataFrame, dict, pd.DataFrame | None]:
    """Run every pipeline on every (context_size, repeat) draw and evaluate it.

    `run_sizes` limits a run to some context sizes (name -> labels; missing or None = every
    size), and `skip` lists (name, context_size, repeat) draws not to run, e.g. because
    their results are already stored (see cached_sweep). `by` is the column naming each
    condition of the sweep (default the context size; Experiment 2: the context mix).

    Returns (metrics with context_size / repeat / n_queries columns, runs keyed by
    (context_size, repeat, name), per-query metrics or None).
    """
    run_sizes, skip = run_sizes or {}, skip or set()
    frames, pq_frames, runs_out = [], [], {}
    for (size, repeat), draw in contexts.groupby([by, "repeat"], sort=False):
        todo = {name: pipe for name, pipe in pipes.items()
                if (run_sizes.get(name) is None or size in run_sizes[name]) and (name, size, int(repeat)) not in skip}
        if not todo:
            continue
        topics = draw[["qid", "seed_docnos"]].reset_index(drop=True)
        log.info("%s %s, repeat %d: %d queries, %d runs", by, size, repeat, len(topics), len(todo))
        runs = {name: pipe(topics) for name, pipe in todo.items()}
        runs_out.update({(size, repeat, name): run for name, run in runs.items()})
        tags = {by: size, "repeat": repeat, "n_queries": len(topics)}
        frames.append(evaluate_runs(runs, topics, qrels, recall_ks, relevance=relevance).assign(**tags))
        if perquery:
            pq_frames.append(evaluate_runs(runs, topics, qrels, recall_ks, perquery=True, relevance=relevance).assign(**tags))
    empty = pd.DataFrame(columns=["name", "view", by, "repeat", "n_queries"])
    metrics = pd.concat(frames, ignore_index=True) if frames else empty
    perquery_table = (pd.concat(pq_frames, ignore_index=True) if pq_frames else empty.assign(qid=[])) if perquery else None
    return metrics, runs_out, perquery_table


SWEEP_KEYS = ["name", "context_size", "repeat"]
SETTINGS_FILE = "sweep_settings.json"
# Runs no longer part of the study: their stored results are deleted from the tables the
# next time they are written, and they are never plotted.
RETIRED_RUNS = frozenset({"seed_fusion"})
# Runs stored under an earlier name: read under the new one, so stored results are reused
# and plotted (and rewritten under the new name the next time the tables are written).
RENAMED_RUNS = {"seed_bm25_rrf": "bm25_DaaQ", "seed_bm25_rrf_nobm25": "bm25_DaaQ_nobm25"}


def drop_retired(df: pd.DataFrame) -> pd.DataFrame:
    """`df` without the rows of retired runs (RETIRED_RUNS), and with renamed runs under
    their current name (RENAMED_RUNS)."""
    if "name" not in df.columns:
        return df
    renamed = df["name"].isin(RENAMED_RUNS)
    if renamed.any():
        log.info("reading %d stored rows of %s under their new names", int(renamed.sum()),
                 sorted(set(df.loc[renamed, "name"])))
        df = df.assign(name=df["name"].replace(RENAMED_RUNS))
    retired = df["name"].isin(RETIRED_RUNS)
    if retired.any():
        log.info("dropping %d stored rows of retired runs %s", int(retired.sum()),
                 sorted(set(df.loc[retired, "name"])))
    return df[~retired].reset_index(drop=True)


def _stored(path: Path, by: str = "context_size") -> pd.DataFrame | None:
    """A stored results table without its derived % columns (they are recomputed) and
    without retired runs."""
    if not path.exists():
        return None
    df = pd.read_csv(path, dtype={"qid": str, by: str})
    return drop_retired(df[[c for c in df.columns if "_pct_" not in c]])


def cached_sweep(pipes: dict, contexts: pd.DataFrame, qrels: pd.DataFrame, recall_ks, out: Path, settings: dict,
                 relevance: Relevance = DL_RELEVANCE, run_sizes: dict | None = None, rerun=False,
                 by: str = "context_size", prefix: str = "metrics"
                 ) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """The context-size sweep, reusing results already stored in `out`.

    out/metrics_by_context_size.csv and out/metrics_perquery.csv hold every run's results
    so far, from this and earlier invocations. A (run, context size, repeat) already in
    both is not run again unless `rerun` is True or lists that run's name. New results
    replace stored ones with the same key; stored results of runs not in `pipes` are kept,
    so runs made separately (e.g. the LLM and the non-LLM query formulation methods)
    accumulate. The draws of a (size, repeat) depend only on the qrels and the random
    seed, so results stay comparable across invocations as long as `settings` (seed,
    result depth, metrics, views) are unchanged; if out/sweep_settings.json differs, the
    stored results are not reused. Stored results of retired runs (RETIRED_RUNS) are
    dropped. `by` names the sweep's condition column (see sweep_context_sizes) and the
    tables are out/<prefix>_by_<by>.csv and out/<prefix>_perquery.csv. Returns (metrics,
    per-query metrics, runs executed).
    """
    out.mkdir(parents=True, exist_ok=True)
    settings_path = out / SETTINGS_FILE
    stored_m, stored_pq = _stored(out / f"{prefix}_by_{by}.csv", by), _stored(out / f"{prefix}_perquery.csv", by)
    if stored_m is not None and settings_path.exists():
        previous = json.loads(settings_path.read_text())
        if previous != settings:
            log.warning("%s: stored results used different settings (%s, now %s); they are not reused",
                        out, previous, settings)
            stored_m = stored_pq = None
    elif stored_m is not None:
        log.warning("%s: no %s next to the stored results; assuming they used the current settings",
                    out, SETTINGS_FILE)
    if stored_m is None or stored_pq is None:
        stored_m = stored_pq = None

    rerun_names = set(pipes) if rerun is True else set(rerun or [])
    skip = set()
    if stored_m is not None:
        keys = lambda df: set(zip(df["name"], df[by].astype(str), df["repeat"].astype(int)))
        skip = {k for k in keys(stored_m) & keys(stored_pq) if k[0] in pipes and k[0] not in rerun_names}
        if skip:
            log.info("%s: reusing stored results for %d (run, size, repeat) draws; pass rerun to recompute",
                     out, len(skip))
    metrics, _, perquery = sweep_context_sizes(pipes, contexts, qrels, recall_ks, perquery=True, relevance=relevance,
                                               run_sizes=run_sizes, skip=skip, by=by)
    executed = sorted(set(metrics["name"])) if len(metrics) else []
    if stored_m is not None:
        def combine(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
            if new.empty:
                return old.reset_index(drop=True)
            new_keys = set(zip(new["name"], new[by].astype(str), new["repeat"].astype(int)))
            keep = [k not in new_keys for k in zip(old["name"], old[by].astype(str), old["repeat"].astype(int))]
            return pd.concat([old[keep], new], ignore_index=True)

        metrics, perquery = combine(stored_m, metrics), combine(stored_pq, perquery)
    for df in (metrics, perquery):
        df[by] = df[by].astype(str)
        df["repeat"] = df["repeat"].astype(int)
    settings_path.write_text(json.dumps(settings, indent=2))
    return metrics, perquery, executed


def relative_to_reference(metrics: pd.DataFrame, metric_cols, reference: str = "true_query",
                          keys=DRAW_KEYS) -> pd.DataFrame:
    """Add <metric>_pct_<reference>: 100 * run / reference, on the same view and draw
    (and query, for per-query tables). Unchanged if the reference run is absent."""
    keys = [k for k in keys if k in metrics.columns] + (["qid"] if "qid" in metrics.columns else [])
    ref = metrics[metrics["name"] == reference]
    if ref.empty:
        return metrics
    ref = ref.set_index(keys)[list(metric_cols)].add_suffix("_ref")
    merged = metrics.join(ref, on=keys)
    out = metrics.copy()
    for m in metric_cols:
        out[f"{m}_pct_{reference}"] = 100.0 * merged[m] / merged[f"{m}_ref"].replace(0, np.nan)
    return out


def summarise_by_context(metrics: pd.DataFrame, metric_cols, reference: str = "true_query") -> pd.DataFrame:
    """Mean and standard deviation over repeats per (run, view, context size)."""
    cols = list(metric_cols) + [f"{m}_pct_{reference}" for m in metric_cols if f"{m}_pct_{reference}" in metrics]
    grouped = metrics.groupby(["name", "view", "context_size"])
    out = grouped[cols].agg(["mean", "std"])
    out.columns = [f"{c}_{stat}" for c, stat in out.columns]
    out["n_queries"] = grouped["n_queries"].mean()
    out["repeats"] = grouped["repeat"].nunique()
    out = out.reset_index()
    order = {s: i for i, s in enumerate(ordered_sizes(out["context_size"]))}
    return out.sort_values(["name", "view", "context_size"], key=lambda c: c.map(order) if c.name == "context_size" else c)


def best_context_size(summary: pd.DataFrame, metric_cols, reference: str = "true_query",
                      exclude=(ALL,)) -> pd.DataFrame:
    """Per run, view and metric: the context size (other than "all") with the highest mean
    share of the reference, with its scores next to those at "all" for comparison. Ties go
    to the smaller k. The reference run itself is ranked on the raw metric."""
    rows = []
    for (name, view), g in summary.groupby(["name", "view"]):
        at_all = g[g["context_size"] == ALL]
        cand = g[~g["context_size"].isin(exclude)]
        if cand.empty:
            continue
        for m in metric_cols:
            pct = f"{m}_pct_{reference}_mean"
            score = pct if pct in g and name != reference else f"{m}_mean"
            best = cand.loc[cand[score].idxmax()]
            rows.append({"name": name, "view": view, "metric": m, "selected_on": score, "best_k": best["context_size"],
                         "best_mean": best[f"{m}_mean"], "best_pct": best.get(pct, np.nan),
                         "best_n_queries": best["n_queries"],
                         "all_mean": at_all[f"{m}_mean"].iloc[0] if len(at_all) else np.nan,
                         "all_pct": at_all[pct].iloc[0] if len(at_all) and pct in at_all else np.nan})
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class SweepFigureStyle:
    """How the context-size figures name and place the Oracle (the true query).

    - oracle_size: compare every context size with the Oracle's mean at this size (a flat
      Oracle line), or None to compare each size with the Oracle at that size;
    - label: the Oracle's name in legends and axes; size_symbol: the context size's symbol;
    - pinned: Oracle scores that replace the computed ones ({metric: {oracle: {view:
      value}}}, pec.analysis.oracle.pinned_values; configs/oracle_values.yaml).
    Every run is compared with its own retriever's Oracle (pec.analysis.oracle)."""
    oracle_size: str | None = None
    label: str = "true_query"
    size_symbol: str = "k"
    pinned: dict | None = field(default=None, compare=False)

    def xlabel(self) -> str:
        return f"context size {self.size_symbol}"

    def note(self) -> str:
        if self.pinned:
            return "; pinned Oracle"
        return f"; Oracle at {self.size_symbol} = {self.oracle_size}" if self.oracle_size else ""


LEGACY_STYLE = SweepFigureStyle()


def figure_summary(metrics: pd.DataFrame, metric_cols, reference: str = "true_query",
                   style: SweepFigureStyle = LEGACY_STYLE) -> pd.DataFrame:
    """The summary the figures plot: <metric>_pct_<reference> against each run's own
    Oracle, at the same size or fixed at style.oracle_size (the saved tables keep their
    own % columns)."""
    from pec.analysis.oracle import pct_of_oracle

    raw = drop_retired(metrics[[c for c in metrics.columns if "_pct_" not in c]])
    plotted = pct_of_oracle(raw, metric_cols, "context_size", reference, style.oracle_size, style.pinned)
    return summarise_by_context(plotted, metric_cols, reference)


def plot_context_sweep(summary: pd.DataFrame, metric_cols, names, out_prefix: Path, reference: str = "true_query",
                       exclude_sizes=(ALL,), style: SweepFigureStyle = LEGACY_STYLE) -> None:
    """One figure per view: % of each run's Oracle against context size, one panel per
    metric, mean over repeats with a +-1 std band; the Oracle is the dashed 100% line.
    Up to eight runs (more than four: a marker per run). Context sizes in `exclude_sizes`
    ("all" by default) stay in the tables but are not plotted."""
    import matplotlib.pyplot as plt

    from pec.analysis.oracle import display_name, oracle_label, oracle_of
    from pec.analysis.plots import MUTED, SERIES8, apply_style, lines, oracle_lines_shown, save, series_style, titles_shown

    apply_style()
    summary = summary[~summary["context_size"].isin(exclude_sizes)]
    sizes = ordered_sizes(summary["context_size"])
    pos = {s: i for i, s in enumerate(sizes)}
    names = [n for n in names if n != reference and not n.startswith(f"{reference}@") and n not in RETIRED_RUNS
             and n in set(summary["name"])][: len(SERIES8)]
    if not names:
        return
    label = oracle_label({oracle_of(n, reference) for n in names}, reference, style.label)
    for view, sv in summary.groupby("view"):
        fig, axes = plt.subplots(1, len(metric_cols), figsize=(3.6 * len(metric_cols), 3.5), squeeze=False)
        for ax, m in zip(axes[0], metric_cols):
            col = f"{m}_pct_{reference}"
            if f"{col}_mean" not in sv:
                continue
            xs, ys = {}, {}
            for i, name in enumerate(names):
                s = sv[sv["name"] == name].sort_values("context_size", key=lambda c: c.map(pos))
                x = s["context_size"].map(pos).to_numpy(dtype=float)
                shown = display_name(name)  # the legend label (static_icl -> fixed_llm)
                xs[shown], ys[shown] = x, s[f"{col}_mean"].to_numpy()
                std = s[f"{col}_std"].fillna(0).to_numpy()
                colour, _ = series_style(i, len(names))
                ax.fill_between(x, ys[shown] - std, ys[shown] + std, color=colour, alpha=0.12, linewidth=0)
            # without titles the y axis names the metric too
            ylabel = f"% of {label}" if titles_shown() else f"{m} (% of {label})"
            lines(ax, xs, ys, log2_x=False, xlabel=style.xlabel(), ylabel=ylabel, ykind="pct")
            if oracle_lines_shown():
                ax.axhline(100, color=MUTED, linewidth=1, linestyle="--", label=label)
            ax.legend(fontsize=6 if len(names) > 4 else 7, loc="best")
            ax.set_xticks(range(len(sizes)), sizes)
            ax.set_title(m)
        fig.suptitle(f"Share of the {label} by context size {style.size_symbol} ({view}{style.note()})")
        save(fig, Path(f"{out_prefix}_{view_tag(view)}.png"))


def write_context_sweep(metrics: pd.DataFrame, metric_cols, out: Path, names, reference: str = "true_query",
                        prefix: str = "", figure_groups: dict[str, list[str]] | None = None, overview: bool = True,
                        style: SweepFigureStyle = LEGACY_STYLE) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Save the per-draw table, the summary over repeats, the best k, and the figures:
    `context_sweep_*` for `names`, plus `context_sweep_<group>_*` per extra figure group.
    The tables' % columns compare every run with `reference` at the same size; the
    figures compare each run with its own retriever's Oracle as `style` says."""
    metrics = relative_to_reference(drop_retired(metrics), metric_cols, reference)
    metrics.to_csv(out / f"{prefix}metrics_by_context_size.csv", index=False)
    summary = summarise_by_context(metrics, metric_cols, reference)
    summary.to_csv(out / f"{prefix}summary_by_context_size.csv", index=False)
    best = best_context_size(summary, metric_cols, reference)
    best.to_csv(out / f"{prefix}best_context_size.csv", index=False)
    plotted = figure_summary(metrics, metric_cols, reference, style)
    plot_context_sweep(plotted, metric_cols, names, out / f"{prefix}context_sweep", reference, style=style)
    for group, group_names in (figure_groups or {}).items():
        plot_context_sweep(plotted, metric_cols, group_names, out / f"{prefix}context_sweep_{group}", reference,
                           style=style)
    if figure_groups and overview:
        plot_overview(plotted, metric_cols, {"main": list(names), **figure_groups},
                      out / f"{prefix}context_sweep_overview", reference, style=style)
    return summary, best


E5_METHODS = ("centroid", "seed_rrf", "zero_shot", "static_icl", "sliding_window", "doc2query")
BM25_METHODS = ("bm25_DaaQ", "bm25_centroid", "bm25_DaaQ_nobm25", "bm25_centroid_nobm25")


def figure_groups(names, reference: str = "true_query") -> dict[str, list[str]]:
    """Figure groups for every run that has results, so runs made in earlier invocations
    are plotted too:
    - main:     every E5 method: centroid, seed_rrf, zero_shot, static_icl,
                sliding_window, doc2query (up to six lines)
    - bm25:     bm25_DaaQ, bm25_centroid and their binary-matching ablations (_nobm25)
    - mmr:      centroid and every seed_mmr@lambda
    - inv_mmr:  centroid and every seed_inv_mmr@lambda
    - qf_<r>:   every query formulation method's queries through text retriever r
    Groups without results are left out (main is always there)."""
    from pec.query_formulation import METHODS

    names = set(names) - RETIRED_RUNS
    have = lambda xs: [x for x in xs if x in names]
    by_lambda = lambda prefix: sorted((n for n in names if n.startswith(prefix)), key=lambda n: float(n[len(prefix):]))
    centroid = have(["centroid"])
    groups = {"main": have(list(E5_METHODS))}
    if have(list(BM25_METHODS)):
        groups["bm25"] = have(list(BM25_METHODS))
    for group, members in {"mmr": by_lambda("seed_mmr@"), "inv_mmr": by_lambda("seed_inv_mmr@")}.items():
        if members:
            groups[group] = centroid + members
    retrievers = sorted({n.split("@", 1)[1] for n in names if "@" in n and n.split("@", 1)[0] in METHODS})
    for r in retrievers:
        groups[f"qf_{r}"] = have([f"{m}@{r}" for m in METHODS])
    return groups


GROUP_TITLES = {"main": "E5 methods", "bm25": "BM25 methods", "mmr": "MMR", "inv_mmr": "inverse context MMR"}


def plot_overview(summary: pd.DataFrame, metric_cols, groups: dict[str, list[str]], out_prefix: Path,
                  reference: str = "true_query", exclude_sizes=(ALL,), style: SweepFigureStyle = LEGACY_STYLE) -> None:
    """Every method family with results, one figure per view and metric
    (<out_prefix>_<metric>_<view>.png): a panel per figure group with its runs, the
    centroid as a grey reference line and each run's Oracle at 100%. Each run keeps the
    colour it has in its own group's figure."""
    import re

    import matplotlib.pyplot as plt

    from pec.analysis.oracle import display_name, oracle_label, oracle_of
    from pec.analysis.plots import (MARKER_SIZE, MUTED, SERIES8, apply_style, oracle_lines_shown, save, series_style,
                                    set_ylim, titles_shown)

    apply_style()
    summary = summary[~summary["context_size"].isin(exclude_sizes)]
    sizes = ordered_sizes(summary["context_size"])
    pos = {s: i for i, s in enumerate(sizes)}
    present = set(summary["name"]) - RETIRED_RUNS
    panels = []
    for g, names in groups.items():
        names = [n for n in names if n in present][: len(SERIES8)]
        runs = [(i, n, len(names)) for i, n in enumerate(names) if n != "centroid"]
        if runs:
            panels.append((g, runs))
    if not panels:
        return

    def series(sv, name, col):
        s = sv[sv["name"] == name].sort_values("context_size", key=lambda c: c.map(pos))
        return s["context_size"].map(pos).to_numpy(dtype=float), s[f"{col}_mean"].to_numpy()

    ncols = min(4, len(panels))
    nrows = -(-len(panels) // ncols)
    for view, sv in summary.groupby("view"):
        for m in metric_cols:
            col = f"{m}_pct_{reference}"
            if f"{col}_mean" not in sv:
                continue
            fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.0 * nrows + 0.6), squeeze=False)
            for ax in axes.ravel()[len(panels):]:
                ax.axis("off")
            for ax, (group, runs) in zip(axes.ravel(), panels):
                if "centroid" in present:
                    x, y = series(sv, "centroid", col)
                    ax.plot(x, y, color=MUTED, linewidth=1.2, label=display_name("centroid"))
                for i, name, n in runs:
                    x, y = series(sv, name, col)
                    colour, marker = series_style(i, n)
                    ax.plot(x, y, color=colour, marker=marker, markersize=MARKER_SIZE, label=display_name(name))
                label = oracle_label({oracle_of(n, reference) for _, n, _ in runs}, reference, style.label)
                if oracle_lines_shown():
                    ax.axhline(100, color=MUTED, linewidth=1, linestyle="--", label=label)
                ax.set_xticks(range(len(sizes)), sizes)
                ax.set_xlabel(style.xlabel())
                set_ylim(ax, "pct")
                group_title = GROUP_TITLES.get(group, f"queries @ {group[3:]}" if group.startswith("qf_") else group)
                ax.set_title(group_title)
                if titles_shown():
                    ax.set_ylabel(f"% of {label}")
                    ax.legend(loc="best", fontsize=6)
                else:  # the metric on the y axis, the method family as the legend's heading
                    ax.set_ylabel(f"{m} (% of {label})")
                    ax.legend(loc="best", fontsize=6, title=group_title, title_fontsize=6.5)
            fig.suptitle(f"{m}: every baseline with results, as a share of its {style.label} ({view}{style.note()})")
            tag = re.sub(r"[^0-9a-z]+", "", m.lower())
            save(fig, Path(f"{out_prefix}_{tag}_{view_tag(view)}.png"))
