"""Figures from stored results only: nothing is retrieved or evaluated.

Reads the per-draw tables the experiments write and plots them with the spread over the
repeated runs (random context draws) of each condition:

| Experiment | Table read                                         | x axis            |
|------------|----------------------------------------------------|-------------------|
| exp01      | exp01_<dataset>/baseline/metrics_by_context_size.csv | context size k  |
| exp02      | exp02_<dataset>/metrics_by_alpha.csv               | α (non-rel share) |

Plots (scripts/plot_results.py --plots):
- sweep:    per experiment, dataset, view and model group; one panel per metric, a row
            per value (raw metric, % of the true query); a line per model with a band
            for the spread over repeats (and optionally every repeat as a thin line)
- datasets: per experiment and model, one line per dataset on its target view

Unless other models are asked for, only each experiment's DEFAULT_MODELS (or the runs of
configs/plot_results.yaml) are plotted, in "main" figures, next to the true query.

The spread is computed from the per-repeat scores (each repeat's score is the mean over
that draw's queries): std (default), sem, ci95 (Student t), minmax or none. Every figure
is written next to a CSV of the numbers it shows.
"""
from __future__ import annotations

import fnmatch
import math
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pec.analysis.plots import (MARKER_SIZE, MUTED, SERIES, SERIES8, apply_style, oracle_lines_shown, save, series_style,
                                set_ylim, y_limits)
from pec.analysis.oracle import display_name, oracle_label, oracle_of, pinned_value, warn_unused_pins
from pec.data.qrels import view_tag
from pec.data.seeds import ALL, ordered_sizes

log = logging.getLogger(__name__)

PLOTS = ("sweep", "datasets")
VALUES = ("raw", "pct")
VARIANCE = ("std", "sem", "ci95", "minmax", "none")
META = {"name", "view", "context_size", "alpha", "repeat", "n_queries", "qid"}
# Experiments 1 and 2's figures have no titles and call the true-query
# reference "Oracle", unless Options.titles / Options.reference_label say otherwise.
UNTITLED = frozenset({"exp01", "exp02"})
REFERENCE_LABELS = {"exp01": "Oracle", "exp02": "Oracle"}
# Experiment 1 compares every context size with the Oracle at m = 5 unless
# Options.fixed_oracle is False.
FIXED_ORACLE = frozenset({"exp01"})
ORACLE_DASHES = ("--", ":", "-.", (0, (5, 1, 1, 1)))
# The models plotted unless configs/plot_results.yaml (runs) or the command line
# (--models, --groups, --all-models) chooses others; their Oracles are always drawn.
DEFAULT_MODELS = {
    "exp01": ("centroid", "seed_rrf", "zero_shot", "static_icl", "sliding_window", "doc2query"),  # every E5 method
    "exp02": ("centroid", "bm25_DaaQ", "bm25_centroid"),
}
MAX_LINES = len(SERIES8)  # lines per figure (more than four: a marker per line)


@dataclass
class Source:
    """One stored per-draw table."""
    experiment: str
    dataset: str
    kind: str        # "sweep"
    path: Path
    x: str           # condition column: context_size or alpha
    xlabel: str


def find_sources(root: Path, experiments=None, datasets=None) -> list[Source]:
    """Every stored table under `root` for the chosen experiments and datasets."""
    layout = {
        "exp01": [("sweep", "exp01_*", "baseline/metrics_by_context_size.csv", "context_size", "context size m")],
        "exp02": [("sweep", "exp02_*", "metrics_by_alpha.csv", "alpha", None)],
    }
    out = []
    for exp in experiments or layout:
        if exp not in layout:
            raise ValueError(f"unknown experiment {exp!r}; choose from {sorted(layout)}")
        for kind, pattern, rel_path, x, xlabel in layout[exp]:
            for d in sorted(root.glob(pattern)):
                ds = d.name.split("_", 1)[1]
                if (datasets and ds not in datasets) or not (d / rel_path).exists():
                    continue
                out.append(Source(exp, ds, kind, d / rel_path, x, xlabel or _alpha_xlabel(d)))
    return out


def _alpha_xlabel(folder: Path) -> str:
    """Experiment 2's x axis, with its context size m (from the stored settings, else 10)."""
    settings = folder / "sweep_settings.json"
    m = json.loads(settings.read_text()).get("context_size", 10) if settings.exists() else 10
    return f"α"


def metric_columns(df: pd.DataFrame) -> list[str]:
    """The metric columns of a stored table (nDCG@10, R@k, ...), in file order."""
    return [c for c in df.columns if c not in META and "_pct_" not in c and "_ref" not in c
            and pd.api.types.is_numeric_dtype(df[c])]


def load_draws(src: Source, reference: str = "true_query", fixed_at=None, pinned: dict | None = None) -> pd.DataFrame:
    """The per-draw table without retired runs, with <metric>_pct_<reference> recomputed
    against each run's own retriever's Oracle (pec.analysis.oracle) on the same view,
    condition and repeat, or with `fixed_at` against the Oracle's mean at that condition;
    `pinned` Oracle scores ({metric: {oracle: {view: value}}}) replace the computed ones."""
    from pec.analysis.oracle import pct_of_oracle
    from pec.evaluate import drop_retired

    cond = src.x
    df = pd.read_csv(src.path, dtype={"qid": str, cond: str})
    df = drop_retired(df[[c for c in df.columns if "_pct_" not in c]])
    return pct_of_oracle(df, metric_columns(df), cond, reference, fixed_at, pinned)


def condition_order(src: Source, values, keep=None, include_all: bool = False) -> list[str]:
    """Order of the x values: increasing k ("all" last, dropped unless include_all),
    or increasing α. `keep` restricts them."""
    values = {str(v) for v in values if pd.notna(v)}
    if src.x == "context_size":
        order = [s for s in ordered_sizes(values) if include_all or s != ALL]
    elif src.x == "alpha":
        order = sorted(values, key=float)
    else:
        order = sorted(values, key=int)
    if keep:
        if src.x == "alpha":  # 0.2 and 0.20 name the same α
            wanted = {round(float(k), 9) for k in keep}
            order = [o for o in order if round(float(o), 9) in wanted]
        else:
            order = [o for o in order if o in {str(k) for k in keep}]
    return order


def select_names(names, patterns=None, exclude=None) -> list[str]:
    """Names matching any of `patterns` (shell-style, e.g. "seed_mmr@*"), in pattern order,
    minus those matching `exclude`; every name when `patterns` is empty."""
    names = list(dict.fromkeys(names))
    if patterns:
        picked = []
        for p in patterns:
            picked += [n for n in names if fnmatch.fnmatchcase(n, p) and n not in picked]
    else:
        picked = names
    return [n for n in picked if not any(fnmatch.fnmatchcase(n, p) for p in exclude or [])]


def model_groups(names, reference: str, patterns=None, groups=None, max_lines: int = MAX_LINES) -> dict[str, list]:
    """Figures of at most `max_lines` models. With `patterns`, the matching models in
    order ("selected", "selected_2", ...); otherwise Experiment 1's figure groups (main,
    bm25, mmr, inv_mmr, qf_<retriever>) and every other model in "other" figures.
    `groups` keeps only the named groups."""
    from pec.evaluate import figure_groups

    names = [n for n in names if n != reference and not n.startswith(f"{reference}@")]
    if patterns:
        out = _chunks("selected", names, max_lines)
    else:
        out = {g: m[:max_lines] for g, m in figure_groups(names, reference).items() if m}
        grouped = {n for m in out.values() for n in m}
        out.update(_chunks("other", [n for n in names if n not in grouped], max_lines))
    if groups:
        out = {g: m for g, m in out.items() if g in groups or re.sub(r"_\d+$", "", g) in groups}
    return out


def _chunks(prefix: str, names: list, size: int) -> dict[str, list]:
    parts = [names[i:i + size] for i in range(0, len(names), size)]
    return {(prefix if i == 0 else f"{prefix}_{i + 1}"): p for i, p in enumerate(parts)}


def spread(values: np.ndarray, how: str) -> tuple[float, float, float]:
    """(mean, low, high) of the per-repeat scores."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return np.nan, np.nan, np.nan
    mean = float(v.mean())
    if how == "none" or len(v) == 1:
        return mean, mean, mean
    if how == "minmax":
        return mean, float(v.min()), float(v.max())
    std = float(v.std(ddof=1))
    if how == "std":
        half = std
    else:
        half = std / np.sqrt(len(v))
        if how == "ci95":
            from scipy.stats import t

            half *= float(t.ppf(0.975, len(v) - 1))
    return mean, mean - half, mean + half


def aggregate(draws: pd.DataFrame, by: list[str], cols: list[str], how: str) -> pd.DataFrame:
    """Per group: mean, the band's low / high, std and the number of repeats of each column."""
    rows = []
    for key, g in draws.groupby(by, sort=False):
        row = dict(zip(by, key if isinstance(key, tuple) else (key,)))
        row["repeats"] = g["repeat"].nunique()
        row["n_queries"] = g["n_queries"].mean() if "n_queries" in g else np.nan
        for c in cols:
            if c not in g:
                continue
            mean, lo, hi = spread(g[c].to_numpy(), how)
            row.update({f"{c}_mean": mean, f"{c}_low": lo, f"{c}_high": hi,
                        f"{c}_std": float(g[c].std(ddof=1)) if len(g) > 1 else 0.0})
        rows.append(row)
    return pd.DataFrame(rows)


def _value_col(metric: str, value: str, reference: str) -> str:
    return metric if value == "raw" else f"{metric}_pct_{reference}"


def _ylabel(metric: str, value: str, label: str, titles: bool = True) -> str:
    """Raw: the metric. %: "% of <label>", with the metric when there is no title to name it."""
    if value == "raw":
        return metric
    return f"% of {label}" if titles else f"{metric} (% of {label})"


def _draw(ax, draws_one, col, order, how, colour, label, show_repeats, xcol, marker="o"):
    """One line: mean over repeats, the spread band and optionally every repeat."""
    pos = {o: i for i, o in enumerate(order)}
    d = draws_one[draws_one[xcol].isin(pos)]
    if d.empty or col not in d:
        return None
    stats = aggregate(d, [xcol], [col], how)
    stats = stats.assign(_x=stats[xcol].map(pos)).sort_values("_x")
    xs = stats["_x"].to_numpy(dtype=float)
    if how != "none":
        ax.fill_between(xs, stats[f"{col}_low"], stats[f"{col}_high"], color=colour, alpha=0.18, linewidth=0)
    if show_repeats:
        for _, r in d.groupby("repeat"):
            r = r.assign(_x=r[xcol].map(pos)).sort_values("_x")
            ax.plot(r["_x"], r[col], color=colour, linewidth=0.7, alpha=0.45)
    ax.plot(xs, stats[f"{col}_mean"], color=colour, marker=marker, markersize=MARKER_SIZE, label=display_name(label))
    return stats


def _finish(ax, order, xlabel, ylabel, title, show_title: bool = True, kind: str | None = None):
    """Ticks, labels, title, legend, and the fixed y range of `kind` ("raw" / "pct")."""
    ax.set_xticks(range(len(order)), order)
    ax.set_xlabel(xlabel, fontsize=16)
    # ax.set_ylabel(ylabel, fontsize=16)
    set_ylim(ax, kind)
    if show_title:
        ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8, ncols=2, bbox_to_anchor=(0, 0.9))


@dataclass
class Options:
    metrics: list | None = None
    values: tuple = VALUES
    views: list | None = None
    variance: str = "std"
    show_repeats: bool = False
    reference: str = "true_query"
    show_reference: bool | None = None   # Oracle reference lines; None: ORACLE_LINES (analysis/plots.py)
    conditions: list | None = None       # context sizes / α to keep
    include_all: bool = False
    fmt: str = "png"
    titles: bool | None = None           # None: titles except for exp01 / exp02 (UNTITLED)
    reference_label: str | None = None   # None: "Oracle" for exp01 / exp02, else the reference's name
    y_limits: dict | None = None         # {"raw": [lo, hi], "pct": [lo, hi]}: the same y range in every figure
    fixed_oracle: bool | None = None     # None: Oracle fixed at oracle_size for exp01 (FIXED_ORACLE)
    oracle_size: str = "5"
    pinned: dict | None = None           # oracle_values: metric -> dataset -> Oracle run -> view -> score

    def __post_init__(self):
        if self.show_reference is None:
            self.show_reference = oracle_lines_shown()

    def pins(self, dataset: str) -> dict | None:
        """The pinned Oracle scores of one dataset ({metric: {oracle: {view: value}}}), or None."""
        from pec.analysis.oracle import pinned_values

        return (pinned_values({"oracle_values": self.pinned}, dataset) or None) if self.pinned else None

    def show_titles(self, experiment: str) -> bool:
        return self.titles if self.titles is not None else experiment not in UNTITLED

    def label(self, experiment: str) -> str:
        return self.reference_label or REFERENCE_LABELS.get(experiment, self.reference)

    def oracle_at(self, src: "Source"):
        """The context size every size is compared with, or None (each with its own)."""
        fixed = self.fixed_oracle if self.fixed_oracle is not None else src.experiment in FIXED_ORACLE
        return str(self.oracle_size) if fixed and src.x == "context_size" and src.kind == "sweep" else None


def _views(df: pd.DataFrame, wanted) -> list[str]:
    views = list(dict.fromkeys(df["view"]))
    if wanted:
        views = [v for v in views if v in wanted or view_tag(v) in wanted]
    return views


def _metrics(df: pd.DataFrame, wanted) -> list[str]:
    cols = metric_columns(df)
    return [m for m in wanted if m in cols] if wanted else cols


def _write(fig, rows: list[pd.DataFrame], path: Path, fmt: str) -> Path:
    path = path.with_suffix(f".{fmt}")
    save(fig, path)
    if rows:
        pd.concat(rows, ignore_index=True).to_csv(path.with_suffix(".csv"), index=False)
    return path


def plot_sweep(src: Source, draws: pd.DataFrame, groups: dict[str, list], opt: Options, out: Path) -> list[Path]:
    """Per view and model group: a panel per metric, a row per value (raw, % of reference)."""
    import matplotlib.pyplot as plt

    apply_style()
    order = condition_order(src, draws[src.x], opt.conditions, opt.include_all)
    metrics, written = _metrics(draws, opt.metrics), []
    if not order or not metrics:
        return written
    for view in _views(draws, opt.views):
        dv = draws[draws["view"] == view]
        titles, base = opt.show_titles(src.experiment), opt.label(src.experiment)
        fixed = opt.oracle_at(src)
        for group, names in groups.items():
            names = [n for n in names if n in set(dv["name"])][:MAX_LINES]
            if not names:
                continue
            pins = opt.pins(src.dataset)
            oracles = [o for o in dict.fromkeys(oracle_of(n, opt.reference) for n in names)
                       if o in set(dv["name"]) or any(pinned_value(pins, m, o, view) is not None for m in metrics)]
            label = oracle_label({oracle_of(n, opt.reference) for n in names}, opt.reference, base)
            fig, axes = plt.subplots(len(opt.values), len(metrics), squeeze=False,
                                     figsize=(3.6 * len(metrics), 3.1 * len(opt.values) + 0.4))
            rows = []
            for i, value in enumerate(opt.values):
                for j, m in enumerate(metrics):
                    ax, col = axes[i, j], _value_col(m, value, opt.reference)
                    for k, name in enumerate(names):
                        colour, marker = series_style(k, len(names))
                        s = _draw(ax, dv[dv["name"] == name], col, order, opt.variance, colour, name,
                                  opt.show_repeats, src.x, marker)
                        if s is not None:
                            rows.append(s.drop(columns="_x").assign(name=name, view=view, metric=m, value=value))
                    if opt.show_reference and oracles:
                        if value == "pct":
                            ax.axhline(100, color=MUTED, linewidth=1, linestyle="--", label=label)
                        else:
                            rows += _oracle_lines(ax, dv, oracles, col, order, src.x, fixed, opt, base, view, m, value,
                                                  pins)
                    _finish(ax, order, src.xlabel, _ylabel(m, value, label, titles), m, titles, value)
            if titles:
                note = "; pinned Oracle" if pins else (f"; Oracle at m = {fixed}" if fixed else "")
                fig.suptitle(f"{src.experiment} {src.dataset}: {group} ({view}{note}); band: {_band(opt)}")
            written.append(_write(fig, rows, out / f"sweep_{group}_{view_tag(view)}", opt.fmt))
    return written


def _oracle_lines(ax, dv, oracles, col, order, xcol, fixed, opt: Options, base: str, view, metric, value,
                  pins: dict | None = None):
    """The raw-score Oracle line of each retriever (E5 dashed, then dotted, ...): a flat
    line at its pinned score where `pins` gives one, else its mean at each x, or with
    `fixed` one flat line at its mean at that context size."""
    from pec.analysis.oracle import oracle_line_label

    rows = []
    for o, dash in zip(oracles, ORACLE_DASHES):
        r = dv[dv["name"] == o]
        label = oracle_line_label(o, len(oracles) > 1, opt.reference, base)
        pin = pinned_value(pins, metric, o, view)
        if pin is not None:
            ax.axhline(pin, color=MUTED, linewidth=1.2, linestyle=dash, label=label)
            rows.append(pd.DataFrame([{f"{col}_mean": pin, "name": o, "view": view, "metric": metric,
                                       "value": value, "pinned_oracle": True}]))
            continue
        if fixed is not None:
            at = r[r[xcol].astype(str) == str(fixed)]
            if at.empty or col not in at:
                continue
            y = float(at[col].mean())
            ax.axhline(y, color=MUTED, linewidth=1.2, linestyle=dash, label=label)
            rows.append(pd.DataFrame([{xcol: str(fixed), f"{col}_mean": y, "repeats": at["repeat"].nunique(),
                                       "name": o, "view": view, "metric": metric, "value": value,
                                       "fixed_oracle": True}]))
            continue
        s = aggregate(r[r[xcol].isin(order)], [xcol], [col], opt.variance)
        if s.empty:
            continue
        s = s.assign(_x=s[xcol].map({x: n for n, x in enumerate(order)})).sort_values("_x")
        ax.plot(s["_x"], s[f"{col}_mean"], color=MUTED, linewidth=1.2, linestyle=dash, label=label)
        rows.append(s.drop(columns="_x").assign(name=o, view=view, metric=metric, value=value))
    return rows


def _band(opt: Options) -> str:
    text = {"std": "±1 std over repeats", "sem": "±1 s.e.m. over repeats", "ci95": "95% CI over repeats",
            "minmax": "min-max over repeats", "none": "none"}[opt.variance]
    return text + (", thin lines = repeats" if opt.show_repeats else "")


# experiments whose datasets plot also writes every panel as its own figure
SPLIT_PANELS = frozenset({"exp02"})


def plot_datasets(experiment: str, sources: list[Source], draws: dict, models: list[str], target_views: dict,
                  opt: Options, out: Path, split: bool | None = None) -> list[Path]:
    """Per model: one line per dataset (at most four) on that dataset's target view.

    With `split` (default: for the experiments in SPLIT_PANELS, i.e. exp02), every panel
    (one metric, raw or % of the reference) is also written as its own figure,
    datasets_<model>_<metric>_<value>, next to a CSV of its numbers."""
    import matplotlib.pyplot as plt

    apply_style()
    sources = [s for s in sources if s.kind == "sweep"][: len(SERIES)]
    if len(sources) < 2:
        return []
    split = experiment in SPLIT_PANELS if split is None else split
    metrics = _metrics(pd.concat([draws[s.dataset] for s in sources]), opt.metrics)
    order = []
    for src in sources:
        order += [o for o in condition_order(src, draws[src.dataset][src.x], opt.conditions, opt.include_all)
                  if o not in order]
    if sources[0].x == "context_size":
        order = ordered_sizes(order)
    elif sources[0].x == "alpha":
        order = sorted(order, key=float)
    titles, base = opt.show_titles(experiment), opt.label(experiment)

    def panel(ax, name: str, m: str, value: str) -> list[pd.DataFrame]:
        label = oracle_label({oracle_of(name, opt.reference)}, opt.reference, base)
        col, rows = _value_col(m, value, opt.reference), []
        for k, src in enumerate(sources):
            d = draws[src.dataset]
            view = target_views.get(src.dataset) or d["view"].iloc[0]
            d = d[(d["name"] == name) & (d["view"] == view)]
            s = _draw(ax, d, col, order, opt.variance, SERIES[k], f"{src.dataset} ({view})", opt.show_repeats, src.x)
            if s is not None:
                rows.append(s.drop(columns="_x").assign(dataset=src.dataset, name=name, view=view, metric=m,
                                                        value=value))
        if opt.show_reference and value == "pct":
            ax.axhline(100, color=MUTED, linewidth=1, linestyle="--", label=label)
        _finish(ax, order, sources[0].xlabel, _ylabel(m, value, label, titles), m, titles, value)
        return rows

    written = []
    for name in models:
        fig, axes = plt.subplots(len(opt.values), len(metrics), squeeze=False,
                                 figsize=(3.6 * len(metrics), 3.1 * len(opt.values) + 0.4))
        rows = []
        for i, value in enumerate(opt.values):
            for j, m in enumerate(metrics):
                rows += panel(axes[i, j], name, m, value)
        if not rows:
            plt.close(fig)
            continue
        if titles:
            fig.suptitle(f"{experiment}: {display_name(name)} on each dataset's target documents; band: {_band(opt)}")
        safe = re.sub(r"[^0-9A-Za-z_.@-]+", "_", name)
        written.append(_write(fig, rows, out / f"datasets_{safe}", opt.fmt))
        if not split:
            continue
        for value in opt.values:
            for m in metrics:
                fig, ax = plt.subplots(figsize=(4.6, 3.6))
                panel_rows = panel(ax, name, m, value)
                if not panel_rows:
                    plt.close(fig)
                    continue
                if titles:
                    label = oracle_label({oracle_of(name, opt.reference)}, opt.reference, base)
                    kind = "raw" if value == "raw" else f"% of {label}"
                    ax.set_title(f"{experiment}: {display_name(name)}, {m} ({kind})\nband: {_band(opt)}", fontsize=9)
                tag = re.sub(r"[^0-9a-z]+", "", m.lower())
                written.append(_write(fig, panel_rows, out / f"datasets_{safe}_{tag}_{value}", opt.fmt))
    return written


PAPER_ROW = "paper_row"  # --plots paper_row: three single-column PDFs and a legend for a LaTeX figure


@dataclass
class PaperRow:
    """Per experiment, three plots side by side in one column of a two-column paper (one
    LaTeX figure per experiment), as separate PDFs (one per dataset, for LaTeX subfigures
    with their own captions) and a legend PDF to place centred below them. The first
    dataset's plot carries the y ticks and label; the others only the x axis. Every plot
    area has the same size in inches, and the PDFs are saved at their exact size, so with
    the widths `latex_widths` gives (which add up to one column) the plots line up and the
    fonts keep their point size. With raw scores every plot also draws the `oracles` (E5:
    true_query, BM25: true_query@bm25; pinned or computed as in the other figures)."""
    experiments: tuple = ("exp01", "exp02")            # one figure (three PDFs and a legend) each
    datasets: tuple = ("dl19", "dl20", "trec_covid")   # left to right; the first gets the y axis
    views: dict | None = None                          # dataset -> view (default: its target view)
    metric: str = "nDCG@10"
    value: str = "raw"                                 # raw scores, or pct of the Oracle
    oracles: tuple = ("true_query", "true_query@bm25")  # Oracle lines drawn with raw scores (E5, BM25)
    column_width: float = 3.33                         # inches: the paper's \columnwidth (ACM 3.33, IEEE 3.5)
    height: float = 1.3                                # inches, each plot
    font_size: float = 7.0
    legend_columns: int | None = None                  # default: one row up to four entries, else two rows
    y_margin: float = 0.52                             # inches left of the first plot for its y ticks and label
    gap: float = 0.06                                  # inches between plots
    bottom: float = 0.36                               # inches below the plots for the x ticks and label
    top: float = 0.04

    @classmethod
    def from_config(cls, spec: dict | None) -> "PaperRow":
        spec = dict(spec or {})
        if "experiment" in spec:  # one experiment, as a single name
            spec["experiments"] = [spec.pop("experiment")]
        unknown = set(spec) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"paper_row: unknown settings {sorted(unknown)}")
        for key in ("datasets", "experiments", "oracles"):
            if key in spec:
                spec[key] = tuple([spec[key]] if isinstance(spec[key], str) else spec[key])
        row = cls(**spec)
        unknown = set(row.experiments) - {"exp01", "exp02"}
        if unknown or not row.experiments:
            raise ValueError(f"paper_row.experiments must name exp01 or exp02, not {list(row.experiments)}")
        if len(row.datasets) != 3:
            raise ValueError(f"paper_row.datasets must name three datasets, not {list(row.datasets)}")
        if row.value not in VALUES:
            raise ValueError(f"paper_row.value must be one of {VALUES}, not {row.value!r}")
        return row

    def axes_width(self) -> float:
        width = (self.column_width - self.y_margin - 3 * self.gap) / 3
        if width <= 0:
            raise ValueError(f"paper_row: column_width {self.column_width} leaves no room for three plots")
        return width

    def figure_widths(self) -> list[float]:
        """Width in inches of each PDF: the plot area, the y margin on the first, and half a
        gap on each side between plots (a full one after the last), adding up to one column."""
        w, half = self.axes_width(), self.gap / 2
        return [self.y_margin + w + half, half + w + half, half + w + self.gap]

    def left_margin(self, i: int) -> float:
        return self.y_margin if i == 0 else self.gap / 2

    def latex_widths(self) -> list[float]:
        """Each subfigure's width as a fraction of \\columnwidth (they add up to 1)."""
        return [w / self.column_width for w in self.figure_widths()]


def plot_paper_row(sources: list[Source], draws: dict, names: list[str], row: PaperRow, opt: Options,
                   out: Path, target_views: dict | None = None, experiment: str | None = None) -> list[Path]:
    """One experiment's PaperRow figure of `names` with the Oracles:
    <out>/<experiment>_<dataset>.<fmt> for each of the three datasets and
    <out>/<experiment>_legend.<fmt>; the three plots share one y range (the fixed y_limits,
    else the union of their data). Returns the paths."""
    experiment = experiment or row.experiments[0]
    import matplotlib.pyplot as plt

    names = [n for n in names if n != opt.reference][:MAX_LINES]
    by_dataset = {s.dataset: s for s in sources if s.kind == "sweep"}
    missing = [ds for ds in row.datasets if ds not in by_dataset]
    if missing:
        log.warning("paper_row: no stored %s results for %s; nothing drawn", experiment, missing)
        return []
    out.mkdir(parents=True, exist_ok=True)
    base = opt.label(experiment)
    label = oracle_label({oracle_of(n, opt.reference) for n in names}, opt.reference, base)
    col = _value_col(row.metric, row.value, opt.reference)
    fs = row.font_size
    style = {"font.size": fs, "axes.labelsize": fs, "xtick.labelsize": fs - 1, "ytick.labelsize": fs - 1,
             "legend.fontsize": fs - 1, "lines.linewidth": 1.2, "lines.markersize": MARKER_SIZE, "pdf.fonttype": 42,
             "ps.fonttype": 42, "xtick.major.size": 2, "ytick.major.size": 2, "xtick.major.pad": 2,
             "ytick.major.pad": 2, "axes.labelpad": 2}
    written, figs, handles = [], [], {}
    apply_style()
    with plt.rc_context(style):
        axw, axh = row.axes_width(), row.height - row.bottom - row.top
        for i, (ds, fw) in enumerate(zip(row.datasets, row.figure_widths())):
            src = by_dataset[ds]
            d = draws[ds]
            view = (row.views or {}).get(ds) or (target_views or {}).get(ds) or d["view"].iloc[0]
            d = d[d["view"] == view]
            if d.empty:
                log.warning("paper_row: %s has no results for view %s", ds, view)
            order = condition_order(src, d[src.x], opt.conditions, opt.include_all)
            left = row.left_margin(i)
            fig = plt.figure(figsize=(fw, row.height))
            ax = fig.add_axes([left / fw, row.bottom / row.height, axw / fw, axh / row.height])
            for k, name in enumerate(names):
                colour, marker = series_style(k, len(names))
                _draw(ax, d[d["name"] == name], col, order, opt.variance, colour, name, opt.show_repeats, src.x,
                      marker)
            if opt.show_reference:
                if row.value == "pct":
                    ax.axhline(100, color=MUTED, linewidth=1, linestyle="--", label=label)
                else:
                    pins = opt.pins(ds)  # every listed Oracle (E5 and BM25), stored or pinned
                    oracles = [o for o in row.oracles
                               if o in set(d["name"]) or pinned_value(pins, row.metric, o, view) is not None]
                    _oracle_lines(ax, d, oracles, col, order, src.x, opt.oracle_at(src), opt, base, view,
                                  row.metric, row.value, pins)
            ax.set_xticks(range(len(order)), order)
            ax.set_xlabel(src.xlabel)
            if i == 0:
                # on two lines, so it fits the height of a one-column plot
                ax.set_ylabel(row.metric if row.value == "raw" else f"{row.metric}\n(% of {label})", labelpad=2)
            else:  # the x axis only
                ax.tick_params(axis="y", left=False, labelleft=False)
            for h in ax.get_lines():
                if not h.get_label().startswith("_"):
                    handles.setdefault(h.get_label(), h)
            figs.append((ds, fig, ax))
        # one y range for all three: the fixed y_limits, else the union of their ranges
        for _, _, ax in figs:
            set_ylim(ax, row.value)
        lows, highs = zip(*(ax.get_ylim() for _, _, ax in figs))
        for _, _, ax in figs:
            ax.set_ylim(min(lows), max(highs))
        for ds, fig, _ in figs:
            path = out / f"{experiment}_{ds}.{opt.fmt}"
            fig.savefig(path, dpi=300)  # at its exact size: no tight bounding box
            plt.close(fig)
            written.append(path)
        ncol = row.legend_columns or (len(handles) if len(handles) <= 4 else -(-len(handles) // 2))
        rows = -(-len(handles) // ncol) if handles else 1
        fig = plt.figure(figsize=(row.column_width, 0.08 + 0.15 * rows))
        fig.legend(list(handles.values()), list(handles), loc="center", ncol=ncol, frameon=False,
                   handlelength=1.6, columnspacing=1.0, borderaxespad=0)
        path = out / f"{experiment}_legend.{opt.fmt}"
        fig.savefig(path, dpi=300)
        plt.close(fig)
        written.append(path)
    # rounded down, so that the three never add up to more than one column in LaTeX
    fractions = [math.floor(w * 1000) / 1000 for w in row.latex_widths()]
    log.info("paper_row: LaTeX subfigure widths (left to right): %s",
             ", ".join(f"{w:.3f}\\columnwidth" for w in fractions))
    (out / f"{experiment}_widths.txt").write_text(
        "\n".join(f"{ds}: {w:.3f}\\columnwidth" for ds, w in zip(row.datasets, fractions)) + "\n")
    return written


def plot_results(root: Path, out: Path, experiments=None, datasets=None, plots=PLOTS, models=None,
                 exclude_models=None, groups=None, opt: Options | None = None, cfg: dict | None = None,
                 all_models: bool = False, runs: dict | None = None,
                 paper_row: "PaperRow | None" = None) -> list[Path]:
    """Every chosen figure from the stored results under `root`, written to
    out/<experiment>_<dataset>/ (and out/<experiment>_comparison/ for the datasets plot).

    Models: `models` (patterns) when given; every model in its figure group with `groups`
    or `all_models`; otherwise each experiment's `runs` (experiment -> run names or
    patterns, configs/plot_results.yaml), or DEFAULT_MODELS for an experiment it does not
    list, in "main" figures of at most MAX_LINES runs per view.
    Returns the figure paths."""
    opt = opt or Options()
    with y_limits(opt.y_limits):
        return _plot_results(root, out, experiments, datasets, plots, models, exclude_models, groups, opt, cfg,
                             all_models, runs, paper_row)


def _plot_results(root, out, experiments, datasets, plots, models, exclude_models, groups, opt, cfg, all_models,
                  runs, paper_row=None) -> list[Path]:
    default = models is None and not groups and not all_models
    runs = {str(k): list(v) for k, v in (runs or {}).items() if v}
    defaults = lambda exp: runs.get(exp) or list(DEFAULT_MODELS[exp])
    sources = find_sources(root, experiments, datasets)
    if not sources:
        log.warning("no stored results under %s for experiments %s, datasets %s", root, experiments, datasets)
    written = []
    by_exp: dict[str, list[Source]] = {}
    loaded: dict[tuple, pd.DataFrame] = {}
    for src in sources:
        pins = opt.pins(src.dataset)
        draws = load_draws(src, opt.reference, opt.oracle_at(src), pins)
        if pins:
            warn_unused_pins(pins, draws, src.dataset, opt.reference)
        loaded[(src.experiment, src.kind, src.dataset)] = draws
        by_exp.setdefault(src.experiment, []).append(src)
        dest = out / f"{src.experiment}_{src.dataset}"
        if src.kind == "sweep" and "sweep" in plots:
            wanted = defaults(src.experiment) if default else models
            names = select_names([n for n in draws["name"].unique() if n != opt.reference], wanted, exclude_models)
            figures = _chunks("main", names, MAX_LINES) if default else model_groups(names, opt.reference, models,
                                                                                     groups)
            written += plot_sweep(src, draws, figures, opt, dest)
    if "datasets" in plots:
        for exp, srcs in by_exp.items():
            sweep = [s for s in srcs if s.kind == "sweep"]
            draws = {s.dataset: loaded[(exp, "sweep", s.dataset)] for s in sweep}
            if len(draws) < 2:
                continue
            names = list(dict.fromkeys(n for d in draws.values() for n in d["name"] if n != opt.reference))
            names = select_names(names, models or defaults(exp), exclude_models)
            written += plot_datasets(exp, sweep, draws, names, _target_views(cfg, draws), opt,
                                     out / f"{exp}_comparison")
    if PAPER_ROW in plots:
        row = paper_row or PaperRow()
        for exp in row.experiments:  # one LaTeX figure per experiment
            sweep = [s for s in by_exp.get(exp, []) if s.kind == "sweep"]
            draws = {s.dataset: loaded[(exp, "sweep", s.dataset)] for s in sweep}
            if not draws:
                log.warning("paper_row: no stored %s results under %s (include it in --experiments)", exp, root)
                continue
            names = list(dict.fromkeys(n for d in draws.values() for n in d["name"] if n != opt.reference))
            names = select_names(names, models or defaults(exp), exclude_models)
            written += plot_paper_row(sweep, draws, names, row, opt, out / "paper", _target_views(cfg, draws), exp)
    for p in written:
        log.info("wrote %s", p)
    return written


def _target_views(cfg: dict | None, draws: dict) -> dict[str, str]:
    """Each dataset's target view (from its relevance scheme in the config, else the view
    holding only the highest grade)."""
    from pec.data.datasets import relevance

    out = {}
    for ds, d in draws.items():
        if cfg and ds in (cfg.get("datasets") or {}):
            out[ds] = relevance(cfg, ds).target_view
        else:
            eq = [v for v in d["view"].unique() if "=" in v and ">" not in v]
            out[ds] = sorted(eq)[-1] if eq else d["view"].iloc[0]
    return out


def describe(root: Path, experiments=None, datasets=None, reference: str = "true_query") -> pd.DataFrame:
    """What can be plotted: per stored table, its models, views, metrics, conditions and repeats."""
    rows = []
    for src in find_sources(root, experiments, datasets):
        d = load_draws(src, reference)
        cond = d[src.x] if src.x in d else d["context_size"]
        rows.append({"experiment": src.experiment, "dataset": src.dataset, "plot": src.kind,
                     "models": " ".join(sorted(d["name"].unique())), "views": " ".join(dict.fromkeys(d["view"])),
                     "metrics": " ".join(metric_columns(d)),
                     "conditions": " ".join(condition_order(src, cond, include_all=True)),
                     "repeats": int(d["repeat"].nunique()), "path": str(src.path)})
    return pd.DataFrame(rows)
