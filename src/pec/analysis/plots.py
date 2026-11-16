"""Shared matplotlib styling for experiment figures.

Colours are the validated default palette (light surface). Line charts use at most four
hues (adjacent-pair safe), or up to eight with a marker per line (SERIES8, MARKERS8).
Every multi-series figure has a legend, and every figure is written next to a CSV of the
numbers it shows.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # blue, orange, aqua, yellow
# The validated palette's eight slots in its fixed order (adjacent-pair CVD-safe for line
# charts). Figures with more than four lines use them with a marker per line as well, so
# identity never rests on colour alone.
SERIES8 = SERIES + ["#e87ba4", "#008300", "#6250d6", "#e34948"]  # + magenta, green, violet, red
MARKERS8 = ["o", "s", "^", "D", "v", "P", "X", "*"]


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif", "font.size": 9, "text.color": INK,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK_SECONDARY, "axes.titlesize": 10,
        "axes.titlecolor": INK, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "xtick.color": MUTED, "ytick.color": MUTED, "legend.frameon": False,
        "lines.linewidth": 2.0, "lines.markersize": 6,
    })


_SHOW_TITLES = [True]


@contextmanager
def figure_titles(enabled: bool):
    """Within the block, `save` keeps figure titles only if `enabled`; without them it
    removes every suptitle and panel title, except panels marked with keep_title (panel
    titles that are labels rather than titles)."""
    _SHOW_TITLES.append(bool(enabled))
    try:
        yield
    finally:
        _SHOW_TITLES.pop()


def titles_shown() -> bool:
    """Whether figures saved now keep their titles (see figure_titles)."""
    return _SHOW_TITLES[-1]


def keep_title(ax) -> None:
    """Mark a panel title as a label that stays when titles are off."""
    ax._pec_keep_title = True


def strip_titles(fig) -> None:
    if fig._suptitle is not None:
        fig._suptitle.set_text("")
    for ax in fig.axes:
        if not getattr(ax, "_pec_keep_title", False):
            ax.set_title("")


_Y_LIMITS: list = [None]


def parse_y_limits(spec) -> dict | None:
    """{"raw": (lo, hi), "pct": (lo, hi)} from a config's y_limits mapping (either key may
    be left out), or None."""
    if not spec:
        return None
    out = {}
    for kind, lim in dict(spec).items():
        if kind not in ("raw", "pct"):
            raise ValueError(f"y_limits: unknown kind {kind!r}; use raw (scores) or pct (% of the Oracle)")
        if lim is None:
            continue
        lo, hi = (float(v) for v in lim)
        if not lo < hi:
            raise ValueError(f"y_limits.{kind}: the lower limit must be below the upper one, not {lim}")
        out[kind] = (lo, hi)
    return out or None


@contextmanager
def y_limits(limits):
    """Within the block, set_ylim gives every result panel of a kind ("raw" scores, "pct"
    of the Oracle) the same fixed y range; `limits` is a y_limits mapping (or None: the
    automatic ranges)."""
    _Y_LIMITS.append(parse_y_limits(limits))
    try:
        yield
    finally:
        _Y_LIMITS.pop()


def current_y_limits() -> dict | None:
    return _Y_LIMITS[-1]


def set_ylim(ax, kind: str | None) -> None:
    """Fix the panel's y range to the active limits of `kind` ("raw" or "pct"), logging
    when plotted values (lines, bands, Oracle lines) fall outside and are clipped."""
    limits = _Y_LIMITS[-1]
    if not kind or not limits or kind not in limits:
        return
    lo, hi = limits[kind]
    values = [np.asarray(line.get_ydata(), dtype=float) for line in ax.get_lines()]
    for coll in ax.collections:
        for path in coll.get_paths():
            values.append(np.asarray(path.vertices[:, 1], dtype=float))
    values = np.concatenate([v[np.isfinite(v)] for v in values if np.size(v)]) if values else np.zeros(0)
    if values.size and (values.min() < lo - 1e-9 or values.max() > hi + 1e-9):
        import logging

        logging.getLogger(__name__).warning(
            "%s axis %r: values from %.3g to %.3g are clipped to the fixed y range [%g, %g] (y_limits.%s)",
            kind, ax.get_ylabel(), values.min(), values.max(), lo, hi, kind)
    ax.set_ylim(lo, hi)


def save(fig, path: str | Path, rect=None) -> None:
    """`rect` reserves figure margins (left, bottom, right, top) for a suptitle or figure
    legend. Titles are removed when figure_titles(False) is in effect."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not titles_shown():
        strip_titles(fig)
    fig.tight_layout(rect=rect)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def series_style(i: int, n: int) -> tuple[str, str]:
    """(colour, marker) of line i of n: the four-colour palette with circles up to four
    lines, else the eight-slot palette with a marker per line."""
    if n <= len(SERIES):
        return SERIES[i % len(SERIES)], "o"
    return SERIES8[i % len(SERIES8)], MARKERS8[i % len(MARKERS8)]


def lines(ax, x_by_series: dict[str, np.ndarray], y_by_series: dict[str, np.ndarray], log2_x: bool = True,
          ylabel: str = "", xlabel: str = "N", ykind: str | None = None) -> None:
    """Up to four series: legend plus a direct label at each line's end (end labels are
    stacked apart when lines finish close together). Five to eight: the eight-slot palette
    with a marker per line and the legend only."""
    ends = []
    many = len(x_by_series) > len(SERIES)
    for i, (name, x) in enumerate(x_by_series.items()):
        y = np.asarray(y_by_series[name], dtype=float)
        colour, marker = series_style(i, len(x_by_series))
        ax.plot(x, y, color=colour, marker=marker, label=name)
        finite = np.isfinite(y)
        if finite.any() and not many:
            j = np.nonzero(finite)[0][-1]
            ends.append((name, x[j], y[j]))
    if log2_x:
        ax.set_xscale("log", base=2)
    ax.margins(x=0.25)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=7)
    set_ylim(ax, ykind)  # a fixed y range (y_limits) before the end labels are placed
    lo, hi = ax.get_ylim()
    last = None
    for name, x_end, y_end in sorted(ends, key=lambda e: -e[2]):
        frac = (y_end - lo) / (hi - lo) if hi > lo else 0.5
        if last is not None and frac > last - 0.07:
            frac = last - 0.07
        last = frac
        # x in data coordinates, y in axes fraction
        ax.annotate(name, xy=(x_end, frac), xycoords=ax.get_xaxis_transform(), xytext=(6, 0),
                    textcoords="offset points", fontsize=7, color=INK_SECONDARY, va="center")
