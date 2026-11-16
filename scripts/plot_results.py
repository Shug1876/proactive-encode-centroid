"""Figures from stored results only (nothing is retrieved or evaluated), with the spread
over the repeated runs of every condition.

Reads outputs/exp01_<dataset>/baseline/ and outputs/exp02_<dataset>/ and writes outputs/plots/<experiment>_<dataset>/ (and <experiment>_comparison/).

    python scripts/plot_results.py --list                                  # what can be plotted
    python scripts/plot_results.py                                         # every plot, the runs of configs/plot_results.yaml
    python scripts/plot_results.py --runs-config my_runs.yaml              # another selection of runs
    python scripts/plot_results.py --all-models                            # every model, in its figure group
    python scripts/plot_results.py --experiments exp01 --datasets dl19 --models centroid seed_rrf "seed_mmr@*"
    python scripts/plot_results.py --experiments exp02 --plots sweep datasets --variance ci95 --show-repeats
    python scripts/plot_results.py --groups main mmr --views eq3 --metrics nDCG@10 R@100 --values pct
    python scripts/plot_results.py --plots paper_row                       # exp01 and exp02: three one-column PDFs + legend each
    python scripts/plot_results.py --plots paper_row --paper-experiments exp02 --paper-metric R@100 --column-width 3.5
"""
import argparse
import logging
from pathlib import Path

import pandas as pd

from pec.analysis.result_plots import PAPER_ROW, PLOTS, VALUES, VARIANCE, Options, PaperRow, describe, plot_results


def _runs(args) -> dict | None:
    """Experiment -> runs to show, from --runs-config (plot_results.runs), unless --no-runs-config."""
    if args.no_runs_config or not args.runs_config:
        return None
    path = Path(args.runs_config)
    if not path.exists():
        if args.runs_config_given:
            raise SystemExit(f"--runs-config {path} not found (or pass --no-runs-config)")
        return None
    from pec.config import load_config

    runs = (load_config(path).get("plot_results") or {}).get("runs") or {}
    unknown = set(runs) - {"exp01", "exp02"}
    if unknown:
        raise SystemExit(f"{path}: unknown experiments {sorted(unknown)} under plot_results.runs (exp01, exp02)")
    return runs


def _paper_row(args) -> PaperRow:
    """The paper_row settings: the paper_row section of --runs-config (if any), overridden
    by the --paper-* / --column-width options."""
    spec = {}
    path = Path(args.runs_config) if args.runs_config else None
    if path is not None and path.exists() and not args.no_runs_config:
        from pec.config import load_config

        spec = dict(load_config(path).get("paper_row") or {})
    flags = {"experiments": args.paper_experiments, "datasets": args.paper_datasets, "metric": args.paper_metric,
             "value": args.paper_value, "column_width": args.column_width, "height": args.paper_height,
             "font_size": args.paper_font_size, "legend_columns": args.paper_legend_columns}
    spec.update({k: v for k, v in flags.items() if v is not None})
    if args.paper_views:
        views = dict(spec.get("views") or {})
        for item in args.paper_views:
            ds, sep, view = item.partition("=")
            if not sep or not view:
                raise SystemExit(f"--paper-views takes DATASET=VIEW (e.g. dl19=grade=3), not {item!r}")
            views[ds] = view
        spec["views"] = views
    try:
        return PaperRow.from_config(spec)
    except ValueError as e:
        raise SystemExit(str(e))


def _y_limits(args) -> dict | None:
    """The fixed y ranges: y_limits of --runs-config, overridden by --ylim-raw / --ylim-pct,
    or None with --no-fixed-ylim."""
    if args.no_fixed_ylim:
        return None
    limits = {}
    path = Path(args.runs_config) if args.runs_config else None
    if path is not None and path.exists() and not args.no_runs_config:
        from pec.config import load_config

        limits = dict(load_config(path).get("y_limits") or {})
    if args.ylim_raw:
        limits["raw"] = args.ylim_raw
    if args.ylim_pct:
        limits["pct"] = args.ylim_pct
    from pec.analysis.plots import parse_y_limits

    try:
        return parse_y_limits(limits)
    except ValueError as e:
        raise SystemExit(str(e))


def _pinned(args) -> dict | None:
    """The oracle_values mapping of --oracle-values, unless --no-pinned-oracle."""
    if args.no_pinned_oracle or not args.oracle_values or not Path(args.oracle_values).exists():
        return None
    from pec.config import load_config

    return load_config(args.oracle_values).get("oracle_values") or None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default="outputs", help="where the experiments wrote their results")
    parser.add_argument("--out", default=None, help="where to write the figures (default: <root>/plots)")
    parser.add_argument("--config", default="configs/paths.yaml",
                        help="config with the datasets' relevance schemes (target view of the datasets plot)")
    parser.add_argument("--experiments", nargs="+", choices=["exp01", "exp02"],
                        help="experiments to plot (default: every one with results)")
    parser.add_argument("--datasets", nargs="+", help="datasets to plot (default: every one with results)")
    parser.add_argument("--plots", nargs="+", choices=list(PLOTS) + [PAPER_ROW], default=list(PLOTS),
                        help="sweep: metric against context size / α; "
                             "datasets: one line per dataset for each model; paper_row: three PDFs to place side by "
                             "side in one column of a two-column paper (y axis on the first only) and a legend PDF "
                             "to centre below them, in <out>/paper/ (not drawn unless asked for)")
    paper = parser.add_argument_group("paper_row (defaults: the paper_row section of --runs-config, else built in)")
    paper.add_argument("--paper-experiments", nargs="+", choices=["exp01", "exp02"],
                       help="experiments, one LaTeX figure each (default: exp01 exp02)")
    paper.add_argument("--paper-datasets", nargs=3, metavar="DATASET",
                       help="the three datasets, left to right (default: dl19 dl20 trec_covid)")
    paper.add_argument("--paper-views", nargs="+", metavar="DATASET=VIEW",
                       help="the view of a dataset, e.g. dl19=grade>=2 (default: each dataset's target view)")
    paper.add_argument("--paper-metric", help="metric (default nDCG@10)")
    paper.add_argument("--paper-value", choices=VALUES,
                       help="raw scores, with the E5 and BM25 Oracle lines (default), or pct of the Oracle")
    paper.add_argument("--column-width", type=float, metavar="INCHES",
                       help="the paper's \\columnwidth in inches (default 3.33; \\the\\columnwidth in pt / 72.27)")
    paper.add_argument("--paper-height", type=float, metavar="INCHES", help="height of each plot (default 1.3)")
    paper.add_argument("--paper-font-size", type=float, metavar="PT", help="font size in pt (default 7)")
    paper.add_argument("--paper-legend-columns", type=int, metavar="N",
                       help="legend columns (default: one row up to four entries, else two rows)")
    parser.add_argument("--models", nargs="+", metavar="PATTERN",
                        help='models to plot in every experiment, shell patterns allowed ("seed_mmr@*"), each '
                             "with its Oracle; at most eight per figure, more are split over several figures "
                             "(default: each experiment's runs in --runs-config)")
    parser.add_argument("--all-models", action="store_true",
                        help="plot every model, in Experiment 1's figure groups (main, bm25, mmr, inv_mmr, "
                             "qf_<retriever>, other)")
    parser.add_argument("--exclude-models", nargs="+", metavar="PATTERN", help="models to leave out")
    parser.add_argument("--runs-config", default=None,
                        help="YAML file choosing the runs each experiment shows (plot_results.runs; default "
                             "configs/plot_results.yaml); --models, --groups and --all-models take precedence")
    parser.add_argument("--no-runs-config", action="store_true",
                        help="ignore the runs file and use the built-in default runs")
    parser.add_argument("--groups", nargs="+",
                        help="figure groups to plot (every model in them) when --models is not given: main (E5 "
                             "methods), bm25, mmr, inv_mmr, qf_<retriever>, other")
    parser.add_argument("--views", nargs="+", help='qrels views, by name or tag ("grade=3" or eq3; default: all)')
    parser.add_argument("--metrics", nargs="+", help="metrics, e.g. nDCG@10 R@100 (default: all stored)")
    parser.add_argument("--values", nargs="+", choices=VALUES, default=list(VALUES),
                        help="raw: the metric; pct: as a %% of the reference (one figure row each)")
    parser.add_argument("--variance", choices=VARIANCE, default="std",
                        help="band over the repeats: std (default), sem, ci95 (Student t), minmax, none")
    parser.add_argument("--show-repeats", action="store_true", help="also draw every repeat as a thin line")
    parser.add_argument("--conditions", nargs="+",
                        help="x values to keep: context sizes (1 3 5) or α (0 0.4 0.8)")
    parser.add_argument("--include-all", action="store_true", help='plot the "all" context size if stored')
    parser.add_argument("--reference", default="true_query", help="the upper-bound run (percentages and dashed line)")
    parser.add_argument("--oracle-lines", action=argparse.BooleanOptionalAction, default=None,
                        help="draw the Oracle reference lines (raw-score Oracle lines, 100%% line of the %% panels); "
                             "default: ORACLE_LINES in src/pec/analysis/plots.py (off)")
    parser.add_argument("--no-reference", action="store_true", help="same as --no-oracle-lines")
    parser.add_argument("--titles", action=argparse.BooleanOptionalAction, default=None,
                        help="figure titles on / off (default: on, except for exp01 and exp02)")
    parser.add_argument("--ylim-raw", nargs=2, type=float, metavar=("LO", "HI"),
                        help="y range of every raw-score panel (default: y_limits.raw of --runs-config, 0 1)")
    parser.add_argument("--ylim-pct", nargs=2, type=float, metavar=("LO", "HI"),
                        help="y range of every %% of Oracle panel (default: y_limits.pct of --runs-config, 0 140)")
    parser.add_argument("--no-fixed-ylim", action="store_true", help="automatic y ranges per panel")
    parser.add_argument("--fixed-oracle", action=argparse.BooleanOptionalAction, default=None,
                        help="compare every context size with the Oracle at --oracle-size (a flat Oracle line; "
                             "default: on for exp01) or each size with the Oracle at that size")
    parser.add_argument("--oracle-size", default="5", help="context size of the fixed Oracle (default 5)")
    parser.add_argument("--oracle-values", default="configs/oracle_values.yaml",
                        help="pinned Oracle scores (oracle_values: metric -> dataset -> Oracle run -> view -> score); "
                             "where given they replace the computed Oracle")
    parser.add_argument("--no-pinned-oracle", action="store_true",
                        help="use the Oracle computed from the stored results, not the pinned scores")
    parser.add_argument("--reference-label", help='name of the reference in legends and axes '
                                                  '(default: "Oracle" for exp01 / exp02, else --reference)')
    parser.add_argument("--format", default="pdf", choices=["png", "pdf", "svg"])
    parser.add_argument("--list", action="store_true", help="list what can be plotted and exit")
    args = parser.parse_args()
    args.runs_config_given = args.runs_config is not None
    args.runs_config = args.runs_config or "configs/plot_results.yaml"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    root = Path(args.root)
    if args.list:
        table = describe(root, args.experiments, args.datasets, args.reference)
        with pd.option_context("display.max_colwidth", None, "display.width", 200):
            print(table.drop(columns="path").to_string(index=False) if len(table) else f"no stored results under {root}")
        return
    cfg = None
    if args.config and Path(args.config).exists():
        from pec.config import load_config

        cfg = load_config(args.config)
    opt = Options(metrics=args.metrics, values=tuple(args.values), views=args.views, variance=args.variance,
                  show_repeats=args.show_repeats, reference=args.reference, show_reference=False if args.no_reference else args.oracle_lines,
                  conditions=args.conditions, include_all=args.include_all, fmt=args.format,
                  titles=args.titles, reference_label=args.reference_label, fixed_oracle=args.fixed_oracle,
                  oracle_size=args.oracle_size, pinned=_pinned(args), y_limits=_y_limits(args))
    written = plot_results(root, Path(args.out) if args.out else root / "plots", args.experiments, args.datasets,
                           args.plots, args.models, args.exclude_models, args.groups, opt, cfg, args.all_models,
                           _runs(args), _paper_row(args) if PAPER_ROW in args.plots else None)
    print(f"{len(written)} figures written")
    for p in written:
        print(p)


if __name__ == "__main__":
    main()
