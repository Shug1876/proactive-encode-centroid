"""Experiment 1 per dataset (TREC-DL19, TREC-DL20, TREC-COVID): every zero-training
baseline and the true query (Oracle) over the context-size sweep.

    python scripts/exp01_dl19.py --config configs/exp01_dl19.yaml
    python scripts/exp01_dl19.py --datasets dl19 trec_covid
    python scripts/exp01_dl19.py --qf-groups llm --qf-only
    python scripts/exp01_dl19.py --rerun seed_inv_mmr@0.5
    python scripts/exp01_dl19.py --bm25-ablation --no-fixed-oracle
"""
import argparse
import json
import logging

from pec.config import load_config
from pec.experiments.exp01 import run_exp01


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/exp01_dl19.yaml")
    parser.add_argument("--datasets", nargs="+", help="subset of datasets to run (default: exp01.datasets)")
    parser.add_argument("--qf-groups", nargs="+", choices=["llm", "text"],
                        help="query formulation groups to run: llm (zero_shot, static_icl), text (sliding_window, "
                             "doc2query); default: baseline.query_formulation.groups")
    parser.add_argument("--qf-only", action="store_true",
                        help="run only the query formulation methods (and the true query), not the other baselines")
    parser.add_argument("--titles", action="store_true", help="give the figures titles (default: none)")
    parser.add_argument("--bm25-ablation", action="store_true",
                        help="also run bm25_DaaQ and bm25_centroid without BM25's term weighting (k1 = 0), "
                             "as bm25_DaaQ_nobm25 and bm25_centroid_nobm25")
    parser.add_argument("--no-fixed-oracle", action="store_true",
                        help="figures: compare each context size m with the Oracle at that m, not at m = 5")
    parser.add_argument("--rerun", nargs="*", metavar="NAME",
                        help="recompute stored baseline results: every run, or only the runs named")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config)
    baseline = cfg["exp01"]["baseline"]
    if args.qf_groups:
        baseline.setdefault("query_formulation", {})["groups"] = args.qf_groups
    if args.qf_only:
        baseline["other_baselines"] = False
    if args.titles:
        cfg["exp01"]["plot_titles"] = True
    if args.bm25_ablation:
        baseline["bm25_binary_ablation"] = True
    if args.no_fixed_oracle:
        baseline["fixed_oracle"] = False
    if args.rerun is not None:
        baseline["rerun"] = args.rerun or True
    print(json.dumps(run_exp01(cfg, datasets=args.datasets), indent=2, default=str))


if __name__ == "__main__":
    main()
