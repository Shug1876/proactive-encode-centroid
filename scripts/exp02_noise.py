"""Experiment 2: the context centroid, bm25_DaaQ and bm25_centroid on contexts of
m = 5 documents in which a share α of the documents are judged non-relevant (grade-0) and
the rest partially relevant, α in {0, 0.2, 0.4, 0.6, 0.8}, with the E5 and BM25 Oracles.

    python scripts/exp02_noise.py --config configs/exp02_noise.yaml
    python scripts/exp02_noise.py --datasets dl19 trec_covid
    python scripts/exp02_noise.py --alphas 0 0.4 0.8 --titles
    python scripts/exp02_noise.py --rerun
"""
import argparse
import json
import logging

from pec.config import load_config
from pec.experiments.exp02 import run_exp02


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/exp02_noise.yaml")
    parser.add_argument("--datasets", nargs="+", help="subset of datasets to run (default: exp02.datasets)")
    parser.add_argument("--alphas", nargs="+", type=float, metavar="ALPHA",
                        help="shares of non-relevant documents in the context (default: exp02.alphas)")
    parser.add_argument("--titles", action="store_true", help="give the figures titles (default: none)")
    parser.add_argument("--rerun", action="store_true", help="recompute stored results")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config)
    if args.alphas:
        cfg["exp02"]["alphas"] = args.alphas
    if args.titles:
        cfg["exp02"]["plot_titles"] = True
    if args.rerun:
        cfg["exp02"]["rerun"] = True
    print(json.dumps(run_exp02(cfg, args.datasets), indent=2, default=str))


if __name__ == "__main__":
    main()
