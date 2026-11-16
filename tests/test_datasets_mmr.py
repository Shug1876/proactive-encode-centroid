"""Multi-dataset Experiment 1 (TREC-COVID-style grades, comparison outputs) and the
multi-query MMR baseline, on the synthetic corpus."""
import faiss
import numpy as np
import pandas as pd
import pytest

from pec.config import load_config
from pec.data.qrels import Relevance, normalise_qrels
from pec.data.seeds import context_draws
from pec.index.faiss_store import DocnoMap, FaissDocStore
from pec.pipeline import SeedFusionRetriever, seed_mmr

COVID = Relevance(context=1, target=2, views={"grade=2": (2,)},
                  names={0: "not relevant", 1: "partially relevant", 2: "relevant"})


def covid_qrels(toy):
    """Toy qrels on a 0-2 scale: former grade 3 -> 2 (relevant), grades 1-2 -> 1 (partial)."""
    return normalise_qrels(toy["qrels"]).assign(label=lambda q: q["label"].map({0: 0, 1: 1, 2: 1, 3: 2}))


def query_vectors(store, qrels, target):
    out = {}
    for qid, g in qrels[qrels["label"] == target].groupby("qid"):
        v = store.vectors(store.rows(g["docno"].tolist())).mean(0)
        out[qid] = v / np.linalg.norm(v)
    return out


def small_exp01_config(tmp_path):
    cfg = load_config("configs/exp01_dl19.yaml")
    ec = cfg["exp01"]
    ec["baseline"].update(num_results=50, recall_ks=[10, 50], context_sizes=[1, 3, "all"], repeats=2,
                          bm25_rrf={"enabled": False}, bm25_centroid={"enabled": False},  # needs a BM25 index: test_baselines_new.py fakes one
                          mmr={"lambdas": [0.5, 0.9], "pool": 100})
    ec["baseline"]["query_formulation"]["methods"] = []  # LLM / doc2query baselines: tests/test_query_formulation.py
    ec["overrides"]["trec_covid"]["baseline"]["context_sizes"] = [1, 3, 5]
    cfg["output_dir"] = str(tmp_path)
    return cfg


def test_seed_mmr_diversifies_and_matches_fusion_first_pick():
    # one seed; candidates: two near-duplicates very close to it, one a bit further but different
    vecs = np.array([[1, 0, 0], [0.98, 0.2, 0], [0.98, 0.2, 0.01], [0.9, -0.1, 0.42]], dtype=np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    index = faiss.IndexFlatIP(3)
    index.add(vecs)
    store = FaissDocStore(index, DocnoMap(docnos=["seed", "dupA", "dupB", "other"]))
    topics = pd.DataFrame({"qid": ["q"], "seed_docnos": [("seed",)]})
    fused = SeedFusionRetriever(store, 3)(topics)["docno"].tolist()
    assert fused[2] == "other"  # by relevance alone the far document comes last
    mmr = seed_mmr(store, 3, Lambda=0.5, pool=10)(topics)
    assert mmr["docno"].tolist()[0] == fused[0]  # the first pick is always the most relevant
    assert mmr["docno"].tolist()[1] == "other"  # then MMR skips the near-duplicate
    assert "seed" not in set(mmr["docno"]) and len(mmr) == 3
    assert mmr["rank"].tolist() == [0, 1, 2]
    with pytest.raises(ValueError):
        seed_mmr(store, 3, Lambda=1.0)


def test_seed_mmr_on_toy_contexts(toy):
    store, qrels = toy["store"], normalise_qrels(toy["qrels"])
    contexts = context_draws(qrels, [3], repeats=1)
    run = seed_mmr(store, 20, Lambda=0.7, pool=60)(contexts)
    assert run.groupby("qid").size().max() == 20 and "doc_vec" not in run.columns
    for qid, seeds in zip(contexts["qid"], contexts["seed_docnos"]):
        assert not set(seeds) & set(run[run["qid"] == qid]["docno"])


def test_context_sweep_figures_leave_out_all(toy, monkeypatch, tmp_path):
    import pec.analysis.plots as plots
    from pec.evaluate import metric_names, relative_to_reference, summarise_by_context, sweep_context_sizes, plot_context_sweep
    from pec.pipeline import CentroidEncoder, FaissRetriever, FixedQueryVectors

    store, qrels = toy["store"], normalise_qrels(toy["qrels"])
    pipes = {"centroid": CentroidEncoder(store) >> FaissRetriever(store, 30),
             "true_query": FixedQueryVectors(query_vectors(store, qrels, 3)) >> FaissRetriever(store, 30)}
    cols = metric_names([10])
    metrics, _, _ = sweep_context_sizes(pipes, context_draws(qrels, [1, 3, "all"], 2), qrels, [10])
    summary = summarise_by_context(relative_to_reference(metrics, cols), cols)
    assert "all" in set(summary["context_size"])  # still in the tables
    seen = []
    real = plots.lines
    monkeypatch.setattr(plots, "lines", lambda ax, xs, ys, **kw: (seen.append(xs), real(ax, xs, ys, **kw)))
    plot_context_sweep(summary, cols, ["centroid"], tmp_path / "sweep")
    assert seen and all(len(x) == 2 for xs in seen for x in xs.values())  # only k=1 and k=3 plotted


def test_exp01_covid_grades(toy, tmp_path):
    from pec.experiments.exp01 import build_context, dataset_config, run_parts

    store, qrels = toy["store"], covid_qrels(toy)
    ctx = build_context(store, toy["topics"].astype({"qid": str}), qrels, query_vectors(store, qrels, 2),
                        {d: f"toy document {d}" for d in toy["docnos"]}, index_key="toy_covid",
                        relevance=COVID, dataset="trec_covid")
    cfg = dataset_config(small_exp01_config(tmp_path), "trec_covid")
    summary = run_parts(ctx, cfg, tmp_path / "covid")
    assert (summary["context_grade"], summary["target_grade"]) == (1, 2)
    assert set(summary["baseline_best_k_centroid_ndcg10"]) == {"grade=2"}
    assert summary["baseline_best_k_centroid_ndcg10"]["grade=2"]["best_k"] in {"1", "3", "5"}
    sweep = pd.read_csv(tmp_path / "covid/baseline/summary_by_context_size.csv", dtype={"context_size": str})
    assert set(sweep["context_size"]) == {"1", "3", "5"} and set(sweep["view"]) == {"grade=2"}
    assert {"seed_mmr@0.5", "seed_mmr@0.9", "seed_rrf"} <= set(sweep["name"])
    for f in ["baseline/context_sweep_eq2.png", "baseline/context_sweep_mmr_eq2.png", "summary.json"]:
        assert (tmp_path / "covid" / f).exists(), f


def test_run_exp01_multi_dataset_and_comparison(toy, tmp_path, monkeypatch):
    import pec.experiments.exp01 as exp01

    store = toy["store"]
    dl = normalise_qrels(toy["qrels"])
    covid = covid_qrels(toy)
    topics = toy["topics"].astype({"qid": str})

    def fake_load_context(cfg, dataset):
        rel = Relevance.from_config(cfg["datasets"][dataset]["relevance"])
        qrels = covid if dataset == "trec_covid" else dl
        return exp01.build_context(store, topics, qrels, query_vectors(store, qrels, rel.target),
                                   index_key=f"toy_{dataset}", relevance=rel, dataset=dataset)

    monkeypatch.setattr(exp01, "load_context", fake_load_context)
    cfg = small_exp01_config(tmp_path)
    summaries = exp01.run_exp01(cfg, parts=["baseline"], datasets=["dl19", "trec_covid"])
    assert set(summaries) == {"dl19", "trec_covid"}
    assert (tmp_path / "exp01_dl19/baseline/best_context_size.csv").exists()
    assert (tmp_path / "exp01_trec_covid/baseline/best_context_size.csv").exists()
    comp = pd.read_csv(tmp_path / "exp01_comparison/best_context_size.csv", dtype={"best_k": str})
    assert set(comp["dataset"]) == {"dl19", "trec_covid"} and "all" not in set(comp["best_k"])
    target_rows = comp[comp["target_view"] & (comp["name"] == "centroid") & (comp["metric"] == "nDCG@10")]
    assert set(target_rows["view"]) == {"grade=3", "grade=2"}
    curves = pd.read_csv(tmp_path / "exp01_comparison/centroid_ndcg10_by_dataset.csv", dtype={"context_size": str})
    assert "all" not in set(curves["context_size"])
    assert (tmp_path / "exp01_comparison/centroid_ndcg10_by_dataset.png").exists()
