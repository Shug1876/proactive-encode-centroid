"""Inverse context MMR, BM25 + RRF, reuse of stored baseline results, the query
formulation group / context-size toggles and the overview figures, on the synthetic corpus."""
import runpy
import sys

import faiss
import numpy as np
import pandas as pd
import pyterrier as pt
import pytest

from pec.data.qrels import normalise_qrels
from pec.data.seeds import context_draws
from pec.evaluate import cached_sweep, figure_groups
from pec.index.faiss_store import DocnoMap, FaissDocStore
from pec.pipeline import CentroidEncoder, FaissRetriever, InverseContextMMR, seed_bm25_rrf
from pec.text_retrievers import ResidualTextRetriever

from test_query_formulation import FakeEncoder, FakeLLM, FakeTextRetriever, toy_texts


def unit(rows):
    v = np.asarray(rows, dtype=np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


@pytest.fixture
def small_store():
    # c1, c2: context; d1 close to c1, d2 between c1 and c2, d3 close to c2
    vecs = unit([[1, 0, 0], [0, 1, 0], [0.95, 0.1, 0.3], [0.7, 0.7, 0.1], [0.1, 0.95, 0.3]])
    index = faiss.IndexFlatIP(3)
    index.add(vecs)
    return FaissDocStore(index, DocnoMap(docnos=["c1", "c2", "d1", "d2", "d3"])), vecs


def test_inverse_context_mmr_scores_by_hand(small_store):
    store, vecs = small_store
    topics = pd.DataFrame({"qid": ["q"], "seed_docnos": [("c1", "c2")]})
    lam = 0.7
    run = InverseContextMMR(store, 3, Lambda=lam, pool=10)(topics).set_index("docno")["score"]
    assert set(run.index) == {"d1", "d2", "d3"}  # context removed
    for d, i in (("d1", 2), ("d2", 3), ("d3", 4)):
        sims = vecs[i] @ vecs[:2].T
        s_r, s_c = sims.max(), sims.min()  # q = the more similar context doc, the other one is C \ {q}
        assert run[d] == pytest.approx(-lam * s_r + (1 - lam) * s_c, abs=1e-5)
    assert run.idxmax() == "d2"  # least tied to a single context document, closest to the other
    single = InverseContextMMR(store, 3, Lambda=lam, pool=10)(topics.assign(seed_docnos=[("c1",)]))
    single = single.set_index("docno")["score"]
    for d, i in (("d1", 2), ("d2", 3), ("d3", 4)):
        assert single[d] == pytest.approx(-lam * float(vecs[i] @ vecs[0]), abs=1e-5)  # no other context: S_c = 0
    with pytest.raises(ValueError):
        InverseContextMMR(store, Lambda=1.5)


def test_seed_bm25_rrf_fuses_one_bm25_query_per_context_passage(toy):
    texts = toy_texts(toy)
    bm25 = ResidualTextRetriever(FakeTextRetriever(texts, 60), 40)
    topics = pd.DataFrame({"qid": ["q0", "q1"], "seed_docnos": [("d14", "d40"), ("d50",)]})
    run = seed_bm25_rrf(texts, bm25, 30, rrf_k=60)(topics)
    assert run.groupby("qid").size().max() <= 30
    for qid, seeds in zip(topics["qid"], topics["seed_docnos"]):
        assert not set(seeds) & set(run[run["qid"] == qid]["docno"])
    # q0's two passages are about queries 0 and 1, so both topics' documents are fused in
    q0 = run[run["qid"] == "q0"]
    about = q0["docno"].map(lambda d: int(d[1:]) // 34)
    assert {0, 1} <= set(about.head(10)) and q0["score"].max() <= 2 / 60 + 1e-9
    one = run[run["qid"] == "q1"]  # a single passage: its BM25 ranking, unfused
    ranking = bm25(pd.DataFrame({"qid": ["q1"], "query": [texts["d50"]], "seed_docnos": [("d50",)]}))
    assert one["docno"].tolist() == ranking["docno"].head(30).tolist()


class Counting(pt.Transformer):
    def __init__(self, inner):
        self.inner, self.calls = inner, 0

    def transform(self, inp):
        self.calls += 1
        return self.inner(inp)


def test_cached_sweep_reuses_reruns_and_accumulates(toy, tmp_path):
    store, qrels = toy["store"], normalise_qrels(toy["qrels"])
    contexts = context_draws(qrels, [1, 3], repeats=2)
    settings = {"random_seed": 0, "num_results": 30}
    a = Counting(CentroidEncoder(store) >> FaissRetriever(store, 30))
    m, pq, ran = cached_sweep({"centroid": a}, contexts, qrels, [10], tmp_path, settings)
    assert ran == ["centroid"] and a.calls == 4 and len(m) == 2 * 4  # views x draws

    def save(metrics, perquery):  # what run_baseline / write_context_sweep store
        metrics.to_csv(tmp_path / "metrics_by_context_size.csv", index=False)
        perquery.to_csv(tmp_path / "metrics_perquery.csv", index=False)

    save(m, pq)
    m2, _, ran = cached_sweep({"centroid": a}, contexts, qrels, [10], tmp_path, settings)
    assert ran == [] and a.calls == 4 and len(m2) == len(m)  # everything reused

    b = Counting(CentroidEncoder(store) >> FaissRetriever(store, 30))
    m3, pq3, ran = cached_sweep({"centroid": a, "other": b}, contexts, qrels, [10], tmp_path, settings,
                                run_sizes={"other": {"1"}})
    assert ran == ["other"] and a.calls == 4 and b.calls == 2  # "other" only at k=1
    assert set(m3[m3["name"] == "other"]["context_size"]) == {"1"}
    save(m3, pq3)
    m4, _, _ = cached_sweep({"other": b}, contexts, qrels, [10], tmp_path, settings, run_sizes={"other": {"1"}})
    assert set(m4["name"]) == {"centroid", "other"} and b.calls == 2  # stored runs of other invocations kept

    m5, _, ran = cached_sweep({"centroid": a, "other": b}, contexts, qrels, [10], tmp_path, settings,
                              run_sizes={"other": {"1"}}, rerun=["centroid"])
    assert ran == ["centroid"] and a.calls == 8 and b.calls == 2 and len(m5) == len(m4)
    m6, _, ran = cached_sweep({"centroid": a}, contexts, qrels, [10], tmp_path, {**settings, "random_seed": 1})
    assert ran == ["centroid"] and set(m6["name"]) == {"centroid"}  # other settings: nothing reused


def test_figure_groups_from_stored_names():
    names = {"centroid", "seed_fusion", "seed_rrf", "bm25_DaaQ", "seed_mmr@0.9", "seed_mmr@0.5",
             "seed_inv_mmr@0.7", "zero_shot", "doc2query", "zero_shot@bm25", "sliding_window@bm25", "true_query"}
    groups = figure_groups(names)
    assert groups["main"] == ["centroid", "seed_rrf", "zero_shot", "doc2query"]  # the E5 methods; seed_fusion retired
    assert groups["bm25"] == ["bm25_DaaQ"]
    assert groups["mmr"] == ["centroid", "seed_mmr@0.5", "seed_mmr@0.9"]
    assert groups["inv_mmr"] == ["centroid", "seed_inv_mmr@0.7"]
    assert "llm" not in groups and "text" not in groups  # the LLM and text methods are in main now
    assert groups["qf_bm25"] == ["zero_shot@bm25", "sliding_window@bm25"]
    every = {"centroid", "seed_rrf", "zero_shot", "static_icl", "sliding_window", "doc2query", "bm25_DaaQ",
             "bm25_centroid", "bm25_DaaQ_nobm25", "bm25_centroid_nobm25"}
    groups = figure_groups(every)
    assert groups["main"] == ["centroid", "seed_rrf", "zero_shot", "static_icl", "sliding_window", "doc2query"]
    assert groups["bm25"] == ["bm25_DaaQ", "bm25_centroid", "bm25_DaaQ_nobm25", "bm25_centroid_nobm25"]


def test_qf_groups_sizes_and_separate_runs_accumulate(toy, tmp_path, monkeypatch):
    import pec.index.e5 as e5
    import pec.query_formulation as qfm
    import types
    from pec.config import load_config
    from pec.experiments.exp01 import build_context, dataset_config, qf_methods, qf_run_names, run_parts

    assert qf_methods({"methods": ["zero_shot", "static_icl", "sliding_window", "doc2query"], "groups": ["llm"]}) \
        == ["zero_shot", "static_icl"]
    assert qf_run_names({"methods": ["sliding_window"], "groups": ["text"], "retrievers": ["e5", "bm25"]}) \
        == ["sliding_window", "sliding_window@bm25", "true_query@bm25"]
    with pytest.raises(ValueError):
        qf_methods({"methods": ["zero_shot"], "groups": ["llms"]})

    store, qrels = toy["store"], normalise_qrels(toy["qrels"])
    encoder = FakeEncoder(store, qrels)
    monkeypatch.setattr(e5, "e5_model", lambda name, **kw: types.SimpleNamespace(query_encoder=lambda: encoder))
    llm = FakeLLM()
    monkeypatch.setattr(qfm.QF, "client", property(lambda self: llm))
    texts = toy_texts(toy)
    ctx = build_context(store, toy["topics"].astype({"qid": str}), qrels, encoder.centres, texts, index_key="toy")
    cfg = dataset_config(load_config("configs/exp01_dl19.yaml"), "dl19")
    c = cfg["exp01"]["baseline"]
    c.update(num_results=50, recall_ks=[10, 50], context_sizes=[1, 3], repeats=2, mmr={"lambdas": [0.5]},
             inv_mmr={"lambdas": [0.5, 0.9], "pool": 100}, bm25_rrf={"enabled": False}, bm25_centroid={"enabled": False})
    c["query_formulation"].update(methods=["zero_shot", "sliding_window"], retrievers=["e5"], groups=["llm"],
                                  context_sizes=[1])
    cfg["cache_dir"] = str(tmp_path / "cache")

    run_parts(ctx, cfg, tmp_path, parts=["baseline"])  # LLM group, every other baseline
    m = pd.read_csv(tmp_path / "baseline/metrics_by_context_size.csv", dtype={"context_size": str})
    assert set(m[m["name"] == "zero_shot"]["context_size"]) == {"1"}  # query formulation only at k=1
    assert "sliding_window" not in set(m["name"]) and {"seed_inv_mmr@0.5", "seed_inv_mmr@0.9"} <= set(m["name"])
    prompts = len(llm.prompts)
    assert prompts == len(context_draws(qrels, [1], 2, cfg["exp01"]["random_seed"]))

    c["query_formulation"]["groups"] = ["text"]  # the other group, separately, without the other baselines
    c["other_baselines"] = False
    run_parts(ctx, cfg, tmp_path, parts=["baseline"])
    m = pd.read_csv(tmp_path / "baseline/metrics_by_context_size.csv", dtype={"context_size": str})
    assert {"zero_shot", "sliding_window", "centroid", "seed_mmr@0.5", "true_query"} <= set(m["name"])
    assert len(llm.prompts) == prompts  # zero_shot results reused, no new LLM call
    assert not m.duplicated(["name", "view", "context_size", "repeat"]).any()
    for f in ["context_sweep_ge2.png", "context_sweep_inv_mmr_ge2.png", "context_sweep_mmr_eq3.png",
              "context_sweep_overview_ndcg10_ge2.png",
              "context_sweep_overview_r50_eq3.png", "sweep_settings.json"]:
        assert (tmp_path / "baseline" / f).exists(), f


def test_cli_toggles(monkeypatch):
    import pec.experiments.exp01 as exp01

    seen = {}
    monkeypatch.setattr(exp01, "run_exp01", lambda cfg, parts=None, datasets=None: seen.update(cfg=cfg, parts=parts) or {})
    argv = ["exp01_dl19.py", "--qf-groups", "llm", "--qf-only", "--rerun", "seed_rrf"]
    monkeypatch.setattr(sys, "argv", argv)
    runpy.run_path("scripts/exp01_dl19.py", run_name="__main__")
    b = seen["cfg"]["exp01"]["baseline"]
    assert b["query_formulation"]["groups"] == ["llm"] and b["other_baselines"] is False and b["rerun"] == ["seed_rrf"]
    monkeypatch.setattr(sys, "argv", ["exp01_dl19.py", "--rerun"])
    runpy.run_path("scripts/exp01_dl19.py", run_name="__main__")
    assert seen["cfg"]["exp01"]["baseline"]["rerun"] is True
    assert "all" not in [str(k) for k in seen["cfg"]["exp01"]["baseline"]["context_sizes"]]  # "all" disabled


def test_retired_seed_fusion_results_are_deleted_and_never_plotted(toy, tmp_path, monkeypatch):
    import pec.analysis.plots as plots
    import pec.evaluate as ev
    from pec.evaluate import metric_names, write_context_sweep
    from pec.pipeline import SeedFusionRetriever

    store, qrels = toy["store"], normalise_qrels(toy["qrels"])
    contexts = context_draws(qrels, [1, 3], repeats=1)
    settings = {"random_seed": 0}
    old = {"centroid": CentroidEncoder(store) >> FaissRetriever(store, 30), "seed_fusion": SeedFusionRetriever(store, 30)}
    m, pq, _ = cached_sweep(old, contexts, qrels, [10], tmp_path, settings)  # results stored before the removal
    m.to_csv(tmp_path / "metrics_by_context_size.csv", index=False)
    pq.to_csv(tmp_path / "metrics_perquery.csv", index=False)
    m, pq, ran = cached_sweep({"centroid": old["centroid"]}, contexts, qrels, [10], tmp_path, settings)
    assert ran == [] and "seed_fusion" not in set(m["name"]) | set(pq["name"])

    plotted = []
    monkeypatch.setattr(plots, "lines", lambda ax, xs, ys, **kw: plotted.extend(xs))
    real_overview = ev.plot_overview
    monkeypatch.setattr(ev, "plot_overview", lambda summary, *a, **kw: (plotted.extend(summary["name"]),
                                                                        real_overview(summary, *a, **kw)))
    stale = pd.concat([m, m.assign(name="seed_fusion")], ignore_index=True)  # even if passed in, it is dropped
    summary, best = write_context_sweep(stale, metric_names([10]), tmp_path, ["centroid", "seed_fusion"],
                                        figure_groups={"mmr": ["centroid", "seed_fusion"]})
    stored = pd.read_csv(tmp_path / "metrics_by_context_size.csv")
    assert "seed_fusion" not in set(stored["name"]) | set(summary["name"]) | set(best["name"]) | set(plotted)
    assert "centroid" in plotted


def test_sparse_centroid_by_hand():
    from pec.pipeline import sparse_centroid

    analyse = lambda t: t.split()
    # p1: a a b (a: 2/3, b: 1/3); p2: a c (a: 1/2, c: 1/2) -> mean a 7/12, c 1/4, b 1/6
    full = sparse_centroid(["a a b", "a c"], analyse, terms=None)
    assert full == pytest.approx({"a": 1.0, "c": (1 / 4) / (7 / 12), "b": (1 / 6) / (7 / 12)})
    assert sparse_centroid(["a a b", "a c"], analyse, terms=2) == pytest.approx({"a": 1.0, "c": (1 / 4) / (7 / 12)})
    assert sparse_centroid(["", "   "], analyse) == {}  # passages without terms are skipped
    assert sparse_centroid(["a a b", ""], analyse, terms=None) == pytest.approx({"a": 1.0, "b": 0.5})


def test_bm25_centroid_weighted_query_and_context_removal(toy):
    from pec.pipeline import SparseCentroidRetriever

    texts = toy_texts(toy)
    seen = []

    class WeightedOverlap(pt.Transformer):  # a stand-in for PISA BM25 with query_weighted=True
        def transform(self, inp):
            assert list(inp.columns) == ["qid", "query_toks"]
            seen.extend(inp["query_toks"])
            rows = []
            for qid, toks in zip(inp["qid"], inp["query_toks"]):
                for d, t in texts.items():
                    score = sum(w for term, w in toks.items() if term in set(t.split()))
                    rows.append({"qid": qid, "docno": d, "score": score - int(d[1:]) * 1e-9})
            return pt.model.add_ranks(pd.DataFrame(rows))

    topics = pd.DataFrame({"qid": ["q0", "q1"], "seed_docnos": [("d14", "d20"), ("d50",)]})
    run = SparseCentroidRetriever(texts, str.split, WeightedOverlap(), num_results=30, terms=3)(topics)
    assert run.groupby("qid").size().max() == 30
    for qid, seeds in zip(topics["qid"], topics["seed_docnos"]):
        assert not set(seeds) & set(run[run["qid"] == qid]["docno"])  # context removed
    assert all(len(toks) <= 3 and max(toks.values()) == pytest.approx(1.0) for toks in seen)
    seen.clear()
    SparseCentroidRetriever(texts, str.split, WeightedOverlap(), num_results=30, terms=None)(topics)
    assert len(seen[0]) == len(set(texts["d14"].split()) | set(texts["d20"].split()))  # all_terms keeps every term


def test_bm25_centroid_in_exp01(toy, tmp_path, monkeypatch):
    import pec.experiments.exp01 as exp01
    from pec.config import load_config
    from pec.evaluate import figure_groups

    assert exp01.bm25_centroid_terms({}) == 100 and exp01.bm25_centroid_terms({"terms": 20}) == 20
    assert exp01.bm25_centroid_terms({"terms": 20, "all_terms": True}) is None
    with pytest.raises(ValueError):
        exp01.bm25_centroid_terms({"terms": -1})

    store, qrels = toy["store"], normalise_qrels(toy["qrels"])
    texts = toy_texts(toy)
    depths, queries = [], []

    def fake_parts(cfg, dataset, depth):
        depths.append(depth)

        def retrieve(inp):
            queries.extend(inp["query_toks"])
            rows = [{"qid": q, "docno": d, "score": sum(w for t, w in toks.items() if t in set(txt.split()))}
                    for q, toks in zip(inp["qid"], inp["query_toks"]) for d, txt in texts.items()]
            return pt.model.add_ranks(pd.DataFrame(rows).sort_values(["qid", "score"], ascending=[True, False]))

        return str.split, pt.apply.generic(retrieve)

    monkeypatch.setattr(exp01, "bm25_centroid_parts", fake_parts)
    import pec.text_retrievers as tr
    monkeypatch.setattr(tr, "build_retrievers", lambda cfg, ds, names, num_results=1000, depth=None, *a, **kw: {
        n: ResidualTextRetriever(FakeTextRetriever(texts, depth or num_results), num_results) for n in names})
    ctx = exp01.build_context(store, toy["topics"].astype({"qid": str}), qrels, texts=texts, index_key="toy")
    cfg = exp01.dataset_config(load_config("configs/exp01_dl19.yaml"), "dl19")
    c = cfg["exp01"]["baseline"]
    assert c["bm25_centroid"]["enabled"] is True and {"terms", "all_terms"} <= set(c["bm25_centroid"])
    c.update(num_results=40, recall_ks=[10, 40], context_sizes=[1, 3], repeats=2, mmr={"lambdas": []},
             inv_mmr={"lambdas": []}, bm25_rrf={"enabled": False})
    c["bm25_centroid"].update(terms=5, all_terms=False)
    c["query_formulation"]["methods"] = []
    exp01.run_parts(ctx, cfg, tmp_path, parts=["baseline"])
    m = pd.read_csv(tmp_path / "baseline/metrics_by_context_size.csv", dtype={"context_size": str})
    assert set(m[m["name"] == "bm25_centroid"]["context_size"]) == {"1", "3"}  # every size, like the others
    assert depths == [40 + 3] and queries and all(len(q) <= 5 for q in queries)
    groups = figure_groups(set(m["name"]))
    assert groups["main"] == ["centroid", "seed_rrf"] and groups["bm25"] == ["bm25_centroid"]
    assert "true_query@bm25" in set(m["name"])  # the BM25 methods' Oracle
    assert (tmp_path / "baseline/context_sweep_bm25_eq3.png").exists()


def test_stored_seed_bm25_rrf_results_are_read_as_bm25_DaaQ(toy, tmp_path):
    from pec.evaluate import drop_retired

    stored = pd.DataFrame({"name": ["seed_bm25_rrf", "seed_bm25_rrf_nobm25", "centroid", "seed_fusion"],
                           "context_size": "1", "repeat": 0, "nDCG@10": [0.1, 0.2, 0.3, 0.4]})
    out = drop_retired(stored)
    assert list(out["name"]) == ["bm25_DaaQ", "bm25_DaaQ_nobm25", "centroid"]
    assert list(out["nDCG@10"]) == [0.1, 0.2, 0.3]
    # cached_sweep reuses them under the new name: nothing is run again
    store, qrels = toy["store"], normalise_qrels(toy["qrels"])
    contexts = context_draws(qrels, [1], repeats=1)
    a = Counting(CentroidEncoder(store) >> FaissRetriever(store, 30))
    m, pq, _ = cached_sweep({"centroid": a}, contexts, qrels, [10], tmp_path, {"s": 1})
    m.assign(name="seed_bm25_rrf").to_csv(tmp_path / "metrics_by_context_size.csv", index=False)
    pq.assign(name="seed_bm25_rrf").to_csv(tmp_path / "metrics_perquery.csv", index=False)
    b = Counting(CentroidEncoder(store) >> FaissRetriever(store, 30))
    m2, _, ran = cached_sweep({"bm25_DaaQ": b}, contexts, qrels, [10], tmp_path, {"s": 1})
    assert ran == [] and b.calls == 0 and set(m2["name"]) == {"bm25_DaaQ"}
