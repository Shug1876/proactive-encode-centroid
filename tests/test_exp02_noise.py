"""Experiment 2: the centroid on contexts in which a share α of the documents are judged
non-relevant and the rest partially relevant, on the synthetic corpus."""
import json
import runpy
import sys

import pandas as pd
import pytest

from pec.config import load_config
from pec.data.qrels import normalise_qrels
from pec.data.seeds import alpha_counts, alpha_label, context_draws, noise_draws, parse_alpha

from test_datasets_mmr import COVID, covid_qrels, query_vectors
from test_query_formulation import FakeTextRetriever, toy_texts

ALPHAS = ["0.0", "0.2", "0.4", "0.6", "0.8"]


def test_alpha_parsing():
    assert parse_alpha("0.2") == 0.2 and parse_alpha(0) == 0.0
    assert [alpha_label(a) for a in (0, 0.2, "0.40", 0.6, 0.8)] == ALPHAS and alpha_label(0.25) == "0.25"
    assert alpha_counts(0.2, 10) == (2, 8) and alpha_counts(0.0, 10) == (0, 10) and alpha_counts(0.8, 10) == (8, 2)
    for bad in (1.0, -0.2, "a", None):
        with pytest.raises(ValueError):
            parse_alpha(bad)
    with pytest.raises(ValueError):
        alpha_counts(0.25, 10)  # 2.5 documents


def test_noise_draws_use_only_judged_grade0_and_are_nested(toy):
    qrels = normalise_qrels(toy["qrels"])
    # q3 keeps only 7 judged non-relevant documents: not enough for α = 0.8, so it is left out of every α
    q3_noise = qrels[(qrels["qid"] == "q3") & (qrels["label"] == 0)]["docno"].tolist()
    qrels = qrels[~qrels["docno"].isin(q3_noise[7:])]
    draws = noise_draws(qrels, 10, [0, 0.2, 0.4, 0.6, 0.8], repeats=3, random_seed=42)
    assert set(draws["qid"]) == {"q0", "q1", "q2"} and list(dict.fromkeys(draws["alpha"])) == ALPHAS
    assert (draws.groupby("alpha")["qid"].nunique() == 3).all()  # the same queries for every α
    label = {(q, d): g for q, d, g in qrels[["qid", "docno", "label"]].itertuples(index=False)}
    for r in draws.itertuples():
        n_noise, n_context = alpha_counts(r.alpha, 10)
        assert len(r.seed_docnos) == 10 and len(set(r.seed_docnos)) == 10
        assert [label.get((r.qid, d)) for d in r.noise_docnos] == [0] * n_noise  # judged grade 0, never unjudged
        assert [label.get((r.qid, d)) for d in r.context_docnos] == [1] * n_context
        assert set(r.seed_docnos) == set(r.noise_docnos) | set(r.context_docnos)
    for (qid, repeat), g in draws.groupby(["qid", "repeat"]):
        g = g.set_index("alpha")
        for a, b in zip(ALPHAS, ALPHAS[1:]):  # a larger α only swaps partially relevant for non-relevant documents
            assert set(g.loc[b, "context_docnos"]) <= set(g.loc[a, "context_docnos"])
            assert set(g.loc[b, "noise_docnos"]) >= set(g.loc[a, "noise_docnos"])
    # α = 0 is Experiment 1's m = 10 context for the same seed and repeat
    exp01 = context_draws(qrels, [10], 3, 42).set_index(["qid", "repeat"])["seed_docnos"]
    clean = draws[draws["alpha"] == "0.0"].set_index(["qid", "repeat"])["seed_docnos"]
    assert all(clean[k] == exp01[k] for k in clean.index)
    assert draws.equals(noise_draws(qrels, 10, ALPHAS, repeats=3, random_seed=42))
    assert not draws.equals(noise_draws(qrels, 10, ALPHAS, repeats=3, random_seed=1))
    with pytest.raises(ValueError):
        noise_draws(qrels, 10, [0.2, 0.20], repeats=1)  # the same α twice


RUNS = {"centroid", "bm25_DaaQ", "bm25_centroid"}
ORACLES = {"true_query", "true_query@bm25"}


@pytest.fixture
def fake_bm25(toy, monkeypatch):
    """Stand-ins for the passage texts and the BM25 PISA index (word-overlap scoring)."""
    import pyterrier as pt

    import pec.experiments.exp01 as exp01
    import pec.experiments.exp02 as exp02
    import pec.text_retrievers as tr
    from pec.text_retrievers import ResidualTextRetriever

    texts = toy_texts(toy)
    loaded = []
    monkeypatch.setattr(exp02, "load_texts", lambda cfg, ds, docnos: loaded.append(set(docnos)) or
                        {d: texts[d] for d in docnos})
    monkeypatch.setattr(tr, "build_retrievers", lambda cfg, ds, names, num_results=1000, depth=None, *a, **kw: {
        n: ResidualTextRetriever(FakeTextRetriever(texts, depth or num_results), num_results) for n in names})

    def weighted(inp):
        rows = [{"qid": q, "docno": d, "score": sum(w for t, w in toks.items() if t in set(txt.split()))}
                for q, toks in zip(inp["qid"], inp["query_toks"]) for d, txt in texts.items()]
        return pt.model.add_ranks(pd.DataFrame(rows).sort_values(["qid", "score"], ascending=[True, False]))

    monkeypatch.setattr(exp01, "bm25_centroid_parts", lambda cfg, ds, depth, k1=None: (str.split, pt.apply.generic(weighted)))
    return loaded


def small_config(tmp_path, datasets=("dl19",)):
    cfg = load_config("configs/exp02_noise.yaml")
    cfg["exp02"].update(datasets=list(datasets), num_results=50, recall_ks=[10, 50], repeats=2)
    cfg["output_dir"] = str(tmp_path)
    return cfg


def test_run_exp02_outputs_and_reuse(toy, tmp_path, monkeypatch, fake_bm25):
    import pec.experiments.exp02 as exp02
    from pec.data.qrels import Relevance
    from pec.experiments.exp01 import build_context

    store, topics = toy["store"], toy["topics"].astype({"qid": str})
    calls = []

    def fake_load_context(cfg, dataset, section="exp01"):
        assert section == "exp02"
        calls.append(dataset)
        rel = Relevance.from_config(cfg["datasets"][dataset]["relevance"])
        qrels = covid_qrels(toy) if dataset == "trec_covid" else normalise_qrels(toy["qrels"])
        return build_context(store, topics, qrels, query_vectors(store, qrels, rel.target), index_key="toy",
                             relevance=rel, dataset=dataset)

    monkeypatch.setattr(exp02, "load_context", fake_load_context)
    cfg = small_config(tmp_path, ["dl19", "trec_covid"])
    alphas = [alpha_label(a) for a in cfg["exp02"]["alphas"]]
    assert alphas == ALPHAS  # the configured α: 0, 0.2, ..., 0.8
    summaries = exp02.run_exp02(cfg)
    assert set(summaries) == {"dl19", "trec_covid"}
    out = tmp_path / "exp02_dl19"
    m = pd.read_csv(out / "metrics_by_alpha.csv", dtype={"alpha": str})
    assert set(m["name"]) == RUNS | ORACLES and set(m["alpha"]) == set(alphas)
    # every run as a share of its own retriever's Oracle, on the same α and draw
    key = ["view", "alpha", "repeat"]
    for run, oracle in (("centroid", "true_query"), ("bm25_DaaQ", "true_query@bm25"),
                        ("bm25_centroid", "true_query@bm25")):
        j = m[m["name"] == run].merge(m[m["name"] == oracle], on=key, suffixes=("", "_o"))
        expect = 100 * j["nDCG@10"] / j["nDCG@10_o"].replace(0, float("nan"))
        assert (j["nDCG@10_pct_true_query"] - expect).abs().max() < 1e-9, run
    assert fake_bm25 and all(len(d) > 0 for d in fake_bm25)  # texts of every context passage, grade 0 included
    assert m.groupby("alpha")["n_queries"].nunique().eq(1).all()  # same queries for every α
    s = pd.read_csv(out / "summary_by_alpha.csv", dtype={"alpha": str})
    cen = s[(s["name"] == "centroid") & (s["view"] == "grade=3")].set_index("alpha")["nDCG@10_mean"]
    assert list(cen.index) == alphas and cen["0.0"] > cen["0.8"]  # non-relevant documents hurt
    assert (s[s["name"].isin(ORACLES)]["nDCG@10_pct_true_query_mean"].round(9) == 100).all()
    ctxs = pd.read_csv(out / "contexts.csv", dtype={"alpha": str})
    assert len(ctxs) == 4 * len(alphas) * 2 and ctxs["seed_docnos"].str.split().map(len).eq(5).all()  # m = 5
    assert (ctxs["n_nonrelevant"] == ctxs["alpha"].astype(float).mul(5).round().astype(int)).all()
    assert (ctxs["n_nonrelevant"] + ctxs["n_partial"] == 5).all()
    every_doc = {d for docs in ctxs["seed_docnos"].str.split() for d in docs}
    assert every_doc <= set().union(*fake_bm25)
    for f in ["noise_sweep_ge2.png", "noise_sweep_eq3.png", "sweep_settings.json", "metrics_perquery.csv"]:
        assert (out / f).exists(), f
    assert (tmp_path / "exp02_trec_covid/noise_sweep_eq2.png").exists()
    comp = pd.read_csv(tmp_path / "exp02_comparison/centroid_ndcg10_by_dataset.csv", dtype={"alpha": str})
    assert set(comp["dataset"]) == {"dl19", "trec_covid"} and set(comp["view"]) == {"grade=3", "grade=2"}
    assert (tmp_path / "exp02_comparison/centroid_ndcg10_by_dataset.png").exists()
    summary = json.loads((out / "summary.json").read_text())
    assert summary["queries_used"] == 4 and set(summary["centroid_ndcg10_grade=3"]) == set(alphas)
    assert summary["context_size"] == 5 and set(summary["runs"]) == RUNS
    assert set(summary["bm25_centroid_ndcg10_grade=3"]) == set(alphas)

    from test_baselines_new import Counting

    counted = []
    real = exp02.noise_pipelines
    monkeypatch.setattr(exp02, "noise_pipelines", lambda *a, **kw: {
        n: counted.append(Counting(p)) or counted[-1] for n, p in real(*a, **kw).items()})
    exp02.run_exp02(small_config(tmp_path, ["dl19"]))
    assert sum(c.calls for c in counted) == 0  # every stored result reused
    rerun = small_config(tmp_path, ["dl19"])
    rerun["exp02"]["rerun"] = True
    counted.clear()
    exp02.run_exp02(rerun)
    assert sum(c.calls for c in counted) == len(RUNS | ORACLES) * len(alphas) * 2  # runs x α x repeats
    assert len(pd.read_csv(out / "metrics_by_alpha.csv")) == len(m)


def test_exp02_covid_grades_use_grade0_and_grade1(toy, tmp_path, fake_bm25):
    from pec.experiments.exp01 import build_context
    from pec.experiments.exp02 import run_noise

    store, qrels = toy["store"], covid_qrels(toy)  # 0-2 scale: grades 1-2 -> 1, 3 -> 2
    ctx = build_context(store, toy["topics"].astype({"qid": str}), qrels, query_vectors(store, qrels, 2),
                        index_key="toy", relevance=COVID, dataset="trec_covid")
    cfg = small_config(tmp_path)
    summary = run_noise(ctx, cfg["exp02"], tmp_path / "covid", random_seed=0, cfg=cfg)
    assert (summary["context_grade"], summary["target_grade"], summary["noise_grade"]) == (1, 2, 0)
    ctxs = pd.read_csv(tmp_path / "covid/contexts.csv", dtype={"alpha": str}).fillna("")
    label = {(q, d): g for q, d, g in qrels[["qid", "docno", "label"]].itertuples(index=False)}
    for r in ctxs.itertuples():
        assert all(label[(r.qid, d)] == 0 for d in r.noise_docnos.split())
        assert all(label[(r.qid, d)] == 1 for d in r.context_docnos.split())


def test_exp02_cli(monkeypatch):
    import pec.experiments.exp02 as exp02

    seen = {}
    monkeypatch.setattr(exp02, "run_exp02", lambda cfg, datasets: seen.update(cfg=cfg, datasets=datasets) or {})
    monkeypatch.setattr(sys, "argv", ["exp02_noise.py", "--datasets", "dl20", "--rerun"])
    runpy.run_path("scripts/exp02_noise.py", run_name="__main__")
    assert seen["datasets"] == ["dl20"] and seen["cfg"]["exp02"]["rerun"] is True
    ec = seen["cfg"]["exp02"]
    assert ec["context_size"] == 5 and ec["random_seed"] == 42
    assert ec["runs"] == ["centroid", "bm25_DaaQ", "bm25_centroid"]
    assert ec["alphas"] == [0.0, 0.2, 0.4, 0.6, 0.8] and ec["plot_titles"] is False
    assert ec["reference_label"] == "Oracle"
    monkeypatch.setattr(sys, "argv", ["exp02_noise.py", "--alphas", "0", "0.4", "--titles"])
    runpy.run_path("scripts/exp02_noise.py", run_name="__main__")
    assert seen["cfg"]["exp02"]["alphas"] == [0.0, 0.4] and seen["cfg"]["exp02"]["plot_titles"] is True


def test_exp02_figures_have_no_titles_and_name_the_oracle(toy, tmp_path, monkeypatch, fake_bm25):
    import matplotlib.figure
    import pec.analysis.plots as plots
    import pec.experiments.exp02 as exp02
    from pec.experiments.exp01 import build_context

    monkeypatch.setattr(plots, "ORACLE_LINES", True)  # the Oracle lines are off by default

    seen = []
    real_save = exp02.save
    monkeypatch.setattr(exp02, "save", lambda fig, path, **kw: (seen.append((path.name, fig)), real_save(fig, path, **kw)))
    texts = lambda fig: {"suptitle": fig._suptitle.get_text() if fig._suptitle else "",
                         "titles": [ax.get_title() for ax in fig.axes],
                         "xlabels": {ax.get_xlabel() for ax in fig.axes if ax.get_xlabel()},
                         "ylabels": [ax.get_ylabel() for ax in fig.axes],
                         "legend": {t.get_text() for ax in fig.axes if ax.get_legend()
                                    for t in ax.get_legend().get_texts()}}
    store, qrels = toy["store"], normalise_qrels(toy["qrels"])
    ctx = build_context(store, toy["topics"].astype({"qid": str}), qrels, query_vectors(store, qrels, 3),
                        index_key="toy")
    cfg = small_config(tmp_path)
    c = cfg["exp02"]
    exp02.run_noise(ctx, c, tmp_path / "plain", random_seed=0, cfg=cfg)
    for name, fig in seen:
        t = texts(fig)
        assert t["suptitle"] == "" and set(t["titles"]) == {""}, name  # no titles by default
        assert t["xlabels"] == {"α (share of non-relevant documents), m = 5"}
        assert {"E5_centroid", "bm25_DaaQ", "bm25_centroid", "Oracle (E5)", "Oracle (BM25)"} <= t["legend"]
        assert "true_query" not in t["legend"] and not any("true_query" in y for y in t["ylabels"])
        assert "nDCG@10 (% of Oracle (own retriever))" in t["ylabels"]
    seen.clear()  # the centroid alone: one Oracle, plain "Oracle"
    exp02.run_noise(ctx, {**c, "runs": ["centroid"]}, tmp_path / "e5", random_seed=0, cfg=cfg)
    t = texts(seen[0][1])
    assert "Oracle" in t["legend"] and "nDCG@10 (% of Oracle)" in t["ylabels"]
    assert isinstance(seen[0][1], matplotlib.figure.Figure)
    seen.clear()
    exp02.run_noise(ctx, {**c, "plot_titles": True}, tmp_path / "titled", random_seed=0, cfg=cfg)
    assert all(texts(fig)["suptitle"] for _, fig in seen)  # the toggle brings them back



def test_exp02_figures_use_the_pinned_oracle(toy, tmp_path, monkeypatch, fake_bm25):
    import pec.analysis.plots as plots
    import pec.experiments.exp02 as exp02
    from pec.experiments.exp01 import build_context

    monkeypatch.setattr(plots, "ORACLE_LINES", True)

    seen = []
    real_save = exp02.save
    monkeypatch.setattr(exp02, "save", lambda fig, path, **kw: (seen.append((path.name, fig)), real_save(fig, path, **kw)))
    store, qrels = toy["store"], normalise_qrels(toy["qrels"])
    ctx = build_context(store, toy["topics"].astype({"qid": str}), qrels, query_vectors(store, qrels, 3),
                        index_key="toy")
    cfg = small_config(tmp_path)
    assert cfg["exp02"]["pin_oracle"] is True
    exp02.run_noise(ctx, cfg["exp02"], tmp_path / "pinned", random_seed=0, cfg=cfg)
    fig = dict(seen)["noise_sweep_eq3.png"]
    raw_ndcg, pct_ndcg = fig.axes[0], fig.axes[len(fig.axes) // 2]
    assert raw_ndcg.get_ylim() == (0, 1) and pct_ndcg.get_ylim() == (0, 140)  # the shared fixed y ranges
    flat = sorted(float(l.get_ydata()[0]) for l in raw_ndcg.get_lines() if l.get_label().startswith("Oracle"))
    assert flat == pytest.approx([0.311247214969467, 0.387636275214749])  # dl19's pinned BM25 and E5 Oracles, grade=3
    m = pd.read_csv(tmp_path / "pinned/metrics_by_alpha.csv", dtype={"alpha": str})
    c0 = m[(m["name"] == "centroid") & (m["view"] == "grade=3") & (m["alpha"] == "0.0")]["nDCG@10"].mean()
    centroid = next(l for l in pct_ndcg.get_lines() if l.get_label() == "E5_centroid")
    assert centroid.get_ydata()[0] == pytest.approx(100 * c0 / 0.387636275214749)
    # the tables keep the computed Oracle
    j = m[m["name"] == "centroid"].merge(m[m["name"] == "true_query"], on=["view", "alpha", "repeat"], suffixes=("", "_o"))
    assert (j["nDCG@10_pct_true_query"] - 100 * j["nDCG@10"] / j["nDCG@10_o"]).abs().max() < 1e-9
    seen.clear()
    exp02.run_noise(ctx, {**cfg["exp02"], "pin_oracle": False}, tmp_path / "computed", random_seed=0, cfg=cfg)
    fig = dict(seen)["noise_sweep_eq3.png"]
    lines = [l for l in fig.axes[0].get_lines() if l.get_label().startswith("Oracle")]
    assert lines and not any(float(l.get_ydata()[0]) == pytest.approx(0.387636275214749) for l in lines)
