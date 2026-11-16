"""Experiment 1's figures: each run against its own retriever's Oracle, the Oracle fixed
at m = 5 (toggle), the "Oracle" / "m" labels, the six-method E5 figure, and the BM25
binary-matching ablation."""
import runpy
import sys
import types

import pandas as pd
import pyterrier as pt
import pytest

from pec.analysis.oracle import oracle_label, oracle_of, pct_of_oracle
from pec.data.qrels import normalise_qrels

from test_query_formulation import FakeTextRetriever, toy_texts


def draws_table():
    rows = []
    for name, base in (("centroid", 0.3), ("bm25_DaaQ", 0.2), ("true_query", 0.6), ("true_query@bm25", 0.5),
                       ("zero_shot@bm25", 0.25), ("bm25_DaaQ_nobm25", 0.15)):
        for i, size in enumerate(["1", "3", "5"]):
            for r in range(2):
                rows.append({"name": name, "view": "grade=3", "context_size": size, "repeat": r, "n_queries": 10,
                             "nDCG@10": base + 0.01 * i + 0.002 * r})
    return pd.DataFrame(rows)


def test_oracle_of_each_retriever():
    assert oracle_of("centroid") == oracle_of("seed_mmr@0.5") == oracle_of("seed_inv_mmr@0.5") == "true_query"
    assert oracle_of("zero_shot") == "true_query"
    assert oracle_of("bm25_DaaQ") == oracle_of("bm25_centroid_nobm25") == "true_query@bm25"
    assert oracle_of("zero_shot@splade") == "true_query@splade" and oracle_of("true_query@bm25") == "true_query@bm25"
    assert oracle_label({"true_query"}) == "Oracle" and oracle_label({"true_query@bm25"}) == "Oracle (BM25)"
    assert oracle_label({"true_query", "true_query@bm25"}) == "Oracle (own retriever)"


def test_pct_of_own_oracle_per_size_and_fixed_at_5():
    d = draws_table()
    per = pct_of_oracle(d, ["nDCG@10"], "context_size")
    get = lambda df, n, s, r: float(df[(df["name"] == n) & (df["context_size"] == s) & (df["repeat"] == r)]
                                    ["nDCG@10_pct_true_query"].iloc[0])
    val = lambda n, s, r: float(d[(d["name"] == n) & (d["context_size"] == s) & (d["repeat"] == r)]["nDCG@10"].iloc[0])
    assert get(per, "centroid", "1", 0) == pytest.approx(100 * val("centroid", "1", 0) / val("true_query", "1", 0))
    assert get(per, "bm25_DaaQ", "3", 1) == pytest.approx(100 * val("bm25_DaaQ", "3", 1)
                                                              / val("true_query@bm25", "3", 1))
    assert get(per, "bm25_DaaQ_nobm25", "1", 0) == pytest.approx(
        100 * val("bm25_DaaQ_nobm25", "1", 0) / val("true_query@bm25", "1", 0))
    assert get(per, "zero_shot@bm25", "5", 0) == pytest.approx(100 * val("zero_shot@bm25", "5", 0)
                                                               / val("true_query@bm25", "5", 0))
    fixed = pct_of_oracle(d, ["nDCG@10"], "context_size", fixed_at="5")
    e5_at_5 = d[(d["name"] == "true_query") & (d["context_size"] == "5")]["nDCG@10"].mean()
    for size in ("1", "3", "5"):  # every size against the same Oracle value
        assert get(fixed, "centroid", size, 0) == pytest.approx(100 * val("centroid", size, 0) / e5_at_5)
    assert fixed[(fixed["name"] == "true_query") & (fixed["context_size"] == "5")]["nDCG@10_pct_true_query"].mean() \
        == pytest.approx(100)
    assert len(per) == len(d) == len(fixed) and list(per["name"]) == list(d["name"])
    # no results at the fixed size: falls back to the Oracle at each size
    back = pct_of_oracle(d[d["context_size"] != "5"], ["nDCG@10"], "context_size", fixed_at="5")
    assert get(back, "centroid", "1", 0) == pytest.approx(get(per, "centroid", "1", 0))


def test_figure_summary_fixed_oracle_toggle():
    from pec.evaluate import SweepFigureStyle, figure_summary

    d = draws_table()
    fixed = figure_summary(d, ["nDCG@10"], style=SweepFigureStyle("5", "Oracle", "m"))
    per = figure_summary(d, ["nDCG@10"], style=SweepFigureStyle(None, "Oracle", "m"))
    pick = lambda s, n, k: float(s[(s["name"] == n) & (s["context_size"] == k)]["nDCG@10_pct_true_query_mean"].iloc[0])
    assert pick(fixed, "true_query", "5") == pytest.approx(100)
    assert pick(fixed, "true_query", "1") < 100 and pick(per, "true_query", "1") == pytest.approx(100)
    assert pick(fixed, "centroid", "1") != pytest.approx(pick(per, "centroid", "1"))


def test_exp01_figures_oracle_m_six_e5_lines_and_bm25_ablation(toy, tmp_path, monkeypatch):
    import pec.analysis.plots as plots
    import pec.experiments.exp01 as exp01
    import pec.text_retrievers as tr
    from pec.config import load_config
    from pec.text_retrievers import ResidualTextRetriever

    store, qrels = toy["store"], normalise_qrels(toy["qrels"])
    texts = toy_texts(toy)
    k1s = []

    def weighted(inp):
        rows = [{"qid": q, "docno": d, "score": sum(w for t, w in toks.items() if t in set(txt.split()))}
                for q, toks in zip(inp["qid"], inp["query_toks"]) for d, txt in texts.items()]
        return pt.model.add_ranks(pd.DataFrame(rows).sort_values(["qid", "score"], ascending=[True, False]))

    monkeypatch.setattr(exp01, "bm25_centroid_parts",
                        lambda cfg, ds, depth, k1=None: k1s.append(("centroid", k1)) or (str.split, pt.apply.generic(weighted)))
    monkeypatch.setattr(exp01, "bm25_binary_text_retriever",
                        lambda cfg, ds, n, depth: k1s.append(("rrf", 0.0)) or ResidualTextRetriever(
                            FakeTextRetriever(texts, depth), n))
    monkeypatch.setattr(tr, "build_retrievers", lambda cfg, ds, names, num_results=1000, depth=None, *a, **kw: {
        n: ResidualTextRetriever(FakeTextRetriever(texts, depth or num_results), num_results) for n in names})
    figs = {}
    real = plots.save
    monkeypatch.setattr(plots, "save", lambda fig, path, **kw: (figs.__setitem__(path.name, fig), real(fig, path, **kw)))
    monkeypatch.setattr(plots, "ORACLE_LINES", True)  # the Oracle lines are off by default

    ctx = exp01.build_context(store, toy["topics"].astype({"qid": str}), qrels, None, texts, index_key="toy")
    from test_datasets_mmr import query_vectors
    ctx.query_vecs = query_vectors(store, qrels, 3)
    cfg = exp01.dataset_config(load_config("configs/exp01_dl19.yaml"), "dl19")
    c = cfg["exp01"]["baseline"]
    assert c["fixed_oracle"] is True and c["oracle_context_size"] == 5 and c["bm25_binary_ablation"] is False
    c.update(num_results=40, recall_ks=[10, 40], context_sizes=[1, 3, 5], repeats=2, mmr={"lambdas": []},
             inv_mmr={"lambdas": []}, bm25_binary_ablation=True)
    c["bm25_rrf"]["depth"] = 40
    c["query_formulation"]["methods"] = []
    exp01.run_parts(ctx, cfg, tmp_path, parts=["baseline"])

    m = pd.read_csv(tmp_path / "baseline/metrics_by_context_size.csv", dtype={"context_size": str})
    assert {"bm25_DaaQ_nobm25", "bm25_centroid_nobm25", "true_query@bm25"} <= set(m["name"])
    assert ("centroid", 0.0) in k1s and ("rrf", 0.0) in k1s and ("centroid", None) in k1s  # k1 = 0 for the ablations
    main, bm25 = figs["context_sweep_eq3.png"], figs["context_sweep_bm25_eq3.png"]
    legend = lambda fig: {t.get_text() for ax in fig.axes if ax.get_legend() for t in ax.get_legend().get_texts()}
    assert {"E5_centroid", "seed_rrf", "Oracle"} <= legend(main) and "true_query" not in legend(main)
    assert {ax.get_xlabel() for ax in main.axes} == {"context size m"}
    # no titles by default (exp01.plot_titles: false): the y axis names the metric instead
    assert {ax.get_ylabel() for ax in main.axes} == {"nDCG@10 (% of Oracle)", "R@10 (% of Oracle)", "R@40 (% of Oracle)"}
    assert main._suptitle.get_text() == "" and {ax.get_title() for ax in main.axes} == {""}
    assert all(ax.get_ylim() == (0, 140) for ax in main.axes)  # y_limits.pct of configs/plot_results.yaml
    overview = figs["context_sweep_overview_ndcg10_eq3.png"]
    assert all(ax.get_ylim() == (0, 140) for ax in overview.axes if ax.get_lines())
    assert overview._suptitle.get_text() == "" and {ax.get_title() for ax in overview.axes} == {""}
    headings = {ax.get_legend().get_title().get_text() for ax in overview.axes if ax.get_legend()}
    assert {"E5 methods", "BM25 methods"} <= headings  # the method family heads the legend instead
    # the nDCG@10 panel divides by the pinned E5 Oracle (0.3876... for dl19, grade=3)
    centroid = next(l for l in main.axes[0].get_lines() if l.get_label() == "E5_centroid")
    at5 = m[(m["name"] == "centroid") & (m["view"] == "grade=3") & (m["context_size"] == "5")]["nDCG@10"].mean()
    assert centroid.get_ydata()[-1] == pytest.approx(100 * at5 / 0.387636275214749)

    figs.clear()  # plot_titles: true brings the titles back
    cfg["exp01"]["plot_titles"] = True
    exp01.run_parts(ctx, cfg, tmp_path / "titled", parts=["baseline"])
    main = figs["context_sweep_eq3.png"]
    assert "pinned Oracle" in main._suptitle.get_text()  # configs/oracle_values.yaml pins dl19's nDCG@10 Oracles
    assert {ax.get_ylabel() for ax in main.axes} == {"% of Oracle"} and {ax.get_title() for ax in main.axes} != {""}
    assert {"bm25_DaaQ", "bm25_centroid", "bm25_DaaQ_nobm25", "bm25_centroid_nobm25",
            "Oracle (BM25)"} <= legend(bm25)
    assert {ax.get_ylabel() for ax in bm25.axes} == {f"{m} (% of Oracle (BM25))" for m in ("nDCG@10", "R@10", "R@40")}


def test_six_e5_lines_in_one_figure(tmp_path, monkeypatch):
    import pec.analysis.plots as plots
    from pec.evaluate import SweepFigureStyle, figure_summary, plot_context_sweep

    six = ["centroid", "seed_rrf", "zero_shot", "static_icl", "sliding_window", "doc2query"]
    rows = [{"name": n, "view": "grade=3", "context_size": s, "repeat": r, "n_queries": 10,
             "nDCG@10": 0.1 * (i + 1) + 0.01 * j + 0.001 * r}
            for i, n in enumerate(six + ["true_query"]) for j, s in enumerate(["1", "5"]) for r in range(2)]
    figs = []
    real = plots.save
    monkeypatch.setattr(plots, "save", lambda fig, path, **kw: (figs.append(fig), real(fig, path, **kw)))
    style = SweepFigureStyle("5", "Oracle", "m")
    plot_context_sweep(figure_summary(pd.DataFrame(rows), ["nDCG@10"], style=style), ["nDCG@10"], six,
                       tmp_path / "s", style=style)
    ax = figs[0].axes[0]
    shown = ["E5_centroid", "seed_rrf", "QF-Zero", "QF-Fixed", "QF-SW", "QF-D2Q"]  # the figures' names
    drawn = [l for l in ax.get_lines() if l.get_label() in shown]
    assert [l.get_label() for l in drawn] == shown
    assert len({l.get_color() for l in drawn}) == 6 and len({l.get_marker() for l in drawn}) == 6


def test_exp01_cli_oracle_and_ablation_toggles(monkeypatch):
    import pec.experiments.exp01 as exp01

    seen = {}
    monkeypatch.setattr(exp01, "run_exp01", lambda cfg, parts=None, datasets=None: seen.update(cfg=cfg) or {})
    monkeypatch.setattr(sys, "argv", ["exp01_dl19.py", "--bm25-ablation", "--no-fixed-oracle"])
    runpy.run_path("scripts/exp01_dl19.py", run_name="__main__")
    b = seen["cfg"]["exp01"]["baseline"]
    assert b["bm25_binary_ablation"] is True and b["fixed_oracle"] is False
    assert seen["cfg"]["exp01"]["plot_titles"] is False  # the config default
    monkeypatch.setattr(sys, "argv", ["exp01_dl19.py", "--titles"])
    runpy.run_path("scripts/exp01_dl19.py", run_name="__main__")
    assert seen["cfg"]["exp01"]["plot_titles"] is True
    monkeypatch.setattr(sys, "argv", ["exp01_dl19.py", "--bm25-ablation", "--no-fixed-oracle"])
    runpy.run_path("scripts/exp01_dl19.py", run_name="__main__")
    b = seen["cfg"]["exp01"]["baseline"]
    style = exp01.figure_style(b)
    assert style.oracle_size is None and style.label == "Oracle" and style.size_symbol == "m"
    assert exp01.figure_style({"oracle_context_size": 3}).oracle_size == "3"


def test_pisa_bm25_k1_and_weighting(monkeypatch):
    import pec.experiments.exp01 as exp01
    import pec.text_retrievers as tr

    calls = []
    fake_index = types.SimpleNamespace(bm25=lambda **kw: calls.append(kw) or "retriever")
    monkeypatch.setattr(tr, "_pisa_index", lambda path, what, **kw: fake_index)
    cfg = {"datasets": {"dl19": {"index": "msmarco_v1"}}, "text_retrievers": {"msmarco_v1": {"bm25": "/x"}}}
    exp01.pisa_bm25(cfg, "dl19", 1005)
    exp01.pisa_bm25(cfg, "dl19", 1005, k1=0.0, weighted=True)
    assert calls[0] == {"num_results": 1005, "query_weighted": None}  # PISA's defaults (k1 0.9, b 0.4)
    assert calls[1] == {"num_results": 1005, "query_weighted": True, "k1": 0.0}


PINS = {"nDCG@10": {"true_query": {"grade=3": 0.5}, "true_query@bm25": {"grade=3": 0.25}}}


def test_pinned_oracle_values_replace_the_computed_ones(caplog):
    from pec.analysis.oracle import pinned_value, pinned_values, warn_unused_pins

    cfg = {"oracle_values": {"nDCG@10": {"dl19": PINS["nDCG@10"], "dl20": {"true_query": {"grade=3": 0.7}}}}}
    assert pinned_values(cfg, "dl19") == PINS and pinned_values(cfg, "trec_covid") == {}
    assert pinned_value(PINS, "nDCG@10", "true_query", "grade=3") == 0.5
    assert pinned_value(PINS, "R@10", "true_query", "grade=3") is None
    d = draws_table()
    for fixed in (None, "5"):
        out = pct_of_oracle(d, ["nDCG@10"], "context_size", fixed_at=fixed, pinned=PINS)
        row = lambda n: out[(out["name"] == n) & (out["context_size"] == "1") & (out["repeat"] == 0)].iloc[0]
        assert row("centroid")["nDCG@10_pct_true_query"] == pytest.approx(100 * row("centroid")["nDCG@10"] / 0.5)
        assert row("bm25_DaaQ")["nDCG@10_pct_true_query"] == pytest.approx(100 * row("bm25_DaaQ")["nDCG@10"] / 0.25)
    # a view without a pin keeps the computed Oracle; an Oracle run that was never stored can still be pinned
    other = d.assign(view="grade>=2")
    out = pct_of_oracle(pd.concat([d, other]), ["nDCG@10"], "context_size", pinned=PINS)
    ge2 = out[(out["view"] == "grade>=2") & (out["name"] == "centroid")]
    assert ge2["nDCG@10_pct_true_query"].notna().all() and (ge2["nDCG@10_pct_true_query"] != 200 * ge2["nDCG@10"]).all()
    no_oracle = d[d["name"] == "centroid"]
    assert pct_of_oracle(no_oracle, ["nDCG@10"], "context_size", pinned=PINS)["nDCG@10_pct_true_query"].notna().all()
    with caplog.at_level("WARNING"):
        warn_unused_pins({"nDCG@10": {"true_query": {"grade=9": 1.0}}}, d, "dl19")
    assert "matches no stored result" in caplog.text


def test_exp01_pin_toggle_and_comparison(toy, tmp_path):
    from pec.config import load_config
    from pec.experiments.exp01 import figure_style, write_comparison

    cfg = load_config("configs/exp01_dl19.yaml")
    assert cfg["exp01"]["baseline"]["pin_oracle"] is True
    assert cfg["oracle_values"]["nDCG@10"]["dl19"]["true_query"]["grade=3"] == pytest.approx(0.387636275214749)
    assert cfg["oracle_values"]["nDCG@10"]["trec_covid"]["true_query@bm25"]["grade=2"] == pytest.approx(0.550955655824402)
    assert figure_style({"pin_oracle": False}, PINS).pinned is None
    assert figure_style({}, PINS).pinned == PINS
    # the dataset comparison divides by the pinned E5 Oracle of each dataset's target view
    cfg["output_dir"] = str(tmp_path)
    for ds, target in (("dl19", "grade=3"), ("dl20", "grade=3")):
        base = tmp_path / f"exp01_{ds}/baseline"
        base.mkdir(parents=True)
        pd.DataFrame({"name": "centroid", "view": "grade=3", "metric": "nDCG@10", "best_k": "5"}, index=[0]).to_csv(
            base / "best_context_size.csv", index=False)
        pd.DataFrame([{"name": n, "view": target, "context_size": k, "nDCG@10_mean": v, "nDCG@10_std": 0.01,
                       "nDCG@10_pct_true_query_mean": 1.0, "nDCG@10_pct_true_query_std": 0.0}
                      for n, v in (("centroid", 0.2), ("true_query", 0.4)) for k in ("1", "5")]).to_csv(
            base / "summary_by_context_size.csv", index=False)
    write_comparison(cfg, ["dl19", "dl20"], tmp_path / "comparison")
    curves = pd.read_csv(tmp_path / "comparison/centroid_ndcg10_by_dataset.csv")
    pins = {"dl19": 0.387636275214749, "dl20": 0.543435501826611}
    for ds, pin in pins.items():
        c = curves[curves["dataset"] == ds]
        assert (c["nDCG@10_pct_true_query_mean"] - 100 * 0.2 / pin).abs().max() < 1e-9 and c["oracle_pinned"].all()


def test_titles_off_keeps_query_labels(tmp_path):
    import matplotlib.pyplot as plt

    from pec.analysis.plots import figure_titles, keep_title, save

    fig, (a, b) = plt.subplots(1, 2)
    a.set_title("a title")
    b.set_title("q123 (n=4)")
    keep_title(b)
    fig.suptitle("figure title")
    with figure_titles(False):
        save(fig, tmp_path / "f.png")
    assert fig._suptitle.get_text() == "" and a.get_title() == "" and b.get_title() == "q123 (n=4)"
    fig, ax = plt.subplots()
    ax.set_title("kept")
    save(fig, tmp_path / "g.png")  # titles stay outside figure_titles(False)
    assert ax.get_title() == "kept"
