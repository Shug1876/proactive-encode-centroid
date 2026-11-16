"""scripts/plot_results.py: figures from stored result tables only, with the spread over
repeats, on synthetic tables shaped like the experiments' outputs."""
import json
import runpy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pec.analysis.result_plots import (Options, describe, find_sources, model_groups, plot_results, select_names,
                                       spread)

DL_VIEWS, COVID_VIEWS = ["grade>=2", "grade=3"], ["grade=2"]


def table(names, views, conditions, col, repeats=5, seed=0, extra=None):
    rng = np.random.default_rng(seed)
    rows = []
    for name in names:
        for view in views:
            for i, c in enumerate(conditions):
                for r in range(1 if c == "all" else repeats):
                    base = 0.8 if name.startswith("true_query") else 0.3 + 0.05 * i
                    rows.append({"name": name, "view": view, col: c, "repeat": r, "n_queries": 40,
                                 "nDCG@10": base + rng.normal(0, 0.03), "R@100": base + 0.1 + rng.normal(0, 0.03),
                                 "nDCG@10_pct_true_query": -1.0, **(extra or {})})  # stale %: recomputed
    return pd.DataFrame(rows)


@pytest.fixture
def stored(tmp_path):
    root = tmp_path / "outputs"
    exp01 = ["centroid", "seed_fusion", "seed_rrf", "bm25_DaaQ", "seed_mmr@0.5", "seed_mmr@0.9", "zero_shot",
             "true_query", "true_query@bm25"]
    for ds, views, seed in (("dl19", DL_VIEWS, 0), ("trec_covid", COVID_VIEWS, 1)):
        d = root / f"exp01_{ds}/baseline"
        d.mkdir(parents=True)
        table(exp01, views, ["1", "3", "5", "all"], "context_size", seed=seed).to_csv(
            d / "metrics_by_context_size.csv", index=False)
        d = root / f"exp02_{ds}"
        d.mkdir(parents=True)
        alphas = ["0.0", "0.2", "0.4", "0.6", "0.8"]
        table(["centroid", "true_query"], views, alphas, "alpha", seed=seed).to_csv(
            d / "metrics_by_alpha.csv", index=False)
        (d / "sweep_settings.json").write_text(json.dumps({"alphas": alphas, "context_size": 10}))
    (root / "exp01_comparison").mkdir()  # no per-draw table: not a dataset
    return root


def test_spread_and_selection():
    v = np.array([0.1, 0.2, 0.3, 0.6])
    m, lo, hi = spread(v, "std")
    assert m == pytest.approx(0.3) and hi - m == pytest.approx(v.std(ddof=1))
    _, slo, shi = spread(v, "sem")
    _, clo, chi = spread(v, "ci95")
    assert shi - slo == pytest.approx(2 * v.std(ddof=1) / 2) and chi - clo > shi - slo
    assert spread(v, "minmax")[1:] == (0.1, 0.6) and spread(v, "none")[1:] == (m, m)
    assert spread(np.array([0.4]), "std") == (0.4, 0.4, 0.4)
    names = ["centroid", "seed_rrf", "seed_mmr@0.5", "seed_mmr@0.9", "true_query"]
    assert select_names(names, ["seed_mmr@*", "centroid"]) == ["seed_mmr@0.5", "seed_mmr@0.9", "centroid"]
    assert select_names(names, None, ["seed_*"]) == ["centroid", "true_query"]
    many = [f"m{i}" for i in range(10)]
    assert model_groups(many, "true_query", patterns=["m*"]) == {"selected": many[:8], "selected_2": many[8:]}
    groups = model_groups(names + ["my_run"], "true_query")
    assert groups["main"] == ["centroid", "seed_rrf"] and groups["other"] == ["my_run"]
    assert list(model_groups(names, "true_query", groups=["mmr"])) == ["mmr"]


def test_every_plot_with_the_spread_over_repeats(stored, tmp_path):
    out = tmp_path / "plots"
    written = plot_results(stored, out, cfg=None, all_models=True)
    names = {p.relative_to(out).as_posix() for p in written}
    for f in ["exp01_dl19/sweep_main_eq3.png", "exp01_dl19/sweep_mmr_ge2.png", "exp01_dl19/sweep_bm25_eq3.png",
              "exp01_trec_covid/sweep_main_eq2.png", "exp02_dl19/sweep_main_eq3.png",
              "exp01_comparison/datasets_centroid.png",
              "exp02_comparison/datasets_centroid.png"]:
        assert f in names, f
        assert (out / f).with_suffix(".csv").exists()
    assert not any("seed_fusion" in n for n in names)

    raw = pd.read_csv(stored / "exp01_dl19/baseline/metrics_by_context_size.csv", dtype={"context_size": str})
    stats = pd.read_csv(out / "exp01_dl19/sweep_main_eq3.csv", dtype={"context_size": str})
    assert "seed_fusion" not in set(stats["name"]) and "all" not in set(stats["context_size"])
    row = stats[(stats["name"] == "centroid") & (stats["context_size"] == "3") & (stats["metric"] == "nDCG@10")
                & (stats["value"] == "raw")].iloc[0]
    reps = raw[(raw["name"] == "centroid") & (raw["view"] == "grade=3") & (raw["context_size"] == "3")]["nDCG@10"]
    assert row["repeats"] == 5 and row["nDCG@10_mean"] == pytest.approx(reps.mean())
    assert row["nDCG@10_high"] - row["nDCG@10_mean"] == pytest.approx(reps.std(ddof=1))
    pct = stats[(stats["name"] == "centroid") & (stats["metric"] == "nDCG@10") & (stats["value"] == "pct")]
    assert (pct["nDCG@10_pct_true_query_mean"] > 0).all()  # recomputed, not the stale -1
    alphas = pd.read_csv(out / "exp02_dl19/sweep_main_eq3.csv", dtype={"alpha": str})
    assert list(dict.fromkeys(alphas["alpha"])) == ["0.0", "0.2", "0.4", "0.6", "0.8"]
    comp = pd.read_csv(out / "exp01_comparison/datasets_centroid.csv")
    assert set(zip(comp["dataset"], comp["view"])) == {("dl19", "grade=3"), ("trec_covid", "grade=2")}


def test_default_shows_only_the_best_methods(stored, tmp_path):
    out = tmp_path / "plots"
    written = plot_results(stored, out)
    names = {p.relative_to(out).as_posix() for p in written}
    assert {"exp01_dl19/sweep_main_ge2.png", "exp01_dl19/sweep_main_eq3.png", "exp02_dl19/sweep_main_eq3.png",
            "exp01_comparison/datasets_centroid.png",
            "exp01_comparison/datasets_seed_rrf.png", "exp01_comparison/datasets_zero_shot.png"} <= names
    assert all("/sweep_main_" in n or "/datasets_" in n for n in names)  # no mmr / bm25 / other figures
    shown = {}
    for p in written:
        shown.setdefault(p.parent.name.split("_")[0], set()).update(pd.read_csv(p.with_suffix(".csv"))["name"])
    assert shown["exp01"] == {"centroid", "seed_rrf", "zero_shot", "true_query"}  # every E5 method stored
    assert shown["exp02"] == {"centroid", "true_query"}
    # stated explicitly otherwise: --models, --groups
    picked = plot_results(stored, tmp_path / "picked", experiments=["exp01"], datasets=["dl19"], plots=["sweep"],
                          models=["seed_mmr@0.5"], opt=Options(views=["eq3"]))
    assert set(pd.read_csv(picked[0].with_suffix(".csv"))["name"]) == {"seed_mmr@0.5", "true_query"}
    grouped = plot_results(stored, tmp_path / "grouped", experiments=["exp01"], datasets=["dl19"], plots=["sweep"],
                           groups=["mmr"], opt=Options(views=["eq3"]))
    assert [p.name for p in grouped] == ["sweep_mmr_eq3.png"]


def test_toggles(stored, tmp_path):
    out = tmp_path / "plots"
    opt = Options(metrics=["R@100"], values=("pct",), views=["eq3"], variance="minmax", show_repeats=True,
                  conditions=["1", "5"])
    written = plot_results(stored, out, experiments=["exp01"], datasets=["dl19"], plots=["sweep"],
                           models=["seed_mmr@*", "centroid"], opt=opt)
    assert [p.name for p in written] == ["sweep_selected_eq3.png"]
    stats = pd.read_csv(written[0].with_suffix(".csv"), dtype={"context_size": str})
    assert set(stats["name"]) == {"seed_mmr@0.5", "seed_mmr@0.9", "centroid"}
    assert set(stats["metric"]) == {"R@100"} and set(stats["value"]) == {"pct"} and set(stats["view"]) == {"grade=3"}
    assert set(stats["context_size"]) == {"1", "5"}
    raw = pd.read_csv(stored / "exp01_dl19/baseline/metrics_by_context_size.csv", dtype={"context_size": str})
    assert stats["R@100_pct_true_query_low"].le(stats["R@100_pct_true_query_mean"]).all()
    assert len(raw)  # the stored tables are only read
    assert plot_results(stored, out, experiments=["exp02"], plots=["datasets"], datasets=["dl19"]) == []  # one dataset
    with_all = plot_results(stored, out, experiments=["exp01"], datasets=["dl19"], plots=["sweep"], groups=["main"],
                            opt=Options(include_all=True, views=["ge2"]))
    assert "all" in set(pd.read_csv(with_all[0].with_suffix(".csv"), dtype={"context_size": str})["context_size"])


def test_describe_and_cli(stored, tmp_path, monkeypatch, capsys):
    d = describe(stored)
    assert set(zip(d["experiment"], d["dataset"], d["plot"])) == {
        ("exp01", "dl19", "sweep"), ("exp01", "trec_covid", "sweep"), ("exp02", "dl19", "sweep"),
        ("exp02", "trec_covid", "sweep")}
    assert "seed_fusion" not in " ".join(d["models"]) and set(d["repeats"]) == {5}
    assert len(find_sources(stored, ["exp02"], ["trec_covid"])) == 1
    with pytest.raises(ValueError):
        find_sources(stored, ["exp03"])

    out = tmp_path / "cli"
    monkeypatch.setattr(sys, "argv", ["plot_results.py", "--root", str(stored), "--out", str(out), "--experiments",
                                      "exp02", "--plots", "sweep", "datasets", "--variance", "ci95", "--format", "pdf"])
    runpy.run_path("scripts/plot_results.py", run_name="__main__")
    assert (out / "exp02_dl19/sweep_main_eq3.pdf").exists() and (out / "exp02_comparison/datasets_centroid.pdf").exists()
    assert not (out / "exp01_dl19").exists()
    monkeypatch.setattr(sys, "argv", ["plot_results.py", "--root", str(stored), "--out", str(out), "--experiments",
                                      "exp01", "--datasets", "dl19", "--plots", "sweep", "--all-models"])
    runpy.run_path("scripts/plot_results.py", run_name="__main__")
    assert (out / "exp01_dl19/sweep_mmr_eq3.pdf").exists() and (out / "exp01_dl19/sweep_bm25_eq3.pdf").exists()  # pdf: the default
    monkeypatch.setattr(sys, "argv", ["plot_results.py", "--root", str(stored), "--list"])
    runpy.run_path("scripts/plot_results.py", run_name="__main__")
    assert "sweep" in capsys.readouterr().out


def test_exp02_comparison_also_writes_every_panel(stored, tmp_path):
    out = tmp_path / "plots"
    written = plot_results(stored, out, plots=["datasets"], models=["centroid"])
    names = {p.relative_to(out).as_posix() for p in written}
    exp02 = sorted(n for n in names if n.startswith("exp02_comparison/"))
    panels = [f"exp02_comparison/datasets_centroid_{m}_{v}.png" for m in ("ndcg10", "r100") for v in ("raw", "pct")]
    assert exp02 == sorted(["exp02_comparison/datasets_centroid.png"] + panels)
    for f in panels:  # one metric and value each, every dataset as a line, with its numbers
        csv = pd.read_csv((out / f).with_suffix(".csv"))
        assert csv["metric"].nunique() == 1 and csv["value"].nunique() == 1
        assert set(csv["dataset"]) == {"dl19", "trec_covid"}
    combined = pd.read_csv(out / "exp02_comparison/datasets_centroid.csv")
    panel = pd.read_csv(out / "exp02_comparison/datasets_centroid_ndcg10_raw.csv")
    same = combined[(combined["metric"] == "nDCG@10") & (combined["value"] == "raw")].reset_index(drop=True)
    assert panel.equals(same[panel.columns])  # the same numbers as that panel of the combined figure
    assert not any(n.startswith("exp01_comparison/datasets_centroid_") for n in names)  # exp01: combined figure only
    only = plot_results(stored, tmp_path / "pct", experiments=["exp02"], plots=["datasets"], models=["centroid"],
                        opt=Options(metrics=["nDCG@10"], values=("pct",)))
    assert sorted(p.name for p in only) == ["datasets_centroid.png", "datasets_centroid_ndcg10_pct.png"]


def test_exp02_figures_untitled_with_oracle_unless_asked(stored, tmp_path, monkeypatch):
    import pec.analysis.result_plots as rp

    figs = {}
    real = rp.save
    monkeypatch.setattr(rp, "save", lambda fig, path, **kw: (figs.__setitem__(f"{path.parent.name}/{path.name}", fig),
                                                              real(fig, path, **kw)))

    def texts(fig):
        legend = {t.get_text() for ax in fig.axes if ax.get_legend() for t in ax.get_legend().get_texts()}
        return (fig._suptitle.get_text() if fig._suptitle else "", {ax.get_title() for ax in fig.axes},
                {ax.get_xlabel() for ax in fig.axes}, {ax.get_ylabel() for ax in fig.axes}, legend)

    plot_results(stored, tmp_path / "a", experiments=["exp02", "exp01"], datasets=["dl19", "trec_covid"],
                 plots=["sweep", "datasets"], models=["centroid"])
    for name in ("exp02_comparison/datasets_centroid.png", "exp02_comparison/datasets_centroid_ndcg10_pct.png",
                 "exp02_dl19/sweep_selected_eq3.png"):
        sup, titles, xl, yl, legend = texts(figs[name])
        assert sup == "" and titles == {""}, name
        assert xl == {"α"}  # main 28d4fdd: the x label is just α
        assert "Oracle" in legend and "true_query" not in legend
    sup, titles, xl, _, legend = texts(figs["exp01_comparison/datasets_centroid.png"])
    assert sup == "" and titles == {""} and "Oracle" in legend and xl == {"context size m"}  # exp01: untitled too
    figs.clear()
    plot_results(stored, tmp_path / "b", experiments=["exp01"], datasets=["dl19"], plots=["sweep"],
                 models=["centroid"], opt=Options(views=["eq3"]))
    sup, titles, xl, yl, legend = texts(next(iter(figs.values())))
    assert sup == "" and titles == {""} and "Oracle" in legend and "true_query" not in legend
    assert xl == {"context size m"}  # (main 28d4fdd: the panels have no y label)
    figs.clear()
    plot_results(stored, tmp_path / "bt", experiments=["exp01"], datasets=["dl19"], plots=["sweep"],
                 models=["centroid"], opt=Options(views=["eq3"], titles=True))
    sup, _, _, yl, _ = texts(next(iter(figs.values())))
    assert "Oracle at m = 5" in sup
    # toggles: titles on for exp02, another label
    figs.clear()
    plot_results(stored, tmp_path / "c", experiments=["exp02"], datasets=["dl19"], plots=["sweep"],
                 models=["centroid"], opt=Options(views=["eq3"], titles=True, reference_label="Upper bound"))
    sup, _, _, yl, legend = texts(next(iter(figs.values())))
    assert sup and "Upper bound" in legend
    assert [o for o in rp.condition_order(rp.find_sources(stored, ["exp02"], ["dl19"])[0],
                                          ["0.8", "0.0", "0.4"], keep=["0", "0.40"])] == ["0.0", "0.4"]


def test_exp01_fixed_oracle_own_retriever_and_oracle_lines(stored, tmp_path, monkeypatch):
    import pec.analysis.result_plots as rp

    figs = {}
    real = rp.save
    monkeypatch.setattr(rp, "save", lambda fig, path, **kw: (figs.__setitem__(path.name, fig), real(fig, path, **kw)))
    raw = pd.read_csv(stored / "exp01_dl19/baseline/metrics_by_context_size.csv", dtype={"context_size": str})
    raw = raw[raw["view"] == "grade=3"]
    mean = lambda n, k: raw[(raw["name"] == n) & (raw["context_size"] == k)]["nDCG@10"].mean()

    w = plot_results(stored, tmp_path / "f", experiments=["exp01"], datasets=["dl19"], plots=["sweep"],
                     models=["centroid", "bm25_DaaQ"], opt=Options(views=["eq3"], metrics=["nDCG@10"]))
    stats = pd.read_csv(w[0].with_suffix(".csv"), dtype={"context_size": str})
    pct = stats[stats["value"] == "pct"].set_index(["name", "context_size"])["nDCG@10_pct_true_query_mean"]
    # fixed (default for exp01): every size against the Oracle at m = 5, each run against its own retriever's
    assert pct[("centroid", "1")] == pytest.approx(100 * mean("centroid", "1") / mean("true_query", "5"))
    assert pct[("bm25_DaaQ", "3")] == pytest.approx(100 * mean("bm25_DaaQ", "3") / mean("true_query@bm25", "5"))
    flat = stats[(stats["value"] == "raw") & stats["name"].isin(["true_query", "true_query@bm25"])]
    assert set(flat["fixed_oracle"].dropna()) == {True} and len(flat) == 2  # one flat line per Oracle
    fig = next(iter(figs.values()))
    legend = {t.get_text() for ax in fig.axes if ax.get_legend() for t in ax.get_legend().get_texts()}
    assert {"Oracle (E5)", "Oracle (BM25)", "Oracle (own retriever)"} <= legend
    # toggle off: each size against the Oracle at that size
    w = plot_results(stored, tmp_path / "g", experiments=["exp01"], datasets=["dl19"], plots=["sweep"],
                     models=["centroid"], opt=Options(views=["eq3"], metrics=["nDCG@10"], fixed_oracle=False))
    stats = pd.read_csv(w[0].with_suffix(".csv"), dtype={"context_size": str})
    row = stats[(stats["value"] == "pct") & (stats["name"] == "centroid") & (stats["context_size"] == "1")].iloc[0]
    per = raw[raw["context_size"] == "1"].pivot_table(index="repeat", columns="name", values="nDCG@10")
    assert row["nDCG@10_pct_true_query_mean"] == pytest.approx((100 * per["centroid"] / per["true_query"]).mean())
    assert "fixed_oracle" not in stats or stats["fixed_oracle"].isna().all()


def test_plot_results_cli_oracle_toggles(stored, tmp_path, monkeypatch):
    import pec.analysis.result_plots as rp

    seen = {}
    monkeypatch.setattr(rp, "plot_results", lambda *a, **kw: seen.update(opt=a[8] if len(a) > 8 else kw["opt"]) or [])
    monkeypatch.setattr(sys, "argv", ["plot_results.py", "--root", str(stored), "--no-fixed-oracle",
                                      "--oracle-size", "3"])
    runpy.run_path("scripts/plot_results.py", run_name="__main__")
    assert seen["opt"].fixed_oracle is False and seen["opt"].oracle_size == "3"


def test_plot_results_pinned_oracle(stored, tmp_path, monkeypatch):
    pins = {"nDCG@10": {"dl19": {"true_query": {"grade=3": 0.5}, "true_query@bm25": {"grade=3": 0.25}}}}
    raw = pd.read_csv(stored / "exp01_dl19/baseline/metrics_by_context_size.csv", dtype={"context_size": str})
    raw = raw[raw["view"] == "grade=3"]
    w = plot_results(stored, tmp_path / "p", experiments=["exp01"], datasets=["dl19"], plots=["sweep"],
                     models=["centroid", "bm25_DaaQ"],
                     opt=Options(views=["eq3"], metrics=["nDCG@10", "R@100"], pinned=pins))
    stats = pd.read_csv(w[0].with_suffix(".csv"), dtype={"context_size": str})
    pct = stats[(stats["value"] == "pct") & (stats["metric"] == "nDCG@10")].set_index(["name", "context_size"])
    mean = lambda n, k: raw[(raw["name"] == n) & (raw["context_size"] == k)]["nDCG@10"].mean()
    assert pct.loc[("centroid", "3"), "nDCG@10_pct_true_query_mean"] == pytest.approx(100 * mean("centroid", "3") / 0.5)
    assert pct.loc[("bm25_DaaQ", "1"), "nDCG@10_pct_true_query_mean"] == \
        pytest.approx(100 * mean("bm25_DaaQ", "1") / 0.25)
    lines = stats[stats["pinned_oracle"].fillna(False).astype(bool)] if "pinned_oracle" in stats else stats.iloc[:0]
    assert set(zip(lines["name"], lines["nDCG@10_mean"])) == {("true_query", 0.5), ("true_query@bm25", 0.25)}
    # R@100 has no pin: still the computed Oracle (fixed at m = 5)
    r = stats[(stats["value"] == "pct") & (stats["metric"] == "R@100")].set_index(["name", "context_size"])
    at5 = raw[(raw["name"] == "true_query") & (raw["context_size"] == "5")]["R@100"].mean()
    rmean = raw[(raw["name"] == "centroid") & (raw["context_size"] == "3")]["R@100"].mean()
    assert r.loc[("centroid", "3"), "R@100_pct_true_query_mean"] == pytest.approx(100 * rmean / at5)

    import pec.analysis.result_plots as rp
    seen = {}
    monkeypatch.setattr(rp, "plot_results", lambda *a, **kw: seen.update(opt=a[8]) or [])
    monkeypatch.setattr(sys, "argv", ["plot_results.py", "--root", str(stored)])
    runpy.run_path("scripts/plot_results.py", run_name="__main__")
    assert seen["opt"].pinned["nDCG@10"]["dl20"]["true_query"]["grade>=2"] == pytest.approx(0.631565385217391)
    monkeypatch.setattr(sys, "argv", ["plot_results.py", "--root", str(stored), "--no-pinned-oracle"])
    runpy.run_path("scripts/plot_results.py", run_name="__main__")
    assert seen["opt"].pinned is None


def test_runs_config_selects_the_runs(stored, tmp_path, monkeypatch):
    from pec.config import load_config

    runs = load_config("configs/plot_results.yaml")["plot_results"]["runs"]
    assert runs["exp01"] and runs["exp02"]  # the runs chosen in configs/plot_results.yaml
    shown = lambda paths: {n for p in paths for n in pd.read_csv(p.with_suffix(".csv"))["name"]}
    w = plot_results(stored, tmp_path / "a", experiments=["exp01"], datasets=["dl19"], plots=["sweep"],
                     opt=Options(views=["eq3"]), runs={"exp01": ["seed_mmr@*", "bm25_DaaQ"]})
    assert [p.name for p in w] == ["sweep_main_eq3.png"]
    assert shown(w) == {"seed_mmr@0.5", "seed_mmr@0.9", "bm25_DaaQ", "true_query", "true_query@bm25"}
    stats = pd.read_csv(w[0].with_suffix(".csv"))
    assert list(dict.fromkeys(stats[~stats["name"].str.startswith("true_query")]["name"])) == \
        ["seed_mmr@0.5", "seed_mmr@0.9", "bm25_DaaQ"]  # the file's order
    # an experiment the file leaves out keeps the built-in default; --models takes precedence
    w = plot_results(stored, tmp_path / "b", experiments=["exp02"], datasets=["dl19"], plots=["sweep"],
                     opt=Options(views=["eq3"]), runs={"exp01": ["seed_rrf"]})
    assert shown(w) == {"centroid", "true_query"}
    w = plot_results(stored, tmp_path / "c", experiments=["exp01"], datasets=["dl19"], plots=["sweep"],
                     models=["centroid"], opt=Options(views=["eq3"]), runs={"exp01": ["seed_rrf"]})
    assert shown(w) == {"centroid", "true_query"}
    # more than eight runs: main, main_2
    many = [f"m{i}" for i in range(10)]
    assert list(rp_chunks("main", many)) == ["main", "main_2"]
    # the datasets plot follows the file too
    w = plot_results(stored, tmp_path / "d", experiments=["exp01"], plots=["datasets"],
                     runs={"exp01": ["seed_rrf"]})
    assert [p.name for p in w] == ["datasets_seed_rrf.png"]

    import pec.analysis.result_plots as rp
    seen = {}
    monkeypatch.setattr(rp, "plot_results", lambda *a, **kw: seen.update(runs=a[11] if len(a) > 11 else kw.get("runs")) or [])
    monkeypatch.setattr(sys, "argv", ["plot_results.py", "--root", str(stored)])
    runpy.run_path("scripts/plot_results.py", run_name="__main__")
    assert seen["runs"] == runs  # configs/plot_results.yaml by default
    mine = tmp_path / "mine.yaml"
    mine.write_text("plot_results:\n  runs:\n    exp01: [centroid, bm25_centroid]\n")
    monkeypatch.setattr(sys, "argv", ["plot_results.py", "--root", str(stored), "--runs-config", str(mine)])
    runpy.run_path("scripts/plot_results.py", run_name="__main__")
    assert seen["runs"] == {"exp01": ["centroid", "bm25_centroid"]}
    monkeypatch.setattr(sys, "argv", ["plot_results.py", "--root", str(stored), "--no-runs-config"])
    runpy.run_path("scripts/plot_results.py", run_name="__main__")
    assert seen["runs"] is None
    bad = tmp_path / "bad.yaml"
    bad.write_text("plot_results:\n  runs:\n    exp9: [centroid]\n")
    for argv in (["--runs-config", str(bad)], ["--runs-config", str(tmp_path / "missing.yaml")]):
        monkeypatch.setattr(sys, "argv", ["plot_results.py", "--root", str(stored)] + argv)
        with pytest.raises(SystemExit):
            runpy.run_path("scripts/plot_results.py", run_name="__main__")


def rp_chunks(prefix, names):
    from pec.analysis.result_plots import MAX_LINES, _chunks

    return _chunks(prefix, names, MAX_LINES)


def test_every_figure_uses_the_same_fixed_y_ranges(stored, tmp_path, monkeypatch, caplog):
    import pec.analysis.result_plots as rp
    from pec.analysis.plots import parse_y_limits, set_ylim, y_limits
    from pec.config import load_config

    assert load_config("configs/plot_results.yaml")["y_limits"] == {"raw": [0, 1], "pct": [0, 140]}
    for f in ("configs/exp01_dl19.yaml", "configs/exp02_noise.yaml"):
        assert load_config(f)["y_limits"] == {"raw": [0, 1], "pct": [0, 140]}, f  # included by every config
    assert parse_y_limits({"raw": [0, 1]}) == {"raw": (0.0, 1.0)} and parse_y_limits(None) is None
    with pytest.raises(ValueError):
        parse_y_limits({"raw": [1, 0]})
    with pytest.raises(ValueError):
        parse_y_limits({"pcts": [0, 1]})

    figs = []
    real = rp.save
    monkeypatch.setattr(rp, "save", lambda fig, path, **kw: (figs.append(fig), real(fig, path, **kw)))
    limits = {"raw": [0, 1], "pct": [0, 140]}
    plot_results(stored, tmp_path / "a", plots=["sweep", "datasets"], opt=Options(y_limits=limits))
    assert figs
    for fig in figs:
        for ax in fig.axes:
            label = ax.get_ylabel()
            if "%" in label:
                assert ax.get_ylim() == (0, 140), label
            elif label:
                assert ax.get_ylim() == (0, 1), label
    figs.clear()
    plot_results(stored, tmp_path / "b", experiments=["exp01"], datasets=["dl19"], plots=["sweep"])  # no limits
    assert any(ax.get_ylim() != (0, 1) and "%" not in ax.get_ylabel() for f in figs for ax in f.axes)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0.5, 1.4])
    with y_limits({"raw": [0, 1]}), caplog.at_level("WARNING"):
        set_ylim(ax, "raw")
    assert ax.get_ylim() == (0, 1) and "clipped" in caplog.text

    seen = {}
    monkeypatch.setattr(rp, "plot_results", lambda *a, **kw: seen.update(opt=a[8]) or [])
    for argv, expect in ((["--root", str(stored)], {"raw": (0.0, 1.0), "pct": (0.0, 140.0)}),
                         (["--root", str(stored), "--ylim-pct", "20", "120"], {"raw": (0.0, 1.0), "pct": (20.0, 120.0)}),
                         (["--root", str(stored), "--no-fixed-ylim"], None)):
        monkeypatch.setattr(sys, "argv", ["plot_results.py"] + argv)
        runpy.run_path("scripts/plot_results.py", run_name="__main__")
        assert seen["opt"].y_limits == expect, argv


def test_paper_row_three_one_column_pdfs_and_a_legend(stored, tmp_path, monkeypatch):
    import shutil

    import matplotlib.figure

    from pec.analysis.result_plots import PaperRow

    shutil.copytree(stored / "exp01_dl19", stored / "exp01_dl20")
    saved = {}
    real = matplotlib.figure.Figure.savefig

    def record(fig, path, *a, **kw):
        saved[Path(path).name] = fig
        return real(fig, path, *a, **kw)

    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", record)
    assert PaperRow().experiments == ("exp01", "exp02") and PaperRow().value == "raw"  # the defaults
    row = PaperRow(experiments=("exp01",), value="pct")
    assert row.datasets == ("dl19", "dl20", "trec_covid") and row.metric == "nDCG@10"
    assert sum(row.latex_widths()) == pytest.approx(1.0)
    out = tmp_path / "plots"
    w = plot_results(stored, out, experiments=["exp01"], plots=["paper_row"],
                     opt=Options(y_limits={"raw": [0, 1], "pct": [0, 140]}, fmt="pdf"),
                     runs={"exp01": ["centroid", "seed_rrf", "zero_shot"]}, paper_row=row)
    assert sorted(p.name for p in w) == ["exp01_dl19.pdf", "exp01_dl20.pdf", "exp01_legend.pdf",
                                         "exp01_trec_covid.pdf"]
    assert all(p.parent == out / "paper" and p.exists() for p in w)
    figs = [saved[f"exp01_{ds}.pdf"] for ds in row.datasets]
    widths = [f.get_size_inches()[0] for f in figs]
    assert widths == pytest.approx(row.figure_widths()) and sum(widths) == pytest.approx(3.33)
    axes = [f.axes[0] for f in figs]
    plot_inches = [ax.get_position().width * wd for ax, wd in zip(axes, widths)]
    assert plot_inches == pytest.approx([row.axes_width()] * 3)  # the same plot area in all three
    assert {round(f.get_size_inches()[1], 6) for f in figs} == {1.3}
    assert {ax.get_ylim() for ax in axes} == {(0, 140)}  # one y range
    left, middle, right = axes
    assert left.get_ylabel() == "nDCG@10\n(% of Oracle)" and any(t.get_visible() and t.get_text()
                                                                  for t in left.get_yticklabels())
    for ax in (middle, right):  # only the x axis
        assert ax.get_ylabel() == "" and not ax.yaxis.get_tick_params()["labelleft"]
    for ax in axes:
        assert ax.get_xlabel() == "context size m" and [t.get_text() for t in ax.get_xticklabels()] == ["1", "3", "5"]
        assert ax.get_legend() is None and ax.get_title() == ""
    legend = saved["exp01_legend.pdf"].legends[0]
    assert [t.get_text() for t in legend.get_texts()] == ["centroid", "seed_rrf", "zero_shot", "Oracle"]
    assert saved["exp01_legend.pdf"].get_size_inches()[0] == pytest.approx(3.33)
    text = (out / "paper/exp01_widths.txt").read_text()
    assert text.splitlines() == ["dl19: 0.428\\columnwidth", "dl20: 0.281\\columnwidth",
                                 "trec_covid: 0.290\\columnwidth"]  # rounded down: they never exceed one column

    # raw scores without fixed limits: Oracle lines, and the union of the three y ranges
    saved.clear()
    plot_results(stored, tmp_path / "raw", experiments=["exp01"], plots=["paper_row"],
                 opt=Options(fmt="pdf"), runs={"exp01": ["centroid"]},
                 paper_row=PaperRow(experiments=("exp01",), value="raw", column_width=3.5, legend_columns=1))
    axes = [saved[f"exp01_{ds}.pdf"].axes[0] for ds in row.datasets]
    assert len({ax.get_ylim() for ax in axes}) == 1 and axes[0].get_ylabel() == "nDCG@10"
    assert sum(saved[f"exp01_{ds}.pdf"].get_size_inches()[0] for ds in row.datasets) == pytest.approx(3.5)
    oracle_entries = [t.get_text() for t in saved["exp01_legend.pdf"].legends[0].get_texts() if t.get_text().startswith("Oracle")]
    assert oracle_entries == ["Oracle (E5)", "Oracle (BM25)"]  # raw scores: both Oracle lines

    for bad in ({"datasets": ["dl19", "dl20"]}, {"value": "both"}, {"colour": "red"}, {"column_width": 0.3}):
        with pytest.raises(ValueError):
            PaperRow.from_config(bad).axes_width()


def test_paper_row_cli(stored, monkeypatch):
    import pec.analysis.result_plots as rp

    seen = {}
    monkeypatch.setattr(rp, "plot_results", lambda *a, **kw: seen.update(plots=a[4], row=a[12]) or [])
    monkeypatch.setattr(sys, "argv", ["plot_results.py", "--root", str(stored), "--plots", "paper_row",
                                      "--paper-metric", "R@100", "--paper-value", "raw", "--column-width", "3.5",
                                      "--paper-views", "dl19=grade>=2", "--paper-datasets", "dl20", "dl19",
                                      "trec_covid"])
    runpy.run_path("scripts/plot_results.py", run_name="__main__")
    row = seen["row"]
    assert seen["plots"] == ["paper_row"] and row.metric == "R@100" and row.value == "raw"
    assert row.column_width == 3.5 and row.views == {"dl19": "grade>=2"} and row.datasets == ("dl20", "dl19", "trec_covid")
    assert row.experiments == ("exp01", "exp02")  # the default
    monkeypatch.setattr(sys, "argv", ["plot_results.py", "--root", str(stored), "--plots", "paper_row",
                                      "--paper-experiments", "exp02"])
    runpy.run_path("scripts/plot_results.py", run_name="__main__")
    assert seen["row"].experiments == ("exp02",)
    monkeypatch.setattr(sys, "argv", ["plot_results.py", "--root", str(stored)])
    runpy.run_path("scripts/plot_results.py", run_name="__main__")
    assert "paper_row" not in seen["plots"] and seen["row"] is None  # only when asked for
    monkeypatch.setattr(sys, "argv", ["plot_results.py", "--root", str(stored), "--plots", "paper_row",
                                      "--paper-views", "dl19"])
    with pytest.raises(SystemExit):
        runpy.run_path("scripts/plot_results.py", run_name="__main__")


def test_paper_row_default_exp01_and_exp02_raw_with_both_oracles(stored, tmp_path, monkeypatch):
    import shutil

    import matplotlib.figure

    from pec.analysis.result_plots import PaperRow

    for exp in ("exp01", "exp02"):
        shutil.copytree(stored / f"{exp}_dl19", stored / f"{exp}_dl20")
    saved = {}
    real = matplotlib.figure.Figure.savefig
    monkeypatch.setattr(matplotlib.figure.Figure, "savefig",
                        lambda fig, path, *a, **kw: (saved.__setitem__(Path(path).name, fig), real(fig, path, *a, **kw))[1])
    # exp02's tables hold no BM25 Oracle: pinned values draw it (as configs/oracle_values.yaml does)
    pins = {"nDCG@10": {ds: {"true_query@bm25": {v: 0.45}} for ds, v in
                        (("dl19", "grade=3"), ("dl20", "grade=3"), ("trec_covid", "grade=2"))}}
    out = tmp_path / "plots"
    w = plot_results(stored, out, experiments=["exp01", "exp02"], plots=["paper_row"],
                     opt=Options(y_limits={"raw": [0, 1], "pct": [0, 140]}, fmt="pdf", pinned=pins),
                     runs={"exp01": ["centroid", "seed_rrf"], "exp02": ["centroid"]}, paper_row=PaperRow())
    names = sorted(p.name for p in w)
    assert names == sorted(f"{e}_{d}.pdf" for e in ("exp01", "exp02") for d in ("dl19", "dl20", "trec_covid", "legend"))
    assert (out / "paper/exp01_widths.txt").exists() and (out / "paper/exp02_widths.txt").exists()
    for exp, xlabel in (("exp01", "context size m"), ("exp02", "α")):
        axes = [saved[f"{exp}_{ds}.pdf"].axes[0] for ds in ("dl19", "dl20", "trec_covid")]
        assert axes[0].get_ylabel() == "nDCG@10"  # the values, not % of the Oracle
        assert {ax.get_ylim() for ax in axes} == {(0, 1)} and {ax.get_xlabel() for ax in axes} == {xlabel}
        for ax in axes:  # both Oracles in every plot
            oracles = {l.get_label() for l in ax.get_lines() if l.get_label().startswith("Oracle")}
            assert oracles == {"Oracle (E5)", "Oracle (BM25)"}, exp
        legend = [t.get_text() for t in saved[f"{exp}_legend.pdf"].legends[0].get_texts()]
        assert legend[-2:] == ["Oracle (E5)", "Oracle (BM25)"]
    bm25 = next(l for l in saved["exp02_dl19.pdf"].axes[0].get_lines() if l.get_label() == "Oracle (BM25)")
    assert set(bm25.get_ydata()) == {0.45}  # the pinned value, a flat line
    with pytest.raises(ValueError):
        PaperRow.from_config({"experiments": ["exp03"]})
    assert PaperRow.from_config({"experiment": "exp02"}).experiments == ("exp02",)


def test_static_icl_is_called_fixed_llm_in_the_figures(stored, tmp_path, monkeypatch):
    import matplotlib.figure

    from pec.analysis.oracle import display_name

    assert display_name("static_icl") == "fixed_llm" and display_name("static_icl@bm25") == "fixed_llm@bm25"
    assert display_name("centroid") == "centroid" and display_name("dl19 (grade=3)") == "dl19 (grade=3)"
    d = stored / "exp01_dl19/baseline/metrics_by_context_size.csv"
    t = pd.read_csv(d, dtype={"context_size": str})
    pd.concat([t, t[t["name"] == "zero_shot"].assign(name="static_icl")]).to_csv(d, index=False)
    figs = []
    real = matplotlib.figure.Figure.savefig
    monkeypatch.setattr(matplotlib.figure.Figure, "savefig",
                        lambda fig, path, *a, **kw: (figs.append(fig), real(fig, path, *a, **kw))[1])
    w = plot_results(stored, tmp_path / "p", experiments=["exp01"], datasets=["dl19"], plots=["sweep"],
                     models=["centroid", "static_icl"], opt=Options(views=["eq3"]))
    legend = {t.get_text() for f in figs for ax in f.axes if ax.get_legend() for t in ax.get_legend().get_texts()}
    assert "fixed_llm" in legend and "static_icl" not in legend
    assert "static_icl" in set(pd.read_csv(w[0].with_suffix(".csv"))["name"])  # the tables keep the run name
