import numpy as np
import pandas as pd

from pec.config import deep_merge, is_placeholder, load_config
from pec.data.qrels import grade_sets, normalise_qrels, qrels_view
from pec.data.seeds import seed_topics


def test_qrels_views():
    qrels = normalise_qrels(pd.DataFrame({"qid": [1, 1, 1, 1], "docno": ["a", "b", "c", "d"], "label": [0, 1, 2, 3]}))
    ge2 = qrels_view(qrels, "grade>=2").set_index("docno")["label"].to_dict()
    eq3 = qrels_view(qrels, "grade=3").set_index("docno")["label"].to_dict()
    assert ge2 == {"a": 0, "b": 0, "c": 2, "d": 3}
    assert eq3 == {"a": 0, "b": 0, "c": 0, "d": 3}  # one grade left: same nDCG / recall as a binary label
    assert grade_sets(qrels)["1"][1] == ["b"]


def test_relevance_scheme_from_config():
    from pec.data.qrels import DL_RELEVANCE, Relevance

    covid = Relevance.from_config({"context": 1, "target": 2, "views": {"grade=2": [2]},
                                   "names": {0: "not relevant", 1: "partially relevant", 2: "relevant"}})
    assert (covid.context, covid.target, covid.view_names, covid.intermediate) == (1, 2, ("grade=2",), ())
    assert covid.primary_view == covid.target_view == "grade=2" and covid.name(2) == "relevant"
    assert DL_RELEVANCE.intermediate == (2,) and DL_RELEVANCE.target_view == "grade=3"
    assert DL_RELEVANCE.primary_view == "grade>=2" and Relevance.from_config(None) == DL_RELEVANCE
    qrels = normalise_qrels(pd.DataFrame({"qid": [1, 1, 1], "docno": ["a", "b", "c"], "label": [0, 1, 2]}))
    assert qrels_view(qrels, "grade=2", covid).set_index("docno")["label"].to_dict() == {"a": 0, "b": 0, "c": 2}
    import pytest

    with pytest.raises(ValueError):
        qrels_view(qrels, "grade=3", covid)


def test_ndcg_unchanged_by_keeping_single_grade_label():
    import pyterrier as pt
    from ir_measures import R, nDCG

    qrels = pd.DataFrame({"qid": ["1"] * 3, "docno": ["a", "b", "c"], "label": [3, 0, 3]})
    run = pd.DataFrame({"qid": ["1"] * 3, "docno": ["b", "a", "c"], "score": [3.0, 2.0, 1.0]})
    binary = qrels.assign(label=(qrels["label"] == 3).astype(int))
    a = pt.Experiment([run], qrels[["qid"]].drop_duplicates(), qrels, [nDCG @ 10, R @ 10])
    b = pt.Experiment([run], qrels[["qid"]].drop_duplicates(), binary, [nDCG @ 10, R @ 10])
    assert a[["nDCG@10", "R@10"]].values.tolist() == b[["nDCG@10", "R@10"]].values.tolist()


def test_dataset_configs_and_topics_variant(monkeypatch):
    import pyterrier as pt

    from pec.data.datasets import load_dataset, relevance

    cfg = load_config("configs/exp01_dl19.yaml")
    assert cfg["exp01"]["datasets"] == ["dl19", "dl20", "trec_covid"]
    assert cfg["datasets"]["trec_covid"]["index"] != cfg["datasets"]["dl19"]["index"] == cfg["datasets"]["dl20"]["index"]
    assert relevance(cfg, "trec_covid").target == 2 and relevance(cfg, "dl20").target == 3
    assert "all" not in [str(k) for k in cfg["exp01"]["overrides"]["trec_covid"]["baseline"]["context_sizes"]]

    seen = {}

    class FakeDataset:
        def get_topics(self, variant=None):
            seen["variant"] = variant
            return pd.DataFrame({"qid": ["1"], "query": ["what is the origin of covid-19"]})

        def get_qrels(self):
            return pd.DataFrame({"qid": ["1", "1"], "docno": ["x", "y"], "label": [1, 2]})

    monkeypatch.setattr(pt, "get_dataset", lambda name: FakeDataset())
    topics, qrels = load_dataset(cfg, "trec_covid")
    assert seen["variant"] == "text" and topics["query"].iloc[0].startswith("what is")
    load_dataset(cfg, "dl20")
    assert seen["variant"] is None


def test_seed_topics_use_grade_one_only(toy):
    topics = seed_topics(normalise_qrels(toy["qrels"]))
    labels = toy["qrels"].set_index(["qid", "docno"])["label"]
    for qid, seeds in zip(topics["qid"], topics["seed_docnos"]):
        assert seeds and all(labels[(qid, d)] == 1 for d in seeds)
    assert "query" not in topics.columns  # the information need stays hidden


def test_config_include_and_placeholders(tmp_path):
    (tmp_path / "base.yaml").write_text("a: {x: 1, y: 2}\npath: /path/to/thing\n")
    (tmp_path / "exp.yaml").write_text("include: [base.yaml]\na: {y: 3}\n")
    cfg = load_config(tmp_path / "exp.yaml")
    assert cfg["a"] == {"x": 1, "y": 3} and is_placeholder(cfg["path"])
    assert deep_merge({"a": {"b": 1}}, {"a": {"c": 2}}) == {"a": {"b": 1, "c": 2}}


def test_repo_configs_load():
    cfg = load_config("configs/exp01_dl19.yaml")
    assert cfg["exp01"]["datasets"][0] == "dl19" and set(cfg["indexes"]) == {"msmarco_v1", "trec_covid"}
    assert load_config("configs/exp02_noise.yaml")["exp02"]["context_size"] == 5


def test_context_draws_nested_deterministic_and_skip_short_queries(toy):
    from pec.data.seeds import context_draws, ordered_sizes

    qrels = normalise_qrels(toy["qrels"])  # 10 grade-1 passages per toy query
    draws = context_draws(qrels, [5, 3, 12, "all"], repeats=3, random_seed=1)
    assert ordered_sizes(draws["context_size"]) == ["3", "5", "all"]  # k=12 > 10 grade-1 docs: skipped
    assert (draws[draws["context_size"] == "all"]["repeat"] == 0).all()
    labels = qrels.set_index(["qid", "docno"])["label"]
    for (qid, repeat), g in draws[draws["context_size"] != "all"].groupby(["qid", "repeat"]):
        ctx = dict(zip(g["context_size"], g["seed_docnos"]))
        assert len(ctx["3"]) == 3 and len(ctx["5"]) == 5 and set(ctx["3"]) <= set(ctx["5"])
        assert all(labels[(qid, d)] == 1 for d in ctx["5"])
    again = context_draws(qrels, [5, 3, 12, "all"], repeats=3, random_seed=1)
    assert draws.equals(again)
    by_repeat = draws[draws["context_size"] == "5"].groupby("repeat")["seed_docnos"].apply(tuple)
    assert by_repeat.nunique() > 1  # different repeats draw different contexts
