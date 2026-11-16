"""Text query formulation baselines (from proactive-icl-query-pair) on the synthetic corpus:
a fake LLM client, a fake doc2query model and a fake query encoder stand in for the real ones."""
import json
import sys
import types
import zlib

import numpy as np
import pandas as pd
import pyterrier as pt
import pytest

from pec.data.qrels import normalise_qrels
from pec.data.seeds import context_draws
from pec.query_formulation import QF, QueryFormulation, TextQueryRetriever, extract_query, rr_fusion


class FakeLLM:
    """OpenAI-client stand-in: records each prompt and answers with a tagged query."""

    def __init__(self, answer="<query>toy query number 1</query>"):
        self.prompts, self.answer = [], answer
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self._create))

    def _create(self, model, messages):
        self.prompts.append((model, messages))
        msg = types.SimpleNamespace(content=self.answer)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])


class FakeEncoder(pt.Transformer):
    """[qid, query] -> query_vec: "... number k ..." maps to query k's perfectly relevant
    centroid, anything else to a fixed pseudo-random direction."""

    def __init__(self, store, qrels):
        self.centres = {}
        for qid, g in qrels[qrels["label"] == 3].groupby("qid"):
            v = store.vectors(store.rows(g["docno"].tolist())).mean(0)
            self.centres[qid] = v / np.linalg.norm(v)
        self.dim = store.dim
        self.calls = 0

    def vec(self, text):
        for word, nxt in zip(text.split(), text.split()[1:]):
            if word == "number" and f"q{nxt}" in self.centres:
                return self.centres[f"q{nxt}"]
        v = np.random.default_rng(zlib.crc32(text.encode())).normal(size=self.dim).astype(np.float32)
        return v / np.linalg.norm(v)

    def transform(self, inp):
        self.calls += len(inp)
        return inp.assign(query_vec=[self.vec(q) for q in inp["query"]])


@pytest.fixture
def rbo_file(tmp_path):
    rows = [(121352, "rbo", f"d{i}", i, s, f"original query| related query {i}")
            for i, s in enumerate(np.linspace(0.1, 0.9, 9))]
    path = tmp_path / "rbo.res"
    pd.DataFrame(rows).to_csv(path, sep="\t", header=False, index=False)
    return path


def toy_texts(toy):
    return {d: f"toy passage {d} about query number {int(d[1:]) // 34 if int(d[1:]) < 136 else 9}"
            for d in toy["docnos"]}


def test_extract_query():
    assert extract_query("blah <query> a b c </query> tail") == "a b c"
    assert extract_query("  no tags here ") == "no tags here"


def test_llm_methods_prompt_and_request():
    llm = FakeLLM()
    qf = QF(client=llm)
    out = qf.zero_shot(["first passage", "second passage"])
    assert out["final_query"] == "toy query number 1" and out["seed_set"] == ["first passage", "second passage"]
    model, messages = llm.prompts[-1]
    assert model == "llama-3-8b-instruct" and len(messages) == 1 and messages[0]["role"] == "user"
    prompt = messages[0]["content"]
    assert prompt.startswith("You are a search engine user who is looking for information on a topic")
    assert prompt.endswith("Context: first passage\nsecond passage")


def test_static_icl_uses_the_static_query_pairs(rbo_file):
    llm = FakeLLM()
    qf = QF(client=llm, rbo_path=str(rbo_file))
    out = qf.run_qf("static", ["a passage"], 1, "121352")
    assert out["static_qid"] == "121352" and out["final_query"] == "toy query number 1"
    # p = 1: of the pairs in the middle score band (0.4-0.6), the one above the middle is "good", the middle one "bad"
    assert out["query-pairs"]["positive"] == '"original query" shares a similar information need to "related query 5"'
    assert out["query-pairs"]["negative"] == '"original query" has a different information need to "related query 4"'
    prompt = llm.prompts[-1][1][0]["content"]
    assert prompt.startswith("You are a helpful query reformulator. Given the context, your task is to reformulate the query.")
    assert "Provided documents: a passage" in prompt and out["query-pairs"]["positive"] in prompt


def test_sliding_window_and_doc2query(monkeypatch):
    qf = QF()
    text = " ".join(f"w{i}" for i in range(60))
    out = qf.run_qf("sw", [text])
    assert len(out["final_queries"]) == 3 and all(len(q.split()) <= 25 for q in out["final_queries"])
    assert sorted(w for q in out["final_queries"] for w in q.split()) == sorted(text.split())

    class Doc2Query(pt.Transformer):
        def __init__(self, num_samples):
            assert num_samples == 1

        def transform(self, df):
            return df.assign(querygen=[f"query for {t}" for t in df["text"]])

    monkeypatch.setitem(sys.modules, "pyterrier_doc2query", types.SimpleNamespace(Doc2Query=Doc2Query))
    out = qf.run_qf("d2q", ["alpha", "beta"])
    assert out["final_queries"] == ["query for alpha", "query for beta"]


def test_rr_fusion_sums_reciprocal_ranks():
    first = pd.DataFrame({"qid": ["q", "q"], "docno": ["a", "b"], "score": [0.9, 0.8]})
    second = pd.DataFrame({"qid": ["q", "q"], "docno": ["b", "c"], "score": [0.95, 0.7]})
    fused = rr_fusion([first, second], k=60)
    assert fused["docno"].tolist() == ["b", "a", "c"]  # one row per document
    assert fused.set_index("docno")["score"]["b"] == pytest.approx(1 / 61 + 1 / 60)


def test_query_formulation_cache_and_retrieval(toy, tmp_path):
    store, qrels = toy["store"], normalise_qrels(toy["qrels"])
    texts = toy_texts(toy)
    contexts = context_draws(qrels, [3], repeats=1)
    llm = FakeLLM()
    cache = tmp_path / "zero_shot.jsonl"
    encoder = FakeEncoder(store, qrels)
    pipe = QueryFormulation(QF(client=llm), "zero_shot", texts, cache_path=cache) >> TextQueryRetriever(store, encoder, 20)
    run = pipe(contexts)
    assert len(llm.prompts) == len(contexts)
    assert run.groupby("qid").size().max() == 20
    for qid, seeds in zip(contexts["qid"], contexts["seed_docnos"]):
        assert not set(seeds) & set(run[run["qid"] == qid]["docno"])  # context removed
    recs = [json.loads(line) for line in cache.read_text().splitlines()]
    assert len(recs) == len(contexts) and recs[0]["result"]["final_query"] == "toy query number 1"
    assert recs[0]["model"] == "llama-3-8b-instruct"
    assert recs[0]["result"]["seed_set"] == [texts[d] for d in recs[0]["seed_docnos"]]

    # a rerun reads the cache: no new LLM calls
    llm2 = FakeLLM()
    QueryFormulation(QF(client=llm2), "zero_shot", texts, cache_path=cache)(contexts)
    assert not llm2.prompts


def test_multi_query_fusion(toy):
    store, qrels = toy["store"], normalise_qrels(toy["qrels"])
    encoder = FakeEncoder(store, qrels)
    seeds = tuple(sorted(qrels[(qrels["qid"] == "q0") & (qrels["label"] == 1)]["docno"])[:2])
    topics = pd.DataFrame({"qid": ["q0", "q1"], "seed_docnos": [seeds, ()],
                           "queries": [("about number 0", "unrelated words"), ("number 1",)]})
    run = TextQueryRetriever(store, encoder, 30)(topics)
    assert set(run["qid"]) == {"q0", "q1"} and run.groupby("qid").size().max() == 30
    assert not set(seeds) & set(run[run["qid"] == "q0"]["docno"])
    assert run.groupby("qid")["rank"].apply(lambda r: r.tolist() == list(range(len(r)))).all()
    # q0's ranking is fused: scores are reciprocal ranks, not cosines
    assert run[run["qid"] == "q0"]["score"].max() <= 2 / 60 + 1e-9
    assert TextQueryRetriever(store, encoder, 30)(topics.assign(queries=[(), ()])).empty


def test_exp01_baseline_runs_query_formulation(toy, tmp_path, monkeypatch, rbo_file):
    from pec.config import load_config
    from pec.experiments.exp01 import build_context, dataset_config, run_parts
    import pec.index.e5 as e5
    import pec.query_formulation as qfm

    store, qrels = toy["store"], normalise_qrels(toy["qrels"])
    encoder = FakeEncoder(store, qrels)
    monkeypatch.setattr(e5, "e5_model", lambda name, **kw: types.SimpleNamespace(query_encoder=lambda: encoder))
    llm = FakeLLM()
    monkeypatch.setattr(qfm.QF, "client", property(lambda self: llm))

    class Doc2Query(pt.Transformer):
        def __init__(self, num_samples):
            pass

        def transform(self, df):
            return df.assign(querygen=list(df["text"]))

    monkeypatch.setitem(sys.modules, "pyterrier_doc2query", types.SimpleNamespace(Doc2Query=Doc2Query))

    ctx = build_context(store, toy["topics"].astype({"qid": str}), qrels, encoder.centres, toy_texts(toy),
                        index_key="toy")
    cfg = dataset_config(load_config("configs/exp01_dl19.yaml"), "dl19")
    c = cfg["exp01"]["baseline"]
    c.update(num_results=50, recall_ks=[10, 50], context_sizes=[1, 3], repeats=2, mmr={"lambdas": []},
             bm25_rrf={"enabled": False}, bm25_centroid={"enabled": False})
    c["query_formulation"]["static_icl"]["rbo_path"] = str(rbo_file)
    c["query_formulation"]["retrievers"] = ["e5"]  # text retrievers: test_exp01_text_retrievers
    cfg["cache_dir"] = str(tmp_path / "cache")
    run_parts(ctx, cfg, tmp_path, parts=["baseline"])

    sweep = pd.read_csv(tmp_path / "baseline/summary_by_context_size.csv", dtype={"context_size": str})
    names = {"zero_shot", "static_icl", "sliding_window", "doc2query"}
    assert names <= set(sweep["name"])
    # same queries per context size as the centroid: the same context draws
    n = sweep.pivot_table(index=["name", "context_size"], values="n_queries").reset_index()
    for name in names:
        assert (n[n["name"] == name]["n_queries"].to_numpy() == n[n["name"] == "centroid"]["n_queries"].to_numpy()).all()
    for method in names:
        assert (tmp_path / f"baseline/query_formulation/{method}.jsonl").exists()
    for f in ["baseline/context_sweep_ge2.png", "baseline/context_sweep_eq3.png"]:  # the E5 methods' figure
        assert (tmp_path / f).exists(), f
    zero = [json.loads(line) for line in (tmp_path / "baseline/query_formulation/zero_shot.jsonl").read_text().splitlines()]
    draws = context_draws(qrels, [1, 3], 2, cfg["exp01"]["random_seed"])  # the draws every baseline gets
    assert {(r["qid"], tuple(r["seed_docnos"])) for r in zero} == set(zip(draws["qid"], draws["seed_docnos"]))


# ------------------------------------------------------------------ text retrievers

class FakeTextRetriever(pt.Transformer):
    """[qid, query] -> documents ranked by words shared with their toy text."""

    def __init__(self, texts, num_results=1000):
        self.texts, self.num_results = texts, num_results

    def transform(self, inp):
        rows = []
        for qid, query in zip(inp["qid"], inp["query"]):
            words = set(query.split())
            scored = sorted(((len(words & set(t.split())) - int(d[1:]) * 1e-6, d) for d, t in self.texts.items()),
                            reverse=True)[: self.num_results]
            rows += [{"qid": qid, "query": query, "docno": d, "score": s} for s, d in scored]
        return pt.model.add_ranks(pd.DataFrame(rows))


def test_get_retr_builds_the_proactive_icl_retrievers(monkeypatch, tmp_path):
    import pec.text_retrievers as tr

    calls = []

    class Stage(pt.Transformer):
        def __init__(self, *args, **kwargs):
            calls.append((type(self).__name__, args, kwargs))

        def transform(self, df):
            return df

    class PisaIndex(Stage):
        def bm25(self, **kw):
            calls.append(("bm25", (), kw))
            return Stage()

        def quantized(self, **kw):
            calls.append(("quantized", (), kw))
            return Stage()

    class FlexIndex(Stage):
        def np_retriever(self, **kw):
            calls.append(("np_retriever", (), kw))
            return Stage()

    class NumpyIndex(Stage):
        pass

    class TctColBert(Stage):
        pass

    class MonoT5ReRanker(Stage):
        pass

    class Splade(Stage):
        def query_encoder(self):
            return Stage()

    monkeypatch.setitem(sys.modules, "pyterrier_pisa", types.SimpleNamespace(PisaIndex=PisaIndex))
    monkeypatch.setitem(sys.modules, "pyterrier_t5", types.SimpleNamespace(MonoT5ReRanker=MonoT5ReRanker))
    monkeypatch.setitem(sys.modules, "pyterrier_splade", types.SimpleNamespace(Splade=Splade))
    monkeypatch.setitem(sys.modules, "pyterrier_dr", types.SimpleNamespace(FlexIndex=FlexIndex, NumpyIndex=NumpyIndex,
                                                                          TctColBert=TctColBert))
    monkeypatch.setattr(pt.Artifact, "from_hf", staticmethod(lambda repo: (calls.append(("from_hf", (repo,), {})),
                                                                           PisaIndex())[1]))
    monkeypatch.setattr(pt, "get_dataset", lambda name: calls.append(("get_dataset", (name,), {})) or name)
    monkeypatch.setattr(pt.text, "get_text", lambda ds, fields: calls.append(("get_text", (ds, fields), {})) or Stage())
    for name in ["bm25.pisa", "tct.flex", "tct.np", "splade.pisa"]:
        (tmp_path / name).mkdir()
    dl = {"bm25": str(tmp_path / "bm25.pisa"), "colbert": {"flex": str(tmp_path / "tct.flex")},
          "splade": str(tmp_path / "splade.pisa")}
    covid = {"bm25": "hf:pyterrier/trec-covid.pisa", "colbert": {"numpy": str(tmp_path / "tct.np")}}

    def built(name, entry, corpus="irds:msmarco-passage"):
        calls.clear()
        tr.get_retr(name, entry, corpus, num_results=1100)
        return calls[:]

    assert built("bm25", dl)[:2] == [("PisaIndex", (str(tmp_path / "bm25.pisa"),), {}), ("bm25", (), {"num_results": 1100})]
    mono = built("monot5", covid, "irds:beir/trec-covid")
    assert ("from_hf", ("pyterrier/trec-covid.pisa",), {}) in mono
    assert ("get_text", ("irds:beir/trec-covid", ["text"]), {}) in mono and ("MonoT5ReRanker", (), {}) in mono
    assert ("TctColBert", ("castorini/tct_colbert-v2-hnp-msmarco",), {}) in built("colbert", dl)
    assert ("np_retriever", (), {"num_results": 1100}) in calls
    assert ("NumpyIndex", (str(tmp_path / "tct.np"),), {"num_results": 1100}) in built("colbert", covid)
    splade = built("splade", dl)
    assert ("Splade", ("naver/splade-cocondenser-ensembledistil",), {}) in splade
    assert ("PisaIndex", (str(tmp_path / "splade.pisa"),), {"stemmer": "none"}) in splade
    assert ("quantized", (), {"num_results": 1100}) in splade
    with pytest.raises(ValueError):
        tr.get_retr("rm3", dl, "irds:msmarco-passage")


def test_monot5_reranks_bm25_top_50(monkeypatch):
    import pec.text_retrievers as tr

    docs = {f"d{i}": "w" for i in range(80)}
    bm25 = FakeTextRetriever(docs)
    seen = []

    class MonoT5ReRanker(pt.Transformer):
        def transform(self, df):
            seen.append(len(df))
            return df

    monkeypatch.setitem(sys.modules, "pyterrier_t5", types.SimpleNamespace(MonoT5ReRanker=MonoT5ReRanker))
    monkeypatch.setattr(tr, "_pisa_index", lambda path, what, **kw: types.SimpleNamespace(bm25=lambda **k: bm25))
    monkeypatch.setattr(pt, "get_dataset", lambda name: name)
    monkeypatch.setattr(pt.text, "get_text", lambda ds, fields: pt.apply.generic(lambda df: df))
    tr.get_retr("monot5", {"bm25": "x"}, "irds:msmarco-passage")(pd.DataFrame({"qid": ["q"], "query": ["w"]}))
    assert seen == [50]


def test_residual_text_retriever_removes_context(toy):
    from pec.text_retrievers import ResidualTextRetriever

    texts = toy_texts(toy)
    retr = ResidualTextRetriever(FakeTextRetriever(texts, 40), 30)
    seeds = ("d0", "d1", "d2")
    run = retr(pd.DataFrame({"qid": ["q0"], "query": ["about query number 0"], "seed_docnos": [seeds]}))
    assert len(run) == 30 and not set(seeds) & set(run["docno"]) and run["rank"].tolist() == list(range(30))
    assert set(run["docno"]) <= {d for d, t in texts.items() if t.endswith("number 0")}


def test_exp01_text_retrievers(toy, tmp_path, monkeypatch):
    from pec.config import load_config
    from pec.experiments.exp01 import build_context, dataset_config, run_parts
    import pec.query_formulation as qfm
    import pec.text_retrievers as tr

    store, qrels = toy["store"], normalise_qrels(toy["qrels"])
    texts = toy_texts(toy)
    built = []
    monkeypatch.setattr(tr, "get_retr", lambda name, entry, corpus, num_results=1000, monot5_depth=50:
                        built.append((name, entry, corpus, num_results)) or FakeTextRetriever(texts, num_results))
    llm = FakeLLM("<query>about query number 2</query>")
    monkeypatch.setattr(qfm.QF, "client", property(lambda self: llm))
    encoder = FakeEncoder(store, qrels)
    ctx = build_context(store, toy["topics"].astype({"qid": str}), qrels, encoder.centres, texts, index_key="toy")
    cfg = dataset_config(load_config("configs/exp01_dl19.yaml"), "dl19")
    c = cfg["exp01"]["baseline"]
    c.update(num_results=50, recall_ks=[10, 50], context_sizes=[1, 3], repeats=1, mmr={"lambdas": []},
             bm25_rrf={"enabled": False}, bm25_centroid={"enabled": False})
    c["query_formulation"].update(methods=["zero_shot", "sliding_window"], retrievers=["bm25", "splade"])
    cfg["cache_dir"] = str(tmp_path / "cache")
    run_parts(ctx, cfg, tmp_path, parts=["baseline"])

    # paths.yaml's MS MARCO entries, searched deep enough for 50 results after removing a 3-passage context
    assert [b[0] for b in built] == ["bm25", "splade"]
    assert built[0][1] == cfg["text_retrievers"]["msmarco_v1"] and built[0][2] == "irds:msmarco-passage"
    assert built[0][3] == 53
    sweep = pd.read_csv(tmp_path / "baseline/summary_by_context_size.csv", dtype={"context_size": str})
    expected = {"zero_shot@bm25", "zero_shot@splade", "sliding_window@bm25", "sliding_window@splade",
                "true_query@bm25", "true_query@splade"}
    assert expected <= set(sweep["name"]) and "zero_shot" not in set(sweep["name"])  # no e5 run asked for
    assert len(llm.prompts) == len(context_draws(qrels, [1, 3], 1, cfg["exp01"]["random_seed"]))  # once per context
    pq = pd.read_csv(tmp_path / "baseline/metrics_perquery.csv", dtype={"context_size": str})
    q2 = pq[(pq["name"] == "zero_shot@bm25") & (pq["view"] == "grade=3")].groupby("qid")["nDCG@10"].mean()
    assert q2["q2"] > q2.drop("q2").max()  # the generated query is about q2: only q2 does well
    assert (tmp_path / "baseline/context_sweep_qf_bm25_ge2.png").exists()
