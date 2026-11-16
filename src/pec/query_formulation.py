"""Text query formulation baselines from proactive-icl-query-pair (src/models/models.py).

Four methods turn a context of partially relevant passages into search queries:

- zero_shot:      an LLM writes one query from the context (no examples)
- static_icl:     an LLM writes one query from the context, shown fixed good/bad
                  query-pair examples of a single static MS MARCO training query
                  (qid 121352, p = 1 pair of each), taken from the RBO query-pair file
- sliding_window: the top_k = 3 windows of window_size = 25 tokens of the context with
                  the highest mean IDF, each used as a query
- doc2query:      one doc2query-T5 query per context passage (num_samples = 1)

The QF methods below are copied from proactive-icl-query-pair unchanged: the prompts, the
LLM request (model llama-3-8b-instruct through the OpenAI client, no sampling
parameters), the <query></query> extraction, the RBO example selection and the
sliding-window and doc2query settings. Only __init__ differs: each resource (LLM client,
RBO file, doc2query model) is loaded on first use, so a method needs only its own.

In Experiment 1 the methods get exactly the contexts the other baselines get
(pec.data.seeds.context_draws) and their queries are searched in the same E5 index:
each query is E5-encoded ("query: " prefix) and searched with FaissRetriever (context
passages removed). The same queries are also run through the text retrievers of
proactive-icl-query-pair (BM25, BM25 >> monoT5, TCT-ColBERT, SPLADE; see
pec.text_retrievers), again with the context passages removed. The rankings of the multi-query methods (sliding_window,
doc2query) are fused by reciprocal rank fusion with pyterrier_alpha.fusion.rr_fusion, the
fusion proactive-icl-query-pair names. (That repository passes rr_fusion a single frame
of the concatenated per-query results, and with a single input rr_fusion merges nothing:
the hits are only re-sorted by raw score, duplicates kept. Here each query's ranking is
passed as its own input, so a document's reciprocal ranks are summed.)
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyterrier as pt
from sklearn.feature_extraction.text import TfidfVectorizer

log = logging.getLogger(__name__)

# Settings of proactive-icl-query-pair (src/models/models.py and src/main.py).
LLM_MODEL = "llama-3-8b-instruct"
LLM_BASE_URL = "http://api.llm.apps.os.dcs.gla.ac.uk/v1"  # outside UofG network: "http://api.terrier.org/v1"
LLM_API_KEY_ENV = "IDA_LLM_API_KEY"
RBO_PATH = "/mnt/datasets/msMarco/DL-train-passages/msmarco.nnq.rbo.res"
STATIC_QID = "121352"

# exp01 baseline name -> method key of QF.run_qf (the --qf choice in proactive-icl-query-pair)
METHODS = {"zero_shot": "zero", "static_icl": "static", "sliding_window": "sw", "doc2query": "d2q"}
LLM_METHODS = ("zero_shot", "static_icl")


def extract_query(text):
    match = re.search(r'<query>(.*?)</query>', text, re.DOTALL)
    return match.group(1).strip() if match else text.strip()


class QF:
    """Query formulation from a seed set (list of passage texts), as in
    proactive-icl-query-pair. Every method returns that repository's result dict:
    `final_query` (one query) or `final_queries` (several), `time_taken`, and what the
    method used (seed set, query-pair examples, windows)."""

    def __init__(self, llm_model: str = LLM_MODEL, base_url: str = LLM_BASE_URL, api_key: str | None = None,
                 api_key_env: str = LLM_API_KEY_ENV, rbo_path: str = RBO_PATH, client=None):
        self.llm_model = llm_model
        self.base_url = base_url
        self.api_key = api_key
        self.api_key_env = api_key_env
        self.rbo_path = rbo_path
        self._client = client
        self._rbo_df = None
        self.test_cache = None  # doc2query transformer, as named in proactive-icl-query-pair

    @property
    def client(self):
        """OpenAI client for the LLM endpoint, created on first use."""
        if self._client is None:
            from openai import OpenAI

            api_key = self.api_key if self.api_key is not None else os.environ.get(self.api_key_env)
            if api_key is None:
                raise KeyError(f"the LLM baselines need an API key: set the {self.api_key_env} environment variable")
            self._client = OpenAI(base_url=self.base_url, api_key=api_key)
        return self._client

    @property
    def rbo_df(self) -> pd.DataFrame:
        """RBO query-pair DataFrame (static_icl examples), read on first use."""
        if self._rbo_df is None:
            from pec.config import require_path

            rbo_path = str(require_path(self.rbo_path, "RBO query-pair file (static_icl)"))
            self._rbo_df = pd.read_csv(rbo_path, delimiter="\t", header=None)
            self._rbo_df.rename(
                columns={0: "qid", 1: "run_id", 2: "doc_id", 3: "rank", 4: "score", 5: "text"},
                inplace=True,
            )
        return self._rbo_df

    # -- Private Helpers --

    def _chat(self, user: str) -> str:
        """Single LLM call; retries until valid output is provided"""
        request = {
            'model': self.llm_model,
            'messages': [{'role': 'user', 'content': user}],
        }
        response = self.client.chat.completions.create(
            **request
        )
        response_text = response.choices[0].message.content
        return extract_query(response_text)

    def _initial_query(self, seed_set) -> str:
        user = f"""You are a search engine user who is looking for information on a topic and you have come across a set of documents, which you have read.
        When given the documents, think about what could be the possible ways in which your information need can evolve so that you can further explore along the broad topic.
        Formulate a single search query, that must be at most 10 terms, that points to related, but most importantly, different information need (e.g., more specific, or more generic)
        Your response must only be the search query in the <query> and </query> tags. No additional explanations or details.
        
        Context: {chr(10).join(seed_set)}"""
        return self._chat(user)

    def _get_query_examples(self, df, qid, sep=0.15, p=1, head=False, tail=False):
        output = df[df['qid']==int(qid)].sort_values('score', ascending=False).copy()
        output['rank'] = range(len(output))
        if head:
            p1 = output["score"].quantile(0.5 + sep)
            p2 = output["score"].quantile(1.0)
        elif tail:
            p1 = output["score"].quantile(0.0)
            p2 = output["score"].quantile(0.5 - sep)
        else:
            p1 = output["score"].quantile(0.5 - sep)
            p2 = output["score"].quantile(0.5 + sep)
        output = output[(output["score"] >= p1) & (output["score"] <= p2)]
        mid = len(output) // 2
        i_p = min(p, mid)
        loc1, loc2 = mid - i_p, mid + i_p
        output = output.iloc[loc1 : loc2]
        return output.head(p), output.tail(p)

    def _qp_info_from_qid(self, qid, p=5, high=False, low=False) -> list[str]:
        pos_df, neg_df = self._get_query_examples(self.rbo_df, qid, p=p, head=high, tail=low)

        def fmt(df, relation):
            parts = []
            for _, row in df.iterrows():
                q1, q2 = row["text"].split("| ")
                parts.append(f'"{q1}" {relation} "{q2}"')
            return "\n".join(parts)

        return [
            fmt(pos_df, "shares a similar information need to"),
            fmt(neg_df, "has a different information need to"),
        ]

    # -- Public query formulation methods --

    def zero_shot(self, seed_set):
        start = time.time()
        query = self._initial_query(seed_set)
        end = time.time()
        result = {
            'seed_set': seed_set,
            'final_query': query,
            'time_taken': end-start
        }
        return result

    def static_icl(self, seed_set, p=1, static_qid='121352'):
        start = time.time()
        qp_info = self._qp_info_from_qid(static_qid, p=p)
        docs = "\n\n".join(seed_set)
        user = f"""You are a helpful query reformulator. Given the context, your task is to reformulate the query.
        For query formulation, you must consider the provided examples of what can be considered good reformulations (topically related yet distinct) as well as those that are bad ones.
        The generated query must be 10 words or fewer, and your response must only be the search query in the <query> and </query> tags. No additional explanations or details.
        
        Provided documents: {docs}


        Example query-pairs:\tGood reformulations:\n  {qp_info[0]}\n\tBad reformulations:\n  {qp_info[1]}"""
        query = self._chat(user)
        end = time.time()
        result = {
            'seed_set': seed_set,
            'static_qid': static_qid,
            'query-pairs': {'positive': qp_info[0], 'negative': qp_info[1]},
            'final_query': query,
            'time_taken': end-start
        }
        return result

    def sw(self, seed_set, top_k=3, window_size=25):
        start = time.time()
        full_text = " ".join(seed_set)
        tokens = full_text.split()
        vectorizer = TfidfVectorizer(
            use_idf=True, smooth_idf=True,
            tokenizer=lambda x: x.split(), lowercase=False
        )
        vectorizer.fit([full_text])
        idf_dict = dict(zip(vectorizer.get_feature_names_out(), vectorizer.idf_))
        windows = [
            (np.mean([idf_dict.get(re.sub(r"[^A-Za-z0-9\s]", "", w), 0.0)
                      for w in tokens[i: i + window_size]]),
             tokens[i: i + window_size])
            for i in range(0, len(tokens), window_size)
        ]
        sorted_windows = sorted(windows, reverse=True)
        queries = [" ".join(w) for _, w in sorted_windows[:top_k]]
        end = time.time()
        result = {
            'seed_set': seed_set,
            'windows': sorted_windows,
            'final_queries': queries,
            'time_taken': end-start
        }
        return result

    def d2q(self, seed_set):
        start = time.time()
        df = pd.DataFrame([{"docno": f"d{i}", "text": t} for i, t in enumerate(seed_set)])
        genquery = self.test_cache(df)
        end = time.time()
        result = {
            'seed_set': seed_set,
            'final_queries': list(genquery['querygen']),
            'time_taken': end-start
        }
        return result

    def run_qf(self, qf_name, seed_set, *args):
        match qf_name:
            case 'd2q':
                import pyterrier_doc2query as ptd2q
                if self.test_cache is None:
                    self.test_cache = ptd2q.Doc2Query(num_samples=1)
                return self.d2q(seed_set)
            case 'sw':
                return self.sw(seed_set)
            case 'zero':
                return self.zero_shot(seed_set)
            case 'static':
                return self.static_icl(seed_set, *args)
            case _:
                raise ValueError(f"Unknown QF Method: {qf_name}")


def final_queries(result: dict) -> list[str]:
    """The method's query or queries, as proactive-icl-query-pair's evaluation reads them."""
    if 'final_query' in result:
        return [result['final_query']]
    if 'final_queries' in result:
        return list(result['final_queries'])
    raise ValueError(f"no final_query or final_queries in {result}")


class QueryFormulation(pt.Transformer):
    """[qid, seed_docnos] -> adds `queries` (tuple of query strings) from one QF method.

    The seed set is the context passages' texts in seed_docnos order (empty texts are
    dropped, as proactive-icl-query-pair's get_context does). When `cache_path` is set,
    every result is appended to that JSON-lines file with its qid and seed_docnos, and a
    context already there is not formulated again (reruns make no new LLM calls).
    """

    def __init__(self, qf: QF, method: str, texts, args: tuple = (), cache_path: str | Path | None = None):
        if method not in METHODS:
            raise ValueError(f"unknown query formulation method {method!r}; choose from {list(METHODS)}")
        self.qf, self.method, self.texts, self.args = qf, method, texts, tuple(args)
        self.cache_path = Path(cache_path) if cache_path is not None else None
        self._cache: dict[tuple, dict] = {}
        if self.cache_path is not None and self.cache_path.exists():
            with open(self.cache_path) as f:
                for line in f:
                    if line.strip():
                        rec = json.loads(line)
                        self._cache[(str(rec["qid"]), tuple(rec["seed_docnos"]))] = rec["result"]

    def _seed_set(self, seed_docnos) -> list[str]:
        seed_set = []
        for docno in seed_docnos:
            text = self.texts.get(docno) if hasattr(self.texts, "get") else self.texts[docno]
            if text and text.strip():
                seed_set.append(text)
        return seed_set

    def formulate(self, qid: str, seed_docnos) -> dict:
        key = (str(qid), tuple(seed_docnos))
        if key not in self._cache:
            result = self.qf.run_qf(METHODS[self.method], self._seed_set(seed_docnos), *self.args)
            self._cache[key] = result
            if self.cache_path is not None:
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.cache_path, "a") as f:
                    record = {"qid": key[0], "seed_docnos": list(key[1]), "qf": METHODS[self.method],
                              "method": self.method, "result": result}
                    if self.method in LLM_METHODS:
                        record["model"] = self.qf.llm_model
                    f.write(json.dumps(record, default=_json_default) + "\n")
        return self._cache[key]

    def transform(self, inp: pd.DataFrame) -> pd.DataFrame:
        pt.validate.columns(inp, includes=["qid", "seed_docnos"])
        queries = [tuple(final_queries(self.formulate(q, s))) for q, s in zip(inp["qid"], inp["seed_docnos"])]
        return inp.assign(queries=queries)


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def rr_fusion(rankings: list[pd.DataFrame], k: int = 60, num_results: int | None = 1000) -> pd.DataFrame:
    """Reciprocal rank fusion of one query's rankings (pyterrier_alpha.fusion.rr_fusion,
    each ranking a separate input): score = sum over rankings of 1 / (k + rank), with
    0-based ranks, so documents found by several queries add up."""
    import pyterrier_alpha as pta

    frames = [r[["qid", "docno", "score"]].assign(query="") for r in rankings]
    return pta.fusion.rr_fusion(*frames, k=k, num_results=num_results).drop(columns="query")


class E5Search(pt.Transformer):
    """[qid, query, seed_docnos] -> FaissRetriever ranking of the E5-encoded query text,
    context passages removed. `query_encoder` adds `query_vec` to a [qid, query] frame
    (e.g. pyterrier-dr's E5 query encoder)."""

    def __init__(self, store, query_encoder, num_results: int = 1000):
        from pec.pipeline import FaissRetriever

        self.encoder = query_encoder
        self.retriever = FaissRetriever(store, num_results)

    def transform(self, inp: pd.DataFrame) -> pd.DataFrame:
        pt.validate.columns(inp, includes=["qid", "query", "seed_docnos"])
        enc = self.encoder(inp[["qid", "query"]])
        sub = inp[["qid", "seed_docnos"]].merge(enc[["qid", "query_vec"]], on="qid")
        return self.retriever(sub)[["qid", "docno", "score", "rank"]]


class MultiQueryRetriever(pt.Transformer):
    """[qid, seed_docnos, queries] -> one ranking per context.

    Every query is searched with `search` (a transformer from [qid, query, seed_docnos] to
    a ranking without the context passages: E5Search, or a
    pec.text_retrievers.ResidualTextRetriever). A context with one query gets that
    ranking; with several queries the rankings are fused by reciprocal rank fusion
    (`rrf_k`), one input per query (see the module docstring). A context with no query
    gets no results."""

    def __init__(self, search, num_results: int = 1000, rrf_k: int = 60):
        self.search, self.num_results, self.rrf_k = search, num_results, rrf_k

    def transform(self, inp: pd.DataFrame) -> pd.DataFrame:
        pt.validate.columns(inp, includes=["qid", "seed_docnos", "queries"])
        # one row per query, under a sub-qid (no whitespace: some backends write queries to TSV files)
        rows = [{"qid": f"{qid}__q{i}", "orig_qid": qid, "query": q, "seed_docnos": seeds}
                for qid, seeds, queries in zip(inp["qid"], inp["seed_docnos"], inp["queries"])
                for i, q in enumerate(q for q in queries if isinstance(q, str) and q.strip())]
        if not rows:
            return pd.DataFrame(columns=["qid", "docno", "score", "rank"])
        sub = pd.DataFrame(rows)
        res = self.search(sub[["qid", "query", "seed_docnos"]])
        if len(res) == 0:
            return pd.DataFrame(columns=["qid", "docno", "score", "rank"])
        res = res[["qid", "docno", "score"]].merge(sub[["qid", "orig_qid"]], on="qid")
        fused = []
        for qid, run in res.groupby("orig_qid", sort=False):
            rankings = [r.assign(qid=qid) for _, r in run.drop(columns="orig_qid").groupby("qid", sort=False)]
            if len(rankings) == 1:
                fused.append(rankings[0])
            else:
                fused.append(rr_fusion(rankings, k=self.rrf_k, num_results=self.num_results)[["qid", "docno", "score"]])
        out = pd.concat(fused, ignore_index=True)
        out = out.sort_values(["qid", "score"], ascending=[True, False]).reset_index(drop=True)
        return pt.model.add_ranks(out)


def TextQueryRetriever(store, query_encoder, num_results: int = 1000, rrf_k: int = 60) -> MultiQueryRetriever:
    """MultiQueryRetriever over the E5 index."""
    return MultiQueryRetriever(E5Search(store, query_encoder, num_results), num_results, rrf_k)


def run_name(method: str, retriever: str) -> str:
    """exp01 run name: the method alone for the E5 index, <method>@<retriever> otherwise."""
    return method if retriever == "e5" else f"{method}@{retriever}"


def query_formulation_pipelines(store, c: dict, texts, query_encoder, num_results: int = 1000,
                                cache_dir: str | Path | None = None, qf: QF | None = None,
                                text_retrievers: dict | None = None) -> dict:
    """exp01 baseline pipelines: for every method in c["methods"] and every retriever in
    c["retrievers"] (default: e5 only), QueryFormulation >> MultiQueryRetriever. `c` is
    exp01.baseline.query_formulation; `text_retrievers` maps the names other than "e5" to
    their ResidualTextRetriever (pec.text_retrievers.build_retrievers). Each method is
    formulated once per context and its queries are shared by every retriever."""
    methods = list(c.get("methods") or [])
    if not methods:
        return {}
    retrievers = list(c.get("retrievers") or ["e5"])
    text_retrievers = text_retrievers or {}
    missing = [r for r in retrievers if r != "e5" and r not in text_retrievers]
    if missing:
        raise ValueError(f"no text retriever built for {missing}")
    llm = c.get("llm") or {}
    static = c.get("static_icl") or {}
    qf = qf or QF(llm_model=llm.get("model", LLM_MODEL), base_url=llm.get("base_url", LLM_BASE_URL),
                  api_key_env=llm.get("api_key_env", LLM_API_KEY_ENV), rbo_path=static.get("rbo_path", RBO_PATH))
    # positional arguments QF.run_qf passes on, as proactive-icl-query-pair's main.py does
    # (static: p, which defaults to 1 there; static_qid keeps its default 121352 unless set)
    args = {"static_icl": (int(static.get("p", 1)), str(static.get("static_qid", STATIC_QID)))}
    rrf_k = c.get("rrf_k", 60)
    searches = {r: (E5Search(store, query_encoder, num_results) if r == "e5" else text_retrievers[r])
                for r in retrievers}
    pipes = {}
    for method in methods:
        cache = Path(cache_dir) / f"{method}.jsonl" if cache_dir is not None else None
        formulate = QueryFormulation(qf, method, texts, args.get(method, ()), cache)
        for r, search in searches.items():
            pipes[run_name(method, r)] = formulate >> MultiQueryRetriever(search, num_results, rrf_k)
    return pipes
