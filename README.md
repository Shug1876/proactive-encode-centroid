# proactive-encode-centroid

Given a set of documents that are only **partially relevant** to some unknown information
need *q*, find the **perfectly relevant** documents for *q*, with **zero-training
baselines** only: the context **centroid** in the E5 document space and its competitors
(per-document fusion, multi-query and inverse context MMR, BM25 query-by-document and a
BM25 centroid, and LLM / text query formulation), each compared with the true query
(the **Oracle**).

This repository holds two experiments and the scripts that plot their stored results:

- **Experiment 1** (`scripts/exp01_dl19.py`): every baseline over a sweep of context sizes
  m, as a share of its own retriever's Oracle.
- **Experiment 2** (`scripts/exp02_noise.py`): the centroid and the BM25 baselines on
  contexts in which a share α of the documents are judged non-relevant.
- **Plotting** (`scripts/plot_results.py`): every figure redrawn from the stored tables,
  with the spread over repeated context draws, plus paper-ready one-column PDFs.

It was split out of
[proactive-q-encoder](https://github.com/Shug1876/proactive-q-encoder), which also holds
the trained graph model, its evaluation and Experiment 1's descriptive analyses (t-SNE,
distances, reachability, context similarity, features). The package here is `pec`; the
code, configs and stored-result formats of the baselines are unchanged, so results
written by proactive-q-encoder's `exp01_<dataset>/baseline/` and `exp02_<dataset>/` folders
are read (and reused) as they are.

Relevance grades follow each collection's own scale. Which grade is the context and which
is the target is set per dataset (`datasets.<name>.relevance` in `configs/paths.yaml`):

| Grade | TREC DL (2019, 2020) | TREC-COVID |
|---|---|---|
| 0 | non-relevant | not relevant |
| 1 | partially relevant (**the context**) | partially relevant (**the context**) |
| 2 | highly relevant | relevant (**the target**) |
| 3 | perfectly relevant (**the target**) | (none) |

Evaluation views: TREC DL is scored on grade ≥ 2 (grades 2 and 3, graded) and grade = 3;
TREC-COVID on grade = 2 only (grade-1 documents are the context, not targets).

## Data and indexes

| Dataset | Queries / qrels | Corpus | Index (in `configs/paths.yaml`) |
|---|---|---|---|
| `dl19` | TREC-DL19 passage (`irds:msmarco-passage/trec-dl-2019/judged`) | MS MARCO v1 passage | `indexes.msmarco_v1` |
| `dl20` | TREC-DL20 passage (`irds:msmarco-passage/trec-dl-2020/judged`) | MS MARCO v1 passage | `indexes.msmarco_v1` (shared with DL19) |
| `trec_covid` | TREC-COVID (`irds:beir/trec-covid`, topic field `text`) | BEIR CORD-19 subset, 171k title + abstract | `indexes.trec_covid` (separate) |

- **No code here builds an index.** Each index entry points at an existing FAISS file plus
  a docno list (`.txt`, one per line, or a pyterrier-dr `.npids` file) mapping index rows
  to passage ids.
- **All indexes must use the same E5 model** (`e5_model`, default `intfloat/e5-base-v2`,
  documents encoded with the `passage: ` prefix and normalised). Flat and HNSW indexes give
  exact stored vectors; IVF gets a direct map on first use; PQ vectors are approximate.
- **BM25 runs need a PISA index** per corpus (`text_retrievers.<index>.bm25`; TREC-COVID
  uses the `pyterrier/trec-covid.pisa` HuggingFace artifact). The other text retrievers of
  the query formulation baselines (monoT5, ColBERT, SPLADE) are configured next to it.
- **Building the TREC-COVID index** (once, outside this repo). Encode title + abstract;
  at 171k documents a flat index is fine:

  ```python
  import faiss, pyterrier as pt, pyterrier_dr

  index = pyterrier_dr.FlexIndex("/data/indexes/e5-base-v2.beir-trec-covid.flex")
  title_abstract = pt.apply.text(lambda r: f"{r['title']} {r['text']}".strip())
  (title_abstract >> pyterrier_dr.E5.base().doc_encoder() >> index.indexer()).index(
      pt.get_dataset("irds:beir/trec-covid").get_corpus_iter())
  faiss.write_index(index.faiss_flat_retriever().faiss_index, "/data/indexes/e5-base-v2.beir-trec-covid.faiss")
  ```

  Then set `indexes.trec_covid.faiss` to that file and `docnos` to the FlexIndex's
  `docnos.npids`.

## Setup

```bash
sh requirements.sh
```

This installs this package (`pip install -e ".[dev]"`) with python-terrier, pyterrier-dr
(E5 encoder, MMR, npids), ir_datasets, ir_measures, faiss-cpu, openai, and the plugins
the baselines need: pyterrier-pisa (BM25, SPLADE), pyterrier_doc2query, pyterrier_t5
(monoT5) and pyterrier_splade. `pip install -r requirements.txt` is the GPU variant
(`faiss-gpu`). Then point every path in `configs/paths.yaml` at your own indexes. The LLM
baselines of Experiment 1 read their API key from `IDA_LLM_API_KEY` (see
[Query formulation baselines](#query-formulation-baselines)).

## Running

```bash
python scripts/exp01_dl19.py --config configs/exp01_dl19.yaml   # Experiment 1, every dataset
python scripts/exp01_dl19.py --datasets dl20 trec_covid         # a subset of datasets
python scripts/exp02_noise.py --config configs/exp02_noise.yaml # Experiment 2, every dataset
python scripts/plot_results.py                                  # redraw every figure from the stored tables
pytest                                                          # synthetic-data tests
```

| Script | Main options |
|---|---|
| `exp01_dl19.py` | `--datasets`, `--qf-groups llm text`, `--qf-only`, `--bm25-ablation`, `--no-fixed-oracle`, `--titles`, `--rerun [NAME ...]` |
| `exp02_noise.py` | `--datasets`, `--alphas 0 0.2 ...`, `--titles`, `--rerun` |
| `plot_results.py` | see [Plotting stored results](#plotting-stored-results-scriptsplot_resultspy) |

Outputs go to `outputs/<experiment>_<dataset>/` (gitignored). Every figure has a CSV of its
numbers next to it.

## Experiment 1: baselines over the context size (`pec.experiments.exp01`)

Runs for every dataset in `exp01.datasets` (DL19, DL20, TREC-COVID) into
`outputs/exp01_<dataset>/baseline/`; `exp01.overrides.<dataset>` changes settings for one
dataset. "Context" and "target" below are each dataset's context and target grades (DL: 1
and 3, TREC-COVID: 1 and 2). With several datasets, `outputs/exp01_comparison/` holds every
dataset's best m (`best_context_size.csv`) and the centroid's nDCG@10 share of the Oracle
against m on each dataset's target view (`centroid_ndcg10_by_dataset.png`).

| Run | Queries / ranking from the context | Section |
|---|---|---|
| `centroid` | the normalised mean of the context's E5 vectors, searched in the FAISS index | |
| `seed_rrf` | every context passage's E5 vector searched separately, fused by reciprocal rank fusion | |
| `seed_mmr@λ` | every context passage a query, re-ranked by MMR | [Multi-query MMR](#multi-query-mmr-seed_mmrλ) |
| `seed_inv_mmr@λ` | the same pool, scored by inverse context MMR | [Inverse context MMR](#inverse-context-mmr-seed_inv_mmrλ) |
| `bm25_DaaQ` | every context passage's text a BM25 query, fused by RRF | [BM25 + RRF](#bm25--rrf-bm25_daaq) |
| `bm25_centroid` | the centroid in BM25's term space, one weighted BM25 query | [BM25 centroid](#bm25-centroid-bm25_centroid) |
| `zero_shot`, `static_icl`, `sliding_window`, `doc2query` (and `<method>@<retriever>`) | text queries written from the context | [Query formulation baselines](#query-formulation-baselines) |
| `true_query`, `true_query@<retriever>` | the Oracle: the real query, with E5 or a text retriever | |

Context documents are removed from every ranking (the user already has them).

### Context-size sweep

The aim is to get as close as possible to the true-query upper bound with a small
context. Experiment 1 therefore runs every system on contexts of k ∈ {1, 3, 5, 7, 10}
grade-1 passages, with `repeats` random draws per k. ("all", every
grade-1 passage, is disabled for every approach for now; adding `all` back to
`context_sizes` runs it, except on TREC-COVID.)

- **Nested draws.** For each query and draw, its grade-1 passages are shuffled once and
  the first k are the context, so within a draw the k=3 context sits inside the k=5 one.
  Differences between sizes come from size, not from a different draw. Draws are fixed by
  `random_seed`, so every run (and Experiment 2's α = 0 context) sees identical contexts.
- **No padding.** A query with fewer than k grade-1 passages is left out at that k (the
  number of queries per k is reported). "all" uses every grade-1 passage and has one draw.
  TREC-COVID topics have hundreds of grade-1 documents, so "all" is not run there.
- **Share of the upper bound.** Because the query set changes with k, every metric also
  gets a `<metric>_pct_true_query` column: the run's score as a % of the true query's
  score on the same queries and the same context (context passages are removed from both
  rankings).
- **Best k.** `best_context_size.csv` gives, per run, view and metric, the k other than
  "all" with the highest mean share of the true query (ties go to the smaller k), next to
  the scores at "all".

Outputs: `metrics_by_context_size.csv` (per draw), `summary_by_context_size.csv` (mean,
std, queries, draws), `best_context_size.csv`, `metrics_perquery.csv`, and figures of the
share of the Oracle against the context size m, one per view (`ge2`, `eq3`; `eq2` for
TREC-COVID):

| Figure | Lines |
|---|---|
| `context_sweep_<view>.png` | every E5 method: centroid, `seed_rrf`, zero-shot, static ICL, sliding window, doc2query (more than four lines: a marker per line) |
| `context_sweep_bm25_<view>.png` | `bm25_DaaQ`, `bm25_centroid` and their ablations `*_nobm25` |
| `context_sweep_mmr_<view>.png` | centroid and each MMR λ |
| `context_sweep_inv_mmr_<view>.png` | centroid and each inverse context MMR λ |
| `context_sweep_qf_<retriever>_<view>.png` | every query formulation method through one text retriever |
| `context_sweep_overview_<metric>_<view>.png` | one panel per figure above that has results, the centroid as a grey reference |

**The Oracle in the figures.** The true query is called **Oracle**, the context size **m**,
and every run is compared with its own retriever's Oracle: E5 runs with the E5 true query
(`true_query`), the BM25 runs with the true query text searched with BM25
(`true_query@bm25`, run whenever a BM25 method runs), `<method>@<retriever>` with
`true_query@<retriever>`. With `baseline.fixed_oracle: true` (the default) every m is
compared with the Oracle's mean at m = `oracle_context_size` (5), so the Oracle is one
flat line and the curves show how each m compares with the same Oracle;
`fixed_oracle: false` (command line `--no-fixed-oracle`) compares each m with the Oracle
at that m. Only the figures change: the tables' `<metric>_pct_true_query` columns, and
best k, still compare with the E5 true query at the same m.

**Figure settings** (every figure: the experiments' own and everything `plot_results.py`
draws). They live at the top of `src/pec/analysis/plots.py`:

| Setting | Default | Effect |
|---|---|---|
| `MARKER_SIZE` | 3.5 | size, in points, of the markers on every line |
| `ORACLE_LINES` | `False` | draw the Oracle as reference lines (each retriever's raw-score Oracle line, and the 100% line of the "% of Oracle" panels). Off, the % panels are still % of the Oracle. `plot_results.py --oracle-lines` turns them on for one call |

Legend names come from `DISPLAY_NAMES` in `src/pec/analysis/oracle.py`; only the figures
change, while the tables, configs and command-line options keep the run names:

| Run | In the figures |
|---|---|
| `centroid` | E5_centroid |
| `zero_shot` | QF-Zero |
| `static_icl` | QF-Fixed |
| `sliding_window` | QF-SW |
| `doc2query` | QF-D2Q |

`<run>@<retriever>` follows its run (`doc2query@bm25` is QF-D2Q@bm25).

**No titles.** Experiment 1's figures (and `exp01_comparison`) have no titles unless `exp01.plot_titles: true` (command line `--titles`), as Experiment 2's. Where a panel
title named something, it moves: the context-sweep y axes read e.g. "nDCG@10 (% of
Oracle)", and the overview's method families head each panel's legend. Panel titles
marked with `keep_title` stay (they are labels, not titles).

**The same y range everywhere.** `y_limits` in `configs/plot_results.yaml` (included by the
Experiment 1 and Experiment 2 configs) fixes the y axis of every result figure: score
panels 0–1 (`raw`), % of the Oracle panels 0–140 (`pct`). That covers the context sweeps,
the overview, the noise sweeps, the comparison figures and everything `plot_results.py` draws (`--ylim-raw` / `--ylim-pct LO
HI` override it, `--no-fixed-ylim` turns it off). Values outside the range are clipped and
logged; remove a key, or set `y_limits: null`, for automatic ranges.

**Pinned Oracle values.** `configs/oracle_values.yaml` (included by the Experiment 1 and 2
configs and read by `plot_results.py`) pins the Oracle scores the figures use: nDCG@10 at
m = 5 for the E5 Oracle (`true_query`) and the BM25 Oracle (`true_query@bm25`) on DL19 and
DL20 (`grade=3`, `grade>=2`) and TREC-COVID (`grade=2`), keyed metric -> dataset -> Oracle
run -> view. Wherever a value is given, every figure divides by it (and, with the Oracle
lines on, draws the Oracle line at it), instead of the Oracle computed from the stored results; metrics, datasets or views
not listed keep the computed Oracle (fixed at m = 5 as above), and a pinned value that
matches no stored result is logged. Toggles: `exp01.baseline.pin_oracle`,
`exp02.pin_oracle` (both default true) and `plot_results.py --no-pinned-oracle` (or another
file with `--oracle-values`). The saved tables of both experiments keep the computed Oracle.

"all" stays in the tables if stored but is never plotted.

**`seed_fusion` (max-cosine fusion) has been removed.** Its stored results are deleted
from the `baseline/` tables the next time Experiment 1 runs, and it is never plotted
(`pec.evaluate.RETIRED_RUNS`). The max-cosine mode of `SeedFusionRetriever` remains, as
the candidate pool of the two MMR baselines.

**Stored results are reused.** Every run's results accumulate in the `baseline/` tables
(`pec.evaluate.cached_sweep`): a run that already has results for a context size and
repeat is not run again, and runs made in separate invocations end up in the same
tables and figures. The figures are built from everything stored, not just the runs of
the latest invocation. Draws depend only on the qrels and `random_seed`, so stored results
stay valid while `baseline/sweep_settings.json` (seed, result depth, metrics, views)
matches; if the settings change, nothing stored is reused. To recompute:
`baseline.rerun: true` (or `--rerun`) reruns everything, `--rerun seed_rrf zero_shot`
only the runs named.

### Query formulation baselines

The `zero_shot`, `static_icl`, `sliding_window` and `doc2query` methods of
[proactive-icl-query-pair](https://github.com/Shug1876/proactive-icl-query-pair)
(`src/models/models.py`) live in `pec.query_formulation`. Their code is copied unchanged:
the prompts, the LLM request, the `<query></query>` extraction, the static query pairs
and the window and doc2query settings. Each turns a context into text queries:

| Run | Queries from the context passages' text | Settings (`baseline.query_formulation`) |
|---|---|---|
| `zero_shot` | one query written by an LLM | `llm.model` `llama-3-8b-instruct`, OpenAI-compatible `llm.base_url`, key in `$IDA_LLM_API_KEY` |
| `static_icl` | one query written by an LLM that is also shown good and bad query pairs of one fixed MS MARCO training query | `static_icl.p` 1 pair of each, `static_icl.static_qid` 121352, from the RBO query-pair file `static_icl.rbo_path` |
| `sliding_window` | the 3 windows of 25 whitespace tokens with the highest mean IDF | as in proactive-icl-query-pair |
| `doc2query` | one doc2query-T5 query per context passage (`pyterrier_doc2query`, `num_samples=1`) | as in proactive-icl-query-pair |

- **Same contexts.** They run on the same context draws as every other baseline (same
  k, repeats and `random_seed`), so their shares of the true query compare directly. The
  seed set is the context passages' text in docno order, skipping empty passages.
  `text_fields` picks the text: by default the dataset's own fields (TREC-COVID: title
  + abstract, as in the E5 index); `[text]` gives the abstract alone, as
  proactive-icl-query-pair used.
- **Same index.** Each query is encoded with `e5_model` (`query: ` prefix) and searched
  in the dataset's FAISS index, context passages removed, like the true query (runs
  named after the method).
- **proactive-icl-query-pair's retrievers.** The same queries also go through the
  retrievers that repository evaluated them with (`pec.text_retrievers`, its
  `get_retr`), each run named `<method>@<retriever>`:

  | `retrievers` | Pipeline |
  |---|---|
  | `bm25` | BM25 over a PISA index |
  | `monot5` | BM25 top 50 (`monot5_depth`) >> passage text >> `MonoT5ReRanker()` |
  | `colbert` | TCT-ColBERT (`castorini/tct_colbert-v2-hnp-msmarco`) over a FlexIndex (MS MARCO) or NumpyIndex (TREC-COVID) |
  | `splade` | SPLADE (`naver/splade-cocondenser-ensembledistil`) over a quantized, unstemmed PISA index |

  The indexes are that repository's, set under `text_retrievers` in
  `configs/paths.yaml` (keyed like `indexes`, so DL19 and DL20 share the MS MARCO
  ones; TREC-COVID's BM25 is the `pyterrier/trec-covid.pisa` HuggingFace artifact, as
  there). Context passages are removed from every ranking, and each retriever searches
  deep enough to keep `num_results` after that (monoT5 keeps what is left of its 50).
  proactive-icl-query-pair cut ColBERT and SPLADE at 10 for nDCG@10; here every
  retriever returns `num_results` so Recall@k is measured too. With `true_query: true`
  the true query text goes through each retriever as well (`true_query@<retriever>`),
  the upper bound on that retriever; the `_pct_true_query` columns stay relative to the
  E5 true query. Each method is formulated once per context, whatever the number of
  retrievers. Figures: `context_sweep_qf_<retriever>_<view>.png`.
- **Fusion.** Where a
  method gives several queries (`sliding_window`, `doc2query`), their rankings are fused
  by reciprocal rank fusion (`rrf_k` 60, pyterrier-alpha's `rr_fusion` with one input
  per query). proactive-icl-query-pair called `rr_fusion` on a single frame of all the
  queries' hits, which fuses nothing (hits are re-sorted by raw score, duplicates kept),
  so here the reciprocal ranks of a document are actually summed.
- **Outputs.** Each method's result for every context (the seed set, the queries, the
  query pairs shown, the windows, the time taken) is appended to
  `baseline/query_formulation/<method>.jsonl`. A rerun reuses those results, so no LLM
  call is repeated; delete the file to formulate again.
- **Which and where.** `methods: []` turns them off; `retrievers: [e5]` keeps only the
  E5 runs. `context_sizes` (default 1, 3, 5, 7, 10) limits the sizes these methods and
  `true_query@<retriever>` run at. The LLM sees the whole context, so very large contexts
  (such as "all" on DL) must fit the model's context window.
- **Running the groups separately.** `groups` picks `llm` (`zero_shot`, `static_icl`)
  and/or `text` (`sliding_window`, `doc2query`); `--qf-only` (`other_baselines: false`)
  skips the other baselines. Results accumulate, so for example

  ```bash
  python scripts/exp01_dl19.py --qf-groups llm
  python scripts/exp01_dl19.py --qf-groups text --qf-only
  ```

  ends with both groups and every other baseline in one set of tables and figures.

### Multi-query MMR (`seed_mmr@λ`)

Each context document acts as a query. Candidates are the union of the context documents'
top-1,000 neighbours (context removed); a candidate's relevance is its highest cosine to
any context document. pyterrier-dr's `MmrScorer` re-ranks the `pool` (2,000) most relevant
candidates, picking next the document that maximises λ · relevance − (1 − λ) · (highest
cosine to the documents already picked), and the top 1,000 are kept. λ ∈ {0.5, 0.7, 0.9}
(`baseline.mmr.lambdas`); λ = 1 would be max seed fusion exactly.

### Inverse context MMR (`seed_inv_mmr@λ`)

The same candidate pool as `seed_mmr` (the 2,000 candidates most similar to any context
document). For a candidate d, q is the context document it is most similar to, and

  S(d) = −λ · S_R(d, q) + (1 − λ) · S_c(d, C)

where S_R(d, q) = cos(d, q) and S_c(d, C) is the mean cosine between d and the other
context documents (0 when the context is q alone, i.e. k = 1). C is fixed, so each
candidate is scored once and the pool sorted (no greedy selection). λ ∈ {0.5, 0.7, 0.9}
(`baseline.inv_mmr`). With λ ≥ 0.5 the −λ · S_R term dominates, so the pool's documents
least similar to their nearest context document tend to come first, and the result
depends on the pool size.

### BM25 + RRF (`bm25_DaaQ`)

Each context passage's text is a BM25 query against the dataset's BM25 PISA index
(`text_retrievers.<index>.bm25` in `paths.yaml`). Each query keeps its top `depth` (1,000)
documents with the context removed, and the per-passage rankings are fused by reciprocal
rank fusion (`rrf_k` 60, pyterrier-alpha's `rr_fusion`, one input per passage, as for the
multi-query query formulation methods). The passage text is the same text the query
formulation methods get (title + abstract on TREC-COVID). `baseline.bm25_rrf.enabled:
false` turns it off.

This run was called `seed_bm25_rrf` (its ablation `seed_bm25_rrf_nobm25`); results stored under
those names are read as `bm25_DaaQ` / `bm25_DaaQ_nobm25` (`pec.evaluate.RENAMED_RUNS`), so nothing
needs rerunning.

### BM25 centroid (`bm25_centroid`)

The centroid idea in BM25's term space, as one weighted BM25 query against the same PISA
index (nothing is indexed). For each context:

1. Each passage's text is turned into the index's terms with PISA's own tokeniser, the
   index's stopwords removed and its stemmer applied (`PisaIndex.tokenize`).
2. Each passage becomes a sparse vector of term counts divided by its length, so long
   passages do not dominate (the counterpart of E5's unit vectors).
3. The vectors are averaged into the centroid, scaled so the largest weight is 1. With
   `bm25_centroid.all_terms: true` (the config's current setting) it keeps every term (a
   longer, slower query); with `false` only its `terms` (100) highest weights.
4. The centroid is searched as `query_toks` with `PisaIndex.bm25(query_weighted=True)`:
   a document scores Σ_t w_t · BM25(t, d), the centroid's dot product with the
   document's BM25 vector (BM25 supplies the idf), and the context is removed.

It runs at every context size and is plotted in `context_sweep_bm25_<view>.png`
(`baseline.bm25_centroid.enabled: false` turns it off). It is expected to do worse than
the E5 centroid:

- **Vocabulary mismatch.** BM25 only credits exact (stemmed) terms. Grade-1 passages are
  only related to the need, so their shared words describe the topic area; perfectly
  relevant passages that word the answer differently get nothing, while E5 places
  paraphrases close together.
- **Averaging does not cancel.** Unit E5 vectors pointing different ways partly cancel,
  keeping the shared semantic direction. Term weights are all non-negative, so the BM25
  centroid is a weighted union of every passage's vocabulary, idiosyncrasies included, and
  BM25 has no notion that two terms are related.
- **Long-query topic drift.** A 100-term query adds up many independent matches, so long,
  generic, topical documents rise and the few discriminative terms are diluted, the known
  failure of Rocchio / pseudo-relevance feedback when the feedback documents are only
  partly relevant (and worse with Experiment 2's noise).
- **Frequency, not usefulness.** Weights come from how often a term occurs in the context;
  common topical words dominate, BM25's idf only partly corrects this, and pruning can cut
  the rare decisive terms.
- **Learned versus hand-built geometry.** E5 is trained contrastively (including MS MARCO)
  so related passages sit together; BM25's k1, b and saturation were designed for short
  keyword queries, not 100-term weighted ones.

On TREC-COVID, where BM25 is unusually strong (`true_query@bm25` beats E5's true query),
the gap may be small or reversed.

### BM25 ablation without term weighting (`bm25_DaaQ_nobm25`, `bm25_centroid_nobm25`)

`baseline.bm25_binary_ablation: true` (command line `--bm25-ablation`; off by default)
runs `bm25_DaaQ` and `bm25_centroid` a second time with BM25's term weighting turned
off: BM25 with k1 = 0, so a matching term scores its idf whatever its frequency in the
document and the document's length (binary term matching; PISA has no scorer without
idf). They are separate runs, `bm25_DaaQ_nobm25` and `bm25_centroid_nobm25`, in the
BM25 figure next to the weighted ones, and are compared with the same BM25 Oracle.

## Experiment 2: non-relevant documents in the context (`pec.experiments.exp02`)

A user's context is rarely all partially relevant: some of what they looked at does not
help. Experiment 2 measures what that costs. Every context has m = `context_size` = 5
documents, of which a share **α** (`alphas`) are non-relevant and the rest partially
relevant: α ∈ {0, 0.2, 0.4, 0.6, 0.8}, i.e. 0, 1, 2, 3 and 4 non-relevant documents
(α · m must be whole; `--alphas` overrides the list).

- **Non-relevant = the query's own judged grade-0 documents, only.** They were pooled
  from real systems, so they look on-topic, which is the realistic case. Documents are
  never sampled at random from the collection. Partially relevant is the dataset's
  context grade (grade 1 on TREC DL and TREC-COVID).
- **The same queries for every α.** Only queries with at least 5 partially relevant and
  4 judged grade-0 documents (and a target document) are used, for every α
  (`summary.json`: `queries_used`).
- **Nested draws.** For each query and draw, both kinds of document are shuffled once and
  each α takes the first n of each list, so going from α = 0.2 to 0.4 swaps a partially
  relevant document for a non-relevant one. The partially relevant shuffle is
  Experiment 1's (same `random_seed`), so the α = 0 context is Experiment 1's m = 5
  context.
- **Runs** (`exp02.runs`): the E5 `centroid`, and Experiment 1's BM25 methods
  `bm25_DaaQ` and `bm25_centroid` (they need `text_retrievers.<index>.bm25`). Each is
  compared with its own retriever's Oracle, run on the same contexts: the E5 true query
  (`true_query`) for the centroid, the true query text with BM25 (`true_query@bm25`) for
  the BM25 runs (`<metric>_pct_true_query` in the tables is that share). Context documents
  of both kinds are removed from every ranking. Metrics and views are Experiment 1's.

Outputs in `outputs/exp02_<dataset>/`: `contexts.csv` (every context, with its α,
non-relevant and partially relevant documents and their counts), `metrics_by_alpha.csv`,
`metrics_perquery.csv`, `summary_by_alpha.csv`, `noise_sweep_<view>.png` (top row: each
metric (with the Oracle lines on, the E5 Oracle dashed and the BM25 Oracle dotted); bottom row: % of each run's
own Oracle; x axis α from 0 to 0.8)
and `summary.json`. With several datasets, `outputs/exp02_comparison/centroid_ndcg10_by_dataset.png`
has the centroid's nDCG@10 share of the reference per dataset.

**Figures.** The true-query reference is called **Oracle** in legends and axes
(`exp02.reference_label`): "Oracle (E5)" and "Oracle (BM25)" for the two raw-score lines (when drawn),
and the y axis of a share reads e.g. "nDCG@10 (% of Oracle (own retriever))" ("% of
Oracle" with the centroid alone). The x axis reads "α (share of non-relevant documents),
m = 5", and figures have no titles unless
`exp02.plot_titles: true` (`--titles`). Results stored under the earlier non-relevant :
partially relevant mixes (`metrics_by_mix.csv`) and at m = 10 are no longer read; rerun
Experiment 2.
Stored results are reused; `--rerun` (or `exp02.rerun: true`) recomputes them.

## Plotting stored results (`scripts/plot_results.py`)

Redraws figures from the per-draw tables the experiments have already written, without
retrieving or evaluating anything. Every line is the mean over the repeated runs (random
context draws) with a band for their spread. Figures go to
`outputs/plots/<experiment>_<dataset>/`, each next to a CSV of the plotted numbers (mean,
band low / high, std, repeats).

```bash
python scripts/plot_results.py --list        # experiments, datasets, models, views, metrics, conditions, repeats
python scripts/plot_results.py               # every plot, the runs of configs/plot_results.yaml (+ their Oracles)
python scripts/plot_results.py --all-models  # every model, in its figure group
python scripts/plot_results.py --experiments exp01 --datasets dl19 --models centroid seed_rrf "seed_mmr@*"
python scripts/plot_results.py --experiments exp02 --variance ci95 --show-repeats
python scripts/plot_results.py --plots paper_row --paper-metric R@10 --column-width 3.5
```

| Plot (`--plots`) | Reads | Figure |
|---|---|---|
| `sweep` | `exp01_<ds>/baseline/metrics_by_context_size.csv`, `exp02_<ds>/metrics_by_alpha.csv` | `sweep_<group>_<view>`: a panel per metric, a row per value (raw, % of true query) against k or α, a line per model |
| `datasets` | the `sweep` tables of each experiment | `<experiment>_comparison/datasets_<model>`: one line per dataset on its target view. For `exp02`, every panel is also saved on its own as `datasets_<model>_<metric>_<raw\|pct>` (e.g. `datasets_centroid_ndcg10_pct.png`) |
| `paper_row` (only when asked for) | the `sweep` tables of `exp01` and `exp02` | `paper/<experiment>_<dataset>.pdf` and `paper/<experiment>_legend.pdf` (default `--format pdf`): per experiment, three plots (DL19, DL20, TREC-COVID) sized to sit side by side in one column of a two-column paper, the y axis on the first only, and a legend to centre below them; `paper/<experiment>_widths.txt` has the LaTeX subfigure widths. Options: `--paper-experiments`, `--paper-datasets`, `--paper-views`, `--paper-metric`, `--paper-value`, `--column-width`, `--paper-height`, `--paper-font-size`, `--paper-legend-columns` |

| Toggle | Effect |
|---|---|
| `--experiments`, `--datasets` | which stored results (default: all found) |
| (default) | the runs of `configs/plot_results.yaml`, else each experiment's built-in models, in `main` figures (eight lines each) with their Oracles: `exp01` every E5 method (centroid, `seed_rrf`, `zero_shot`, `static_icl`, `sliding_window`, `doc2query`), `exp02` `centroid`, `bm25_DaaQ`, `bm25_centroid`; the `datasets` plot makes one figure for each of them |
| `--runs-config FILE`, `--no-runs-config` | which runs each experiment shows: `configs/plot_results.yaml` by default (`plot_results.runs`: experiment -> runs, patterns allowed, in that order; an experiment it leaves out keeps the built-in default) |
| `--models PATTERN ...`, `--exclude-models` | models, shell patterns allowed (`"seed_mmr@*"`); at most eight per figure (more than four: a marker per line), more are split over `selected`, `selected_2`, ... |
| `--all-models` | every model, in Experiment 1's figure groups and an `other` group |
| `--groups` | without `--models`: every model of these groups (`main` = the E5 methods, `bm25`, `mmr`, `inv_mmr`, `qf_<retriever>`, `other`) |
| `--views`, `--metrics`, `--values raw pct` | views (`grade=3` or `eq3`), metrics, figure rows |
| `--variance std\|sem\|ci95\|minmax\|none` | the band over repeats (default ±1 std; `ci95` uses Student's t) |
| `--show-repeats` | also draw every repeat as a thin line |
| `--conditions`, `--include-all` | x values to keep (context sizes or α); plot the stored "all" size |
| `--reference` | the upper-bound run behind the percentages |
| `--oracle-lines` / `--no-oracle-lines` (`--no-reference`) | draw the Oracle reference lines or not (default: `ORACLE_LINES` in `src/pec/analysis/plots.py`, off) |
| `--titles` / `--no-titles`, `--reference-label` | figure titles (default: off) and the reference's name in legends and axes (default: `Oracle`, with `context size m` on `exp01`'s x axis) |
| `--oracle-values FILE`, `--no-pinned-oracle` | the pinned Oracle scores (default `configs/oracle_values.yaml`; see *Pinned Oracle values*), or the Oracle computed from the stored results |
| `--ylim-raw LO HI`, `--ylim-pct LO HI`, `--no-fixed-ylim` | the y range of every score / % of Oracle panel (default: `y_limits` of the runs file, 0–1 and 0–140), or automatic ranges |
| `--fixed-oracle` / `--no-fixed-oracle`, `--oracle-size` | compare every context size with the Oracle at `--oracle-size` (5): flat Oracle lines (default: on for `exp01`), or each size with the Oracle at that size |
| `--format png\|pdf\|svg`, `--root`, `--out` | output format and folders |

Retired runs (`seed_fusion`) are never plotted, and runs carry their figure names (see
*Figure settings*). The % values are recomputed from the stored scores against each run's
own retriever's Oracle (`true_query` for E5 runs, `true_query@bm25` for the BM25 runs,
...), on the same view, condition and repeat, or at the fixed Oracle size. With the Oracle
lines on, the raw-score rows draw each retriever's Oracle (E5 dashed, BM25 dotted).

## Tests

```bash
pytest
```

The tests (73) run on a small synthetic corpus in an in-memory FAISS index, with fake BM25,
LLM, doc2query and text retrievers, so they need no index, download or API key: the config,
data, contexts and noise draws, the index store, every baseline pipeline, the
stored-result cache, both experiments end to end (with figures, the Oracle toggles and
pinned values) and every `plot_results.py` figure and option.

## Layout

```
configs/      paths.yaml (indexes, text retrievers, datasets), exp01_dl19.yaml, exp02_noise.yaml,
              oracle_values.yaml (pinned Oracle scores), plot_results.yaml (runs shown, y ranges)
scripts/      exp01_dl19.py, exp02_noise.py, plot_results.py
src/pec/
  data/       qrels grades + eval views, PyTerrier dataset loading, context and noise draws
  index/      read-only FAISS store (docno <-> row), E5 query encoding via pyterrier-dr
  analysis/   figure style, the Oracle (shares, pinned values), plots from stored results
  experiments/exp01.py, exp02.py
  pipeline.py        PyTerrier transformers: centroid, seed fusion, MMR, inverse MMR, BM25 baselines
  query_formulation.py   zero-shot / static ICL / sliding window / doc2query (from proactive-icl-query-pair)
  text_retrievers.py     BM25, monoT5, ColBERT, SPLADE (from proactive-icl-query-pair)
  evaluate.py        pt.Experiment per view, the context-size sweep, its cache, tables and figures
tests/        synthetic fixtures and tests
```
