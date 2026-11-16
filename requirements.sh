#!/bin/bash
# Installs this repository (the pec package) and the extra PyTerrier plugins the
# baselines need. Run from the repository root.

pip uninstall -y python-terrier pyterrier faiss-cpu
pip install -e ".[dev]"
# doc2query baseline in Experiment 1 (as in proactive-icl-query-pair)
pip install --upgrade git+https://github.com/terrierteam/pyterrier_doc2query.git
# BM25 (bm25_DaaQ, bm25_centroid, the BM25 Oracle) and the text retrievers of the query
# formulation baselines (as in proactive-icl-query-pair): PISA (BM25, SPLADE index), monoT5, SPLADE
pip install pyterrier-pisa
pip install --upgrade git+https://github.com/terrierteam/pyterrier_t5.git
pip install -q git+https://github.com/naver/splade.git git+https://github.com/cmacdonald/pyt_splade.git
pip install --upgrade transformers
echo DONE
