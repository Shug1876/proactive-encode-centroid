"""Contexts of partially relevant (grade-1) documents, optionally mixed with judged
non-relevant (grade-0) ones (noise_draws, Experiment 2).

A context is passed through PyTerrier as a topics frame with one row per query and a
`seed_docnos` tuple column. The true query text is deliberately not part of that frame:
the information need is unknown to every baseline.
"""
from __future__ import annotations

import zlib

import numpy as np
import pandas as pd

from pec.data.qrels import NON_RELEVANT, PARTIAL, grade_sets

ALL = "all"
NOISE_STREAM = 7  # extra seed word of the grade-0 shuffle, independent of the grade-1 one


def seed_topics(qrels: pd.DataFrame, max_seeds: int | None = None, random_seed: int = 0,
                require_grade: int | None = 3, context_grade: int = PARTIAL) -> pd.DataFrame:
    """Topics frame [qid, seed_docnos] from each query's context-grade (default grade-1)
    documents.

    Queries without context documents are skipped, as are queries without any document of
    `require_grade` (the target: nothing to discover) unless it is None.
    """
    rng = np.random.default_rng(random_seed)
    records = []
    for qid, by_grade in sorted(grade_sets(qrels).items()):
        seeds = sorted(by_grade[context_grade])
        if not seeds or (require_grade is not None and not by_grade[require_grade]):
            continue
        if max_seeds is not None and len(seeds) > max_seeds:
            seeds = sorted(rng.choice(seeds, size=max_seeds, replace=False).tolist())
        records.append({"qid": qid, "seed_docnos": tuple(seeds)})
    return pd.DataFrame.from_records(records, columns=["qid", "seed_docnos"])


def size_label(size) -> str:
    return ALL if str(size).lower() == ALL else str(int(size))


def ordered_sizes(labels) -> list[str]:
    """Context-size labels in increasing order, "all" last."""
    labels = {size_label(s) for s in labels}
    return sorted((s for s in labels if s != ALL), key=int) + ([ALL] if ALL in labels else [])


def context_draws(qrels: pd.DataFrame, sizes, repeats: int, random_seed: int = 0,
                  require_grade: int | None = 3, context_grade: int = PARTIAL) -> pd.DataFrame:
    """Contexts of k context-grade (default grade-1) documents per query, for every k in
    `sizes` and every repeat. `require_grade` is the target grade a query must have.

    Returns [qid, seed_docnos, context_size, repeat]. For each (query, repeat) the grade-1
    documents are shuffled once and the first k form the context, so within a repeat the
    k=3 context is contained in the k=5 one, and so on: differences between sizes come from
    size, not from a different draw. A query with fewer than k grade-1 documents is left
    out at that k rather than given a smaller context. "all" uses every grade-1 document
    and has a single repeat (every repeat would be identical).
    """
    labels = ordered_sizes(sizes)
    records = []
    for qid, by_grade in sorted(grade_sets(qrels).items()):
        docs = sorted(by_grade[context_grade])
        if not docs or (require_grade is not None and not by_grade[require_grade]):
            continue
        for repeat in range(repeats):
            rng = np.random.default_rng([random_seed, repeat, zlib.crc32(qid.encode())])
            shuffled = [docs[i] for i in rng.permutation(len(docs))]
            for label in labels:
                if label == ALL:
                    if repeat == 0:
                        records.append({"qid": qid, "seed_docnos": tuple(docs), "context_size": ALL, "repeat": 0})
                elif len(docs) >= int(label):
                    records.append({"qid": qid, "seed_docnos": tuple(sorted(shuffled[: int(label)])),
                                    "context_size": label, "repeat": repeat})
    return pd.DataFrame.from_records(records, columns=["qid", "seed_docnos", "context_size", "repeat"])


def parse_alpha(alpha) -> float:
    """α, the share of non-relevant documents in a context: a number in [0, 1)."""
    try:
        value = float(alpha)
    except (TypeError, ValueError):
        raise ValueError(f"alpha must be a number in [0, 1), the share of non-relevant documents; got {alpha!r}")
    if not 0.0 <= value < 1.0:
        raise ValueError(f"alpha {alpha!r} must be in [0, 1): a context needs at least one partially relevant document")
    return value


def alpha_label(alpha) -> str:
    """Table key of an α: one decimal place ("0.0", "0.2", ...), more only when needed ("0.25")."""
    value = parse_alpha(alpha)
    one = f"{value:.1f}"
    return one if abs(float(one) - value) < 1e-9 else f"{value:.6g}"


def alpha_counts(alpha, size: int) -> tuple[int, int]:
    """(non-relevant, context-grade) documents of a `size`-document context with share α
    of non-relevant documents; α · size must be a whole number."""
    n_noise = parse_alpha(alpha) * size
    if abs(n_noise - round(n_noise)) > 1e-9:
        raise ValueError(f"alpha {alpha_label(alpha)} does not split a {size}-document context into whole documents "
                         f"({n_noise:g} non-relevant)")
    return int(round(n_noise)), size - int(round(n_noise))


def noise_draws(qrels: pd.DataFrame, size: int, alphas, repeats: int, random_seed: int = 0,
                require_grade: int | None = 3, context_grade: int = PARTIAL) -> pd.DataFrame:
    """Contexts of `size` documents in which a share α of the documents are the query's
    judged non-relevant (grade-0) documents and the rest its context-grade (default
    grade-1) documents, for every α in `alphas` (e.g. 0, 0.2, ..., 0.8) and repeat.

    Non-relevant documents are only ever the query's own judged grade-0 documents, never
    documents sampled from the collection. Only queries with enough of both for every α
    (and a `require_grade` document) are used, so every α is evaluated on the same queries.

    For each (query, repeat) the context-grade documents are shuffled exactly as in
    context_draws (so the α = 0 context is context_draws' k=`size` context) and the
    grade-0 documents are shuffled once with their own generator; an α takes the first
    n of each list, so contexts are nested across α (0.2 -> 0.4 swaps two partially
    relevant documents for two non-relevant ones).

    Returns [qid, seed_docnos, alpha, repeat, noise_docnos, context_docnos].
    """
    labels = [alpha_label(a) for a in alphas]
    if len(set(labels)) != len(labels):
        raise ValueError(f"duplicate alphas in {labels}")
    counts = {label: alpha_counts(label, size) for label in labels}
    need_noise = max(n for n, _ in counts.values())
    need_context = max(c for _, c in counts.values())
    records = []
    for qid, by_grade in sorted(grade_sets(qrels).items()):
        docs, noise = sorted(by_grade[context_grade]), sorted(by_grade[NON_RELEVANT])
        if len(docs) < need_context or len(noise) < need_noise:
            continue
        if require_grade is not None and not by_grade[require_grade]:
            continue
        for repeat in range(repeats):
            rng = np.random.default_rng([random_seed, repeat, zlib.crc32(qid.encode())])
            shuffled = [docs[i] for i in rng.permutation(len(docs))]
            noise_rng = np.random.default_rng([random_seed, repeat, zlib.crc32(qid.encode()), NOISE_STREAM])
            shuffled_noise = [noise[i] for i in noise_rng.permutation(len(noise))]
            for label in labels:
                n_noise, n_context = counts[label]
                picked_noise, picked = tuple(sorted(shuffled_noise[:n_noise])), tuple(sorted(shuffled[:n_context]))
                records.append({"qid": qid, "seed_docnos": tuple(sorted(picked_noise + picked)), "alpha": label,
                                "repeat": repeat, "noise_docnos": picked_noise, "context_docnos": picked})
    return pd.DataFrame.from_records(records, columns=["qid", "seed_docnos", "alpha", "repeat", "noise_docnos",
                                                       "context_docnos"])
