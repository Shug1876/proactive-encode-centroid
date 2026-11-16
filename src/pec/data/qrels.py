"""Graded qrels helpers.

Grades follow TREC DL: 0 non-relevant, 1 partially relevant ("related"),
2 highly relevant, 3 perfectly relevant. Grade-1 documents form the context a
baseline is given; grade-3 documents are what it should find.

Other collections use other scales (TREC-COVID: 0 / 1 partially relevant / 2 relevant),
so which grade is the context, which is the target and how runs are scored is a
per-dataset `Relevance` scheme (configs/paths.yaml, datasets.<name>.relevance).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import pandas as pd

NON_RELEVANT, PARTIAL, HIGH, PERFECT = 0, 1, 2, 3
GRADES = (NON_RELEVANT, PARTIAL, HIGH, PERFECT)
GRADE_NAMES = {
    NON_RELEVANT: "non-relevant",
    PARTIAL: "partially relevant",
    HIGH: "highly relevant",
    PERFECT: "perfectly relevant",
}
UNJUDGED = -1


def normalise_qrels(qrels: pd.DataFrame) -> pd.DataFrame:
    """PyTerrier-style qrels (qid, docno, label) with string ids and int grades.
    Labels outside 0..3 are clipped into range."""
    out = qrels[["qid", "docno", "label"]].copy()
    out["qid"] = out["qid"].astype(str)
    out["docno"] = out["docno"].astype(str)
    out["label"] = out["label"].astype(int).clip(NON_RELEVANT, PERFECT)
    return out.drop_duplicates(["qid", "docno"], keep="last").reset_index(drop=True)


def grade_sets(qrels: pd.DataFrame) -> dict[str, dict[int, list[str]]]:
    """qid -> grade -> docnos (every grade key present, possibly empty)."""
    out: dict[str, dict[int, list[str]]] = defaultdict(lambda: {g: [] for g in GRADES})
    for qid, docno, label in qrels[["qid", "docno", "label"]].itertuples(index=False):
        out[qid][int(label)].append(docno)
    return dict(out)


def label_lookup(qrels: pd.DataFrame) -> dict[str, dict[str, int]]:
    """qid -> docno -> grade."""
    out: dict[str, dict[str, int]] = defaultdict(dict)
    for qid, docno, label in qrels[["qid", "docno", "label"]].itertuples(index=False):
        out[qid][docno] = int(label)
    return dict(out)


@dataclass(frozen=True)
class Relevance:
    """Which grade is the context, which is the target, and the evaluation views.

    A view lists the grades that count as relevant; they keep their label (so nDCG stays
    graded across them) and every other grade is scored 0.
    """
    context: int = PARTIAL
    target: int = PERFECT
    views: dict[str, tuple[int, ...]] = field(default_factory=lambda: {"grade>=2": (HIGH, PERFECT), "grade=3": (PERFECT,)})
    names: dict[int, str] = field(default_factory=lambda: dict(GRADE_NAMES))

    @classmethod
    def from_config(cls, entry: dict | None) -> "Relevance":
        if not entry:
            return cls()
        default = cls()
        views = {str(k): tuple(int(g) for g in v) for k, v in entry.get("views", default.views).items()}
        names = {int(k): str(v) for k, v in entry.get("names", default.names).items()}
        return cls(int(entry.get("context", default.context)), int(entry.get("target", default.target)), views, names)

    @property
    def view_names(self) -> tuple[str, ...]:
        return tuple(self.views)

    @property
    def primary_view(self) -> str:
        """The first view: used where one view has to be chosen."""
        return next(iter(self.views))

    @property
    def target_view(self) -> str:
        """The view that scores only the target grade (falls back to the primary view)."""
        return next((v for v, g in self.views.items() if tuple(g) == (self.target,)), self.primary_view)

    @property
    def intermediate(self) -> tuple[int, ...]:
        """Grades strictly between the context and the target (DL: 2; TREC-COVID: none)."""
        return tuple(range(self.context + 1, self.target))

    def name(self, grade: int) -> str:
        return self.names.get(grade, f"grade {grade}")


DL_RELEVANCE = Relevance()


def qrels_view(qrels: pd.DataFrame, view: str, relevance: Relevance = DL_RELEVANCE) -> pd.DataFrame:
    """Relabel qrels for one evaluation view: grades in the view keep their label, every
    other grade becomes 0. For TREC DL, "grade>=2" keeps 2 and 3 (nDCG graded over both,
    Recall counts both) and "grade=3" keeps only 3."""
    if view not in relevance.views:
        raise ValueError(f"unknown qrels view {view!r}; this dataset has {relevance.view_names}")
    out = qrels.copy()
    out.loc[~out["label"].isin(relevance.views[view]), "label"] = 0
    return out


EVAL_VIEWS = DL_RELEVANCE.view_names


def view_tag(view: str) -> str:
    """Short file/column-safe name for a view: "grade>=2" -> "ge2", "grade=3" -> "eq3"."""
    return view.replace("grade", "").replace(">=", "ge").replace("=", "eq")
