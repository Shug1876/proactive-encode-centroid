"""Read-only access to a pre-built E5 FAISS index.

Nothing here builds or trains an index: `FaissDocStore.load` opens an existing FAISS file
plus the docno list that maps index rows to MS MARCO passage ids. Documents are handled
as integer rows internally; docnos are only used at the edges (qrels, PyTerrier frames).
"""
from __future__ import annotations

import logging
from pathlib import Path

import faiss
import numpy as np

from pec.config import require_path

log = logging.getLogger(__name__)


class DocnoMap:
    """Bidirectional docno <-> row mapping, from a text file (one docno per line) or an
    npids file (as written by pyterrier-dr's FlexIndex, memory-mapped)."""

    def __init__(self, docnos: list[str] | None = None, lookup=None):
        if (docnos is None) == (lookup is None):
            raise ValueError("pass exactly one of docnos or lookup")
        self._lookup = lookup
        if docnos is not None:
            self._docnos = np.asarray(docnos, dtype=object)
            self._rows = {d: i for i, d in enumerate(self._docnos)}

    @classmethod
    def load(cls, path: str | Path) -> "DocnoMap":
        path = Path(path)
        if path.suffix == ".npids":
            import npids

            return cls(lookup=npids.Lookup(str(path)))
        with open(path) as f:
            return cls(docnos=[line.strip() for line in f if line.strip()])

    def __len__(self) -> int:
        return len(self._lookup) if self._lookup is not None else len(self._docnos)

    def rows(self, docnos) -> np.ndarray:
        """Row for each docno; -1 when the docno is not in the index."""
        docnos = [str(d) for d in docnos]
        if not docnos:
            return np.zeros(0, dtype=np.int64)
        if self._lookup is not None:
            return np.asarray(self._lookup.inv.lookup(np.asarray(docnos)), dtype=np.int64).reshape(-1)
        return np.fromiter((self._rows.get(d, -1) for d in docnos), dtype=np.int64, count=len(docnos))

    def docnos(self, rows) -> list[str]:
        rows = np.asarray(rows, dtype=np.int64)
        if rows.size == 0:
            return []
        if self._lookup is not None:
            return [str(d) for d in self._lookup.fwd.lookup(rows)]
        return [str(d) for d in self._docnos[rows]]


class FaissDocStore:
    """Vectors and nearest-neighbour search over a FAISS index of E5 passage vectors."""

    def __init__(self, index: faiss.Index, docno_map: DocnoMap, search_params: dict | None = None):
        if index.ntotal != len(docno_map):
            raise ValueError(f"index has {index.ntotal} vectors but docno map has {len(docno_map)} entries")
        self.index = index
        self.docno_map = docno_map
        self.dim = index.d
        # For unit vectors, squared L2 distance d relates to cosine by cos = 1 - d / 2.
        self._l2 = index.metric_type == faiss.METRIC_L2
        self._direct_map_ready = False
        for name, value in (search_params or {}).items():
            try:
                faiss.ParameterSpace().set_index_parameter(index, name, value)
            except RuntimeError:
                log.debug("index has no search parameter %s; ignored", name)

    @classmethod
    def load(cls, index_path, docnos_path, search_params: dict | None = None, mmap: bool = True) -> "FaissDocStore":
        index_path = require_path(index_path, "FAISS index")
        docnos_path = require_path(docnos_path, "docno list")
        index = None
        if mmap:
            try:
                index = faiss.read_index(str(index_path), faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY)
            except RuntimeError:
                log.info("index type does not support mmap; reading into memory")
        if index is None:
            index = faiss.read_index(str(index_path))
        return cls(index, DocnoMap.load(docnos_path), search_params)

    @classmethod
    def from_config(cls, cfg: dict, index_key: str) -> "FaissDocStore":
        entry = cfg["indexes"][index_key]
        return cls.load(entry["faiss"], entry["docnos"], entry.get("search_params"), entry.get("mmap", True))

    def __len__(self) -> int:
        return self.index.ntotal

    def rows(self, docnos) -> np.ndarray:
        return self.docno_map.rows(docnos)

    def docnos(self, rows) -> list[str]:
        return self.docno_map.docnos(rows)

    def vectors(self, rows) -> np.ndarray:
        """Stored vectors for index rows. Exact for Flat/HNSW indexes; IVF indexes need a
        direct map (made on first use); PQ-compressed indexes return approximations."""
        rows = np.asarray(rows, dtype=np.int64)
        if rows.size == 0:
            return np.zeros((0, self.dim), dtype=np.float32)
        if not self._direct_map_ready:
            try:
                faiss.extract_index_ivf(self.index).make_direct_map()
            except RuntimeError:
                pass  # not an IVF index
            self._direct_map_ready = True
        return np.asarray(self.index.reconstruct_batch(rows), dtype=np.float32)

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Top-k (cosine similarities, rows) for each query vector; missing slots are -1."""
        queries = np.ascontiguousarray(np.atleast_2d(queries), dtype=np.float32)
        k = min(k, self.index.ntotal)
        dist, rows = self.index.search(queries, k)
        sims = 1.0 - dist / 2.0 if self._l2 else dist
        return sims.astype(np.float32), rows.astype(np.int64)
