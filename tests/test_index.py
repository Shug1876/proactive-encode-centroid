import faiss
import numpy as np
import pytest

from pec.index.faiss_store import DocnoMap, FaissDocStore


def test_store_vectors_and_search(toy):
    store, vecs = toy["store"], toy["vecs"]
    rows = store.rows(["d3", "d10", "missing"])
    assert rows.tolist() == [3, 10, -1]
    np.testing.assert_allclose(store.vectors(rows[:2]), vecs[[3, 10]], atol=1e-6)
    sims, found = store.search(vecs[5], 3)
    assert found[0, 0] == 5 and sims[0, 0] == pytest.approx(1.0, abs=1e-5)


def test_l2_index_returns_cosine(toy):
    index = faiss.IndexFlatL2(toy["vecs"].shape[1])
    index.add(toy["vecs"])
    store = FaissDocStore(index, DocnoMap(docnos=toy["docnos"]))
    sims, found = store.search(toy["vecs"][7], 2)
    expected = toy["vecs"][found[0]] @ toy["vecs"][7]
    np.testing.assert_allclose(sims[0], expected, atol=1e-5)


def test_docno_map_npids(tmp_path):
    import npids

    npids.Lookup.build(["a", "b", "c"], str(tmp_path / "docnos.npids"))
    m = DocnoMap.load(tmp_path / "docnos.npids")
    assert len(m) == 3 and m.rows(["c", "zz"]).tolist() == [2, -1] and m.docnos([1]) == ["b"]
