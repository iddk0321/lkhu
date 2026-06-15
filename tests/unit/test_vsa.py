"""Phase 2 — VSA (HRR) core operation tests.

Validates key-value bind→unbind recovery rate, bundle membership, normalization, and determinism.
"""

from __future__ import annotations

import numpy as np

from lkhu.core import vsa


def test_normalize_unit_norm() -> None:
    v = np.array([3.0, 4.0], dtype=np.float32)
    out = vsa.normalize(v)
    assert np.isclose(np.linalg.norm(out), 1.0)


def test_normalize_zero_vector_is_safe() -> None:
    out = vsa.normalize(np.zeros(8, dtype=np.float32))
    assert np.all(np.isfinite(out))
    assert np.linalg.norm(out) == 0.0


def test_unitary_vector_unbind_is_near_exact() -> None:
    rng = np.random.default_rng(42)
    key = vsa.unitary_vector(1024, rng)
    value = vsa.normalize(rng.standard_normal(1024).astype(np.float32))

    bound = vsa.bind(key, value)
    recovered = vsa.unbind(bound, key)

    # A unitary key should be recovered almost exactly.
    assert vsa.cosine(recovered, value) > 0.95


def test_bind_changes_representation() -> None:
    rng = np.random.default_rng(7)
    key = vsa.unitary_vector(1024, rng)
    value = vsa.normalize(rng.standard_normal(1024).astype(np.float32))
    bound = vsa.bind(key, value)
    # The bound result should not resemble the originals.
    assert vsa.cosine(bound, value) < 0.3
    assert vsa.cosine(bound, key) < 0.3


def test_bundle_membership() -> None:
    rng = np.random.default_rng(3)
    members = [vsa.normalize(rng.standard_normal(1024).astype(np.float32)) for _ in range(3)]
    outsider = vsa.normalize(rng.standard_normal(1024).astype(np.float32))
    bundle = vsa.bundle(members)
    for m in members:
        assert vsa.cosine(bundle, m) > 0.3
    assert vsa.cosine(bundle, outsider) < 0.2


def test_bundle_probe_in_memory() -> None:
    """Probing a single memory (sum of two key-value pairs) with a key, that value is closest."""
    rng = np.random.default_rng(11)
    k1 = vsa.unitary_vector(1024, rng)
    k2 = vsa.unitary_vector(1024, rng)
    v1 = vsa.normalize(rng.standard_normal(1024).astype(np.float32))
    v2 = vsa.normalize(rng.standard_normal(1024).astype(np.float32))
    memory = vsa.bundle([vsa.bind(k1, v1), vsa.bind(k2, v2)])
    probe = vsa.unbind(memory, k1)
    assert vsa.cosine(probe, v1) > vsa.cosine(probe, v2)
    assert vsa.cosine(probe, v1) > 0.3


def test_cosine_bounds_and_self() -> None:
    rng = np.random.default_rng(5)
    v = vsa.normalize(rng.standard_normal(64).astype(np.float32))
    assert np.isclose(vsa.cosine(v, v), 1.0, atol=1e-5)
    assert -1.0001 <= vsa.cosine(v, -v) <= -0.9999


def test_bind_is_deterministic() -> None:
    rng = np.random.default_rng(9)
    a = vsa.unitary_vector(256, rng)
    b = vsa.normalize(rng.standard_normal(256).astype(np.float32))
    assert np.allclose(vsa.bind(a, b), vsa.bind(a, b))


def test_unitary_vector_is_reproducible_from_seed() -> None:
    a = vsa.unitary_vector(128, np.random.default_rng(123))
    b = vsa.unitary_vector(128, np.random.default_rng(123))
    assert np.allclose(a, b)
