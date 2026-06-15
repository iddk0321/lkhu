"""Phase 2 — Codebook (key-flavor dictionary) tests.

Verifies generation determinism, save/load round-trip, the regeneration guard, and that
existing keys stay unchanged when a new key is added.
"""

from __future__ import annotations

import numpy as np
import pytest

from lkhu.core.codebook import Codebook

INITIAL = ["K_USER", "K_TOPIC", "K_DECISION", "K_FILE", "K_LANGUAGE"]


def test_generate_has_all_keys_with_dim() -> None:
    cb = Codebook.generate(INITIAL, dim=1024, seed=12345)
    assert len(cb) == len(INITIAL)
    for k in INITIAL:
        assert k in cb
        assert cb[k].shape == (1024,)
        assert cb[k].dtype == np.float32


def test_keys_are_distinct() -> None:
    cb = Codebook.generate(INITIAL, dim=256, seed=1)
    from lkhu.core import vsa

    for i, a in enumerate(INITIAL):
        for b in INITIAL[i + 1 :]:
            assert vsa.cosine(cb[a], cb[b]) < 0.2


def test_generation_is_reproducible_from_seed() -> None:
    cb1 = Codebook.generate(INITIAL, dim=256, seed=999)
    cb2 = Codebook.generate(INITIAL, dim=256, seed=999)
    for k in INITIAL:
        assert np.allclose(cb1[k], cb2[k])


def test_save_load_roundtrip(tmp_path) -> None:
    cb = Codebook.generate(INITIAL, dim=256, seed=7)
    path = tmp_path / "codebook.npy"
    cb.save(path)
    assert path.exists()

    loaded = Codebook.load(path)
    assert loaded.keys() == cb.keys()
    assert loaded.dim == cb.dim
    for k in INITIAL:
        assert np.allclose(loaded[k], cb[k])


def test_save_creates_triple_backup(tmp_path) -> None:
    cb = Codebook.generate(INITIAL, dim=128, seed=2)
    path = tmp_path / "codebook.npy"
    b1 = tmp_path / "backup" / "codebook.backup.npy"
    b2 = tmp_path / "docs" / "lkhu_codebook.backup.npy"
    cb.save(path, backups=[b1, b2])
    assert path.exists() and b1.exists() and b2.exists()
    # The backups are loadable identically too
    assert np.allclose(Codebook.load(b1)["K_USER"], cb["K_USER"])


def test_regeneration_guard_blocks_overwrite(tmp_path) -> None:
    cb = Codebook.generate(INITIAL, dim=128, seed=3)
    path = tmp_path / "codebook.npy"
    cb.save(path)
    # If it already exists, overwriting is refused (the Codebook is inviolable)
    with pytest.raises(FileExistsError):
        cb.save(path)


def test_add_key_is_deterministic_and_preserves_existing(tmp_path) -> None:
    cb = Codebook.generate(INITIAL, dim=256, seed=55)
    before = {k: cb[k].copy() for k in INITIAL}

    v_new = cb.add_key("K_CUSTOM")
    assert "K_CUSTOM" in cb
    assert cb["K_CUSTOM"].shape == (256,)
    # Existing keys must not change
    for k in INITIAL:
        assert np.allclose(cb[k], before[k])

    # The same name deterministically produces the same vector (reproducibility)
    cb2 = Codebook.generate(INITIAL, dim=256, seed=55)
    assert np.allclose(cb2.add_key("K_CUSTOM"), v_new)


def test_add_existing_key_returns_same_vector() -> None:
    cb = Codebook.generate(INITIAL, dim=128, seed=8)
    assert np.allclose(cb.add_key("K_USER"), cb["K_USER"])


def test_load_detects_corruption(tmp_path) -> None:
    cb = Codebook.generate(INITIAL, dim=64, seed=4)
    path = tmp_path / "codebook.npy"
    cb.save(path)
    # Corrupting the matrix must make the integrity check fail
    mat = np.load(path)
    mat[0, 0] += 1.0
    np.save(path, mat)
    with pytest.raises(ValueError, match="integrity|checksum"):
        Codebook.load(path)
