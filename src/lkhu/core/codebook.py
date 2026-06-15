"""Codebook — the key-flavor dictionary (the system's DNA). See design doc §6.2.

⚠️ The Codebook is inviolable. Once created, it is never changed or regenerated
(CLAUDE.md Hard Rule 2). Losing it invalidates all memories, so multiple backups are
made on save.

Design decisions:
    Each key vector is a unitary HRR flavor derived deterministically from
    ``(master_seed, key_name)``. This ensures that (1) the codebook is fully reproducible
    given the same seed, and (2) adding a new key never changes existing keys. On save, the
    derived result matrix is persisted as-is (``.npy``) so that existing flavors are
    preserved even if the generation algorithm changes later.
"""

from __future__ import annotations

import hashlib
import json
import secrets
import shutil
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from lkhu.core import vsa

__all__ = ["Codebook"]

_META_SUFFIX = ".meta.json"


def _meta_path(path: Path) -> Path:
    """Metadata (JSON) path corresponding to the codebook matrix path."""
    return path.with_name(path.name + _META_SUFFIX)


def _checksum(matrix: np.ndarray) -> str:
    """SHA-256 checksum of the matrix bytes (for integrity checks)."""
    return hashlib.sha256(np.ascontiguousarray(matrix, dtype=np.float32).tobytes()).hexdigest()


class Codebook:
    """Dictionary mapping key name → flavor (numpy float32).

    Attributes:
        dim: Flavor dimension (default 1024).
        master_seed: Base seed for key derivation (permanently fixed).
    """

    def __init__(self, dim: int, master_seed: int, keys: dict[str, np.ndarray] | None = None):
        self.dim = dim
        self.master_seed = master_seed
        self._keys: dict[str, np.ndarray] = keys or {}

    # ----- generation / derivation -----

    @classmethod
    def generate(
        cls,
        initial_keys: Iterable[str],
        dim: int = 1024,
        seed: int | None = None,
    ) -> Codebook:
        """Generate a codebook from the initial keys.

        Args:
            initial_keys: System-defined key names.
            dim: Flavor dimension.
            seed: Base seed. If ``None``, generated from OS entropy (then permanently fixed).

        Returns:
            The generated ``Codebook``.
        """
        master_seed = seed if seed is not None else secrets.randbits(63)
        cb = cls(dim=dim, master_seed=master_seed)
        for name in initial_keys:
            cb.add_key(name)
        return cb

    def _derive(self, name: str) -> np.ndarray:
        """Deterministically derive a unitary key flavor from ``(master_seed, name)``."""
        digest = hashlib.blake2b(
            f"{self.master_seed}::{name}".encode(),
            digest_size=8,
        ).digest()
        rng = np.random.default_rng(int.from_bytes(digest, "big"))
        return vsa.unitary_vector(self.dim, rng)

    def add_key(self, name: str) -> np.ndarray:
        """Add a key and return its flavor (returns the existing flavor if already present).

        Existing keys are never changed.

        Args:
            name: Key name.

        Returns:
            The flavor for the given key.
        """
        if name in self._keys:
            return self._keys[name]
        vec = self._derive(name)
        self._keys[name] = vec
        return vec

    # ----- lookup -----

    def keys(self) -> list[str]:
        """List of registered key names (insertion order)."""
        return list(self._keys.keys())

    def __contains__(self, name: object) -> bool:
        return name in self._keys

    def __getitem__(self, name: str) -> np.ndarray:
        return self._keys[name]

    def __len__(self) -> int:
        return len(self._keys)

    def matrix(self) -> np.ndarray:
        """Return the key flavors as a ``[num_keys, dim]`` matrix (preserving key order)."""
        if not self._keys:
            return np.empty((0, self.dim), dtype=np.float32)
        return np.stack([self._keys[k] for k in self._keys], axis=0).astype(np.float32)

    # ----- save / load -----

    def save(
        self,
        path: str | Path,
        backups: Iterable[str | Path] = (),
        overwrite: bool = False,
    ) -> None:
        """Save the codebook (matrix ``.npy`` + meta ``.json``), then replicate backups.

        Args:
            path: Matrix save path (e.g. ``codebook.npy``).
            backups: Additional backup paths (secondary/tertiary backups).
            overwrite: If False (default), refuses to overwrite an existing file.

        Raises:
            FileExistsError: When the file already exists and ``overwrite`` is False
                (regeneration prevention).
        """
        path = Path(path)
        if path.exists() and not overwrite:
            raise FileExistsError(
                f"Codebook already exists: {path}. Per the inviolability rule, it is not "
                f"overwritten."
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        matrix = self.matrix()
        np.save(path, matrix)
        meta = {
            "dim": self.dim,
            "master_seed": self.master_seed,
            "keys": self.keys(),
            "checksum": _checksum(matrix),
            "format": 1,
        }
        _meta_path(path).write_text(json.dumps(meta, ensure_ascii=False, indent=2))

        for backup in backups:
            backup = Path(backup)
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, backup)
            shutil.copy2(_meta_path(path), _meta_path(backup))

    @classmethod
    def load(cls, path: str | Path) -> Codebook:
        """Load a saved codebook and verify its integrity.

        Args:
            path: Matrix ``.npy`` path.

        Returns:
            The loaded ``Codebook``.

        Raises:
            ValueError: On checksum mismatch (integrity corruption) or missing metadata.
        """
        path = Path(path)
        meta_path = _meta_path(path)
        if not meta_path.exists():
            raise ValueError(f"Codebook metadata is missing: {meta_path}")
        meta = json.loads(meta_path.read_text())
        matrix = np.load(path).astype(np.float32)

        if _checksum(matrix) != meta.get("checksum"):
            raise ValueError("Codebook integrity (checksum) check failed — the file is corrupted.")

        names: list[str] = meta["keys"]
        if matrix.shape[0] != len(names):
            raise ValueError(
                "Codebook integrity check failed — key count differs from matrix size."
            )

        keys = {name: matrix[i].copy() for i, name in enumerate(names)}
        return cls(dim=int(meta["dim"]), master_seed=int(meta["master_seed"]), keys=keys)

    @staticmethod
    def is_initialized(path: str | Path) -> bool:
        """Whether a codebook already exists at the given path."""
        path = Path(path)
        return path.exists() and _meta_path(path).exists()
