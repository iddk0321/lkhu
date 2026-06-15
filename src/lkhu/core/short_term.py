"""ShortTermBundle — recently accumulated scent (short-term memory). Design doc §4, §7.5.

Corresponds to the hippocampal short-term index. Scents are continually added and accumulated
(bundled), and decay rapidly at ×0.7 per day (nearly gone after 3 days). Frequently occurring
scents remain strong.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from lkhu.core import vsa

__all__ = ["ShortTermBundle"]


class ShortTermBundle:
    """A single bottle of accumulated scent.

    Args:
        dim: Scent dimensionality.
    """

    def __init__(self, dim: int):
        self.dim = dim
        self._acc = np.zeros(dim, dtype=np.float32)

    @property
    def raw(self) -> np.ndarray:
        """Unnormalized accumulated scent (retains strength information)."""
        return self._acc

    def add(self, vector: np.ndarray) -> None:
        """Accumulate a scent."""
        self._acc = (self._acc + np.asarray(vector, dtype=np.float32)).astype(np.float32)

    def bundle(self, normalized: bool = True) -> np.ndarray:
        """Return the current accumulated scent (normalized by default)."""
        return vsa.normalize(self._acc) if normalized else self._acc.copy()

    def decay(self, factor: float) -> None:
        """Decay the accumulated scent (e.g. ×0.7 per day)."""
        self._acc = (self._acc * float(factor)).astype(np.float32)

    def clear(self) -> None:
        """Reset the accumulation."""
        self._acc = np.zeros(self.dim, dtype=np.float32)

    def save(self, path: str | Path) -> None:
        """Save the accumulated scent to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, self._acc)

    @classmethod
    def load(cls, path: str | Path, dim: int) -> ShortTermBundle:
        """Load a saved accumulated scent (empty bundle if absent)."""
        st = cls(dim=dim)
        path = Path(path)
        if path.exists():
            acc = np.load(path).astype(np.float32)
            if acc.shape == (dim,):
                st._acc = acc
        return st
