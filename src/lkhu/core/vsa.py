"""VSA (Vector Symbolic Architecture) core operations — HRR style.

Design doc §3.3, §7. All scents (vectors) are passed as numpy ``float32`` 1-D arrays.

In HRR (Holographic Reduced Representation), bind is *circular convolution* and unbind is
*circular correlation*, which are exactly equivalent to multiplication/conjugate-multiplication
in the frequency domain. Therefore we implement them with numpy FFT.

    - ``bind``    : irfft(rfft(a) * rfft(b))         — reversible binding
    - ``unbind``  : irfft(rfft(c) * conj(rfft(key))) — unbind with a key (inverse correlation)
    - ``bundle``  : add several scents together (preserves membership)

If a key scent is made *unitary* (frequency-domain magnitude 1), unbind becomes an exact inverse.

Implementation note: the initial design used torchhd (HRRTensor), but torch and faiss each link
their own OpenMP runtime, causing a fatal clash (OMP Error #179) under multithreading
(FastMCP workers + APScheduler background). Since HRR can be implemented identically with FFT as
above, the torch dependency was removed. (Same unbind recovery rate and bundle membership as
torchhd, install size reduced by ~1GB, thread-safe across all 3 OSes.)
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "normalize",
    "cosine",
    "bind",
    "unbind",
    "bundle",
    "random_vector",
    "unitary_vector",
]


def normalize(v: np.ndarray) -> np.ndarray:
    """L2 normalize. Leaves the zero vector unchanged (avoids division by zero).

    Args:
        v: Input scent.

    Returns:
        A ``float32`` scent with L2 norm 1 (zero vector if the input is the zero vector).
    """
    arr = np.asarray(v, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if norm == 0.0:
        return arr.copy()
    return (arr / norm).astype(np.float32)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two scents.

    Args:
        a: Scent A.
        b: Scent B.

    Returns:
        A similarity in the range -1.0 to 1.0 (0.0 if either is the zero vector).
    """
    av = np.asarray(a, dtype=np.float32)
    bv = np.asarray(b, dtype=np.float32)
    na = float(np.linalg.norm(av))
    nb = float(np.linalg.norm(bv))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(av, bv) / (na * nb))


def bind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Bind two scents (circular convolution).

    Args:
        a: Scent A (e.g. a key).
        b: Scent B (e.g. a value).

    Returns:
        The bound scent. ``unbind(bind(a, b), a) ≈ b``.
    """
    av = np.asarray(a, dtype=np.float32)
    bv = np.asarray(b, dtype=np.float32)
    n = av.shape[-1]
    bound = np.fft.irfft(np.fft.rfft(av) * np.fft.rfft(bv), n=n)
    return bound.astype(np.float32)


def unbind(bound: np.ndarray, key: np.ndarray) -> np.ndarray:
    """Unbind a bound scent with a key (circular correlation = conjugate multiplication).

    Args:
        bound: A scent bound via ``bind``.
        key: The key scent to unbind with. Recovery is exact for a unitary key.

    Returns:
        The recovered value scent (approximates the value that was bound to the key).
    """
    cv = np.asarray(bound, dtype=np.float32)
    kv = np.asarray(key, dtype=np.float32)
    n = cv.shape[-1]
    recovered = np.fft.irfft(np.fft.rfft(cv) * np.conj(np.fft.rfft(kv)), n=n)
    return recovered.astype(np.float32)


def bundle(vectors: list[np.ndarray] | np.ndarray, *, normalized: bool = True) -> np.ndarray:
    """Add several scents into one bottle (preserves membership).

    Args:
        vectors: A list (or 2-D array) of scents to add.
        normalized: If True, L2 normalize the result.

    Returns:
        The summed (optionally normalized) scent. Empty input yields the zero vector.
    """
    mat = np.asarray(vectors, dtype=np.float32)
    if mat.ndim == 1:
        mat = mat[None, :]
    if mat.shape[0] == 0:
        raise ValueError("An empty scent list was passed to bundle().")
    summed = mat.sum(axis=0).astype(np.float32)
    return normalize(summed) if normalized else summed


def random_vector(dim: int, rng: np.random.Generator) -> np.ndarray:
    """A random scent drawn from a normal distribution and L2 normalized.

    Args:
        dim: Dimension.
        rng: numpy random generator (guarantees determinism).

    Returns:
        A unit-norm ``float32`` scent.
    """
    return normalize(rng.standard_normal(dim).astype(np.float32))


def unitary_vector(dim: int, rng: np.random.Generator) -> np.ndarray:
    """A unitary HRR scent (frequency-domain magnitude 1, random phase).

    Such a scent has a near-exact ``unbind`` inverse, making it suitable as a key scent.

    Args:
        dim: Dimension.
        rng: numpy random generator (guarantees determinism).

    Returns:
        A real ``float32`` unitary scent (norm 1).
    """
    half = dim // 2 + 1
    phase = rng.uniform(-np.pi, np.pi, size=half)
    spectrum = np.exp(1j * phase)
    spectrum[0] = 1.0  # DC component is real
    if dim % 2 == 0:
        spectrum[-1] = 1.0  # Nyquist component is also real
    v = np.fft.irfft(spectrum, n=dim).astype(np.float32)
    return normalize(v)
