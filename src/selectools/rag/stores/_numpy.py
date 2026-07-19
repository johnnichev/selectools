"""NumPy helpers shared by vector-store implementations."""

from __future__ import annotations

import numpy as np


def cosine_numpy(matrix: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Return row-wise cosine similarities, using zero for zero-norm rows."""
    norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(query)
    return np.divide(
        np.dot(matrix, query),
        norms,
        out=np.zeros(matrix.shape[0], dtype=np.float32),
        where=norms != 0,
    )
