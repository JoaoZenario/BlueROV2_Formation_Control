import numpy as np


Array = np.ndarray


def smooth_saturate(
    v: np.ndarray, max_norm: float, eps: float = 1e-9, switch_ratio: float = 4.0
) -> np.ndarray:
    """
    Piecewise-smooth radial saturation.

    - Near the origin: approximately identity
    - Intermediate range: smooth tanh saturation
    - Large range: exact hard saturation to max_norm

    Parameters
    ----------
    v : np.ndarray
        Input vector.
    max_norm : float
        Maximum allowed norm.
    eps : float
        Small threshold to avoid division by zero.
    switch_ratio : float
        Switch to exact saturation when ||v|| / max_norm exceeds this value.
    """
    v = np.asarray(v, dtype=float).reshape(-1)
    if max_norm <= 0.0:
        raise ValueError("max_norm must be positive.")

    norm_v = np.linalg.norm(v)
    if norm_v < eps:
        return v.copy()

    ratio = norm_v / max_norm

    if ratio >= switch_ratio:
        return (max_norm / norm_v) * v

    scale = max_norm * np.tanh(ratio) / norm_v
    return scale * v


def skew(w: Array) -> Array:
    w = np.asarray(w, dtype=float).reshape(3)
    return np.array(
        [
            [0.0, -w[2], w[1]],
            [w[2], 0.0, -w[0]],
            [-w[1], w[0], 0.0],
        ]
    )


def normalize_vector(v: Array, eps: float = 1e-9) -> Array:
    v = np.asarray(v, dtype=float).reshape(-1)
    norm_v = np.linalg.norm(v)
    if norm_v < eps:
        raise ValueError("Cannot normalize zero vector")
    return v / norm_v
