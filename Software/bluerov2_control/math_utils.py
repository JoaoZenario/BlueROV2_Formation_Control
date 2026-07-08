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


def smooth_saturate_scalar(v: float, max_abs: float, eps: float = 1e-9) -> float:
    """
    Smooth saturation for a scalar value using tanh.

    - Near zero: approximately identity
    - For large values: smoothly saturates to max_abs

    Parameters
    ----------
    v : float
        Input scalar.
    max_abs : float
        Maximum allowed absolute value.
    eps : float
        Small threshold to avoid division by zero.
    """
    if max_abs <= 0.0:
        raise ValueError("max_abs must be positive.")

    scaled_v = v / (max_abs + eps)
    saturated_v = max_abs * np.tanh(scaled_v)
    return saturated_v


def smooth_saturate_componentwise(
    v: np.ndarray, max_abs: np.ndarray, eps: float = 1e-9
) -> np.ndarray:
    """
    Component-wise smooth saturation using tanh.

    - Near zero: approximately identity
    - For large values: smoothly saturates to max_abs

    Parameters
    ----------
    v : np.ndarray
        Input vector.
    max_abs : np.ndarray
        Maximum allowed absolute value for each component.
    eps : float
        Small threshold to avoid division by zero.
    """
    v = np.asarray(v, dtype=float).reshape(-1)
    max_abs = np.asarray(max_abs, dtype=float).reshape(-1)

    if np.any(max_abs <= 0.0):
        raise ValueError("All elements of max_abs must be positive.")

    # Avoid division by zero by adding a small epsilon inside the tanh argument
    scaled_v = v / (max_abs + eps)
    saturated_v = max_abs * np.tanh(scaled_v)
    return saturated_v


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
