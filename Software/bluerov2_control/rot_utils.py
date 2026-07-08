import numpy as np
from scipy.spatial.transform import Rotation as R

from .math_utils import normalize_vector

Array = np.ndarray


def ZYXeul2quat(roll, pitch, yaw):
    """
    Convert ZYX Euler angles (roll, pitch, yaw) to quaternion [w, x, y, z].
    Uses the standard convention: q = [w, x, y, z].
    """
    # Use scipy for robust conversion (scipy returns [x, y, z, w], so reorder)
    r = R.from_euler("ZYX", [yaw, pitch, roll], degrees=False)  # note order for SciPy
    qxyzw = r.as_quat()  # [x, y, z, w]
    return np.array([qxyzw[3], qxyzw[0], qxyzw[1], qxyzw[2]])


def quaternion_to_roll_pitch_yaw(q: np.ndarray) -> tuple[float, float, float]:
    """
    Convert quaternion [x, y, z, w] to ZYX roll, pitch, yaw.
    """
    qx, qy, qz, qw = normalize_vector(q)

    # roll
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # pitch
    sinp = 2.0 * (qw * qy - qz * qx)
    if abs(sinp) >= 1.0:
        pitch = np.sign(sinp) * (np.pi / 2.0)
    else:
        pitch = np.arcsin(sinp)

    # yaw
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


def quat_mul(q1, q2):
    """Hamilton product. Both quaternions are [w, x, y, z]."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.array([w, x, y, z])


def quat_error(q_cmd, q):
    """
    Quaternion error (vector part) for desired q_cmd and current q.
    Both in [w, x, y, z].
    """
    q_cmd_conj = np.array([q_cmd[0], -q_cmd[1], -q_cmd[2], -q_cmd[3]])
    q_err = quat_mul(q_cmd_conj, q)
    return q_err[1:]  # vector part


def quatKDE(q: np.ndarray, body_omega: np.ndarray) -> np.ndarray:
    """
    Quaternion kinematic differential equation dq/dt = 0.5 * Omega(omega) * q,
    where q = [w, x, y, z] and body_omega = [wx, wy, wz].
    Returns dq (4,).
    """
    q = np.asarray(q, dtype=float).flatten()
    omega = np.asarray(body_omega, dtype=float).flatten()
    if q.shape != (4,):
        raise ValueError("q must be shape (4,) in [w,x,y,z] order")
    if omega.shape != (3,):
        raise ValueError("body_omega must be shape (3,)")

    return 0.5 * omega_matrix(omega) @ q


def quaternion_derivative_body_rates(q: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """
    Quaternion derivative for body-frame angular velocity.

    Quaternion convention: q = [qx, qy, qz, qw] (scalar last).

    Implements:
        q_dot = 0.5 * q ⊗ [omega, 0]
    """
    qx, qy, qz, qw = normalize_vector(q)
    wx, wy, wz = np.asarray(omega, dtype=float).reshape(3)

    qdot = 0.5 * np.array(
        [
            qw * wx + qy * wz - qz * wy,
            qw * wy + qz * wx - qx * wz,
            qw * wz + qx * wy - qy * wx,
            -(qx * wx + qy * wy + qz * wz),
        ],
        dtype=float,
    )
    return qdot


def vrrotvec2mat(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues' rotation formula for axis-angle."""
    ax = np.asarray(axis, dtype=float)
    if np.linalg.norm(ax) < 1e-6:
        raise ValueError("Axis must not be zero")
    ax = ax / np.linalg.norm(ax)
    x, y, z = ax
    c, s, t = np.cos(angle), np.sin(angle), 1 - np.cos(angle)
    Rm = np.array(
        [
            [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
            [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
            [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
        ]
    )
    return Rm


def rotz(psi: float) -> np.ndarray:
    return np.array(
        [
            [np.cos(psi), -np.sin(psi), 0.0],
            [np.sin(psi), np.cos(psi), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def quat2rotm(q: np.ndarray) -> np.ndarray:
    """Quaternion to rotation matrix. q = [w, x, y, z]."""
    q = np.asarray(q, dtype=float).flatten()
    if abs(np.linalg.norm(q) - 1.0) > 1e-6:
        raise ValueError("Quaternion must be normalized")
    w, x, y, z = q
    Rm = np.array(
        [
            [w * w + x * x - y * y - z * z, 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), w * w - x * x + y * y - z * z, 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), w * w - x * x - y * y + z * z],
        ]
    )
    return Rm


def rotation_matrix_from_quaternion(q: Array) -> Array:
    """
    SciPy convention: quaternion = [x, y, z, w]
    """
    q = np.asarray(q, dtype=float).reshape(4)
    return R.from_quat(q).as_matrix()


def yaw_from_quat(q: np.ndarray) -> float:
    """Extract yaw angle from quaternion [w, x, y, z]."""
    w, x, y, z = q
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def omega_matrix(omega):
    """Quaternion kinematic matrix Ω(ω) such that dq = 0.5 * Ω(ω) q.
    Expects omega = [p, q, r] (body rates).
    """
    p, q, r = omega
    return np.array(
        [[0.0, -p, -q, -r], [p, 0.0, r, -q], [q, -r, 0.0, p], [r, q, -p, 0.0]]
    )
