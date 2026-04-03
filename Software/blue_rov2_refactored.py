#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BlueROV2 quaternion-based wrench-input model.

State:
    x = [p, q, nu] in R^13
    p  = [x, y, z]
    q  = [qx, qy, qz, qw]   (SciPy convention, scalar last)
    nu = [u, v, w, wx, wy, wz]   body-fixed velocity

Input:
    tau = [Fx, Fy, Fz, Mx, My, Mz] in body frame
"""

from __future__ import annotations

import numpy as np
from math_utils import smooth_saturate, skew, normalize_vector

from rot_utils import (
    rotation_matrix_from_quaternion,
    quaternion_derivative_body_rates,
    quaternion_to_roll_pitch_yaw,
)


class BlueROV2:
    """
    Quaternion-based BlueROV2 heavy configuration dynamic model with direct wrench input.
    """

    def __init__(self, rho: float = 1000.0, current_speed: np.ndarray | None = None):
        # Physical parameters
        self.rho = rho
        self.g = 9.82
        self.m = 13.5
        self.volume = 0.0134
        self.W = self.m * self.g
        self.Buoy = self.rho * self.g * self.volume

        # Centers of gravity / buoyancy
        self.xg = 0.0
        self.yg = 0.0
        self.zg = 0.0
        self.xb = 0.0
        self.yb = 0.0
        self.zb = -0.01

        # Inertia
        self.Ix = 0.26
        self.Iy = 0.23
        self.Iz = 0.37

        # Rigid-body mass matrix
        self.MRB = np.zeros((6, 6), dtype=float)
        self.MRB[0, 0] = self.m
        self.MRB[1, 1] = self.m
        self.MRB[2, 2] = self.m
        self.MRB[3, 3] = self.Ix
        self.MRB[4, 4] = self.Iy
        self.MRB[5, 5] = self.Iz

        # Added mass derivatives
        self.Xu_dot = -6.36
        self.Yv_dot = -7.12
        self.Zw_dot = -18.68
        self.Kp_dot = -0.189
        self.Mq_dot = -0.135
        self.Nr_dot = -0.222

        self.MA = np.zeros((6, 6), dtype=float)
        self.MA[0, 0] = -self.Xu_dot
        self.MA[1, 1] = -self.Yv_dot
        self.MA[2, 2] = -self.Zw_dot
        self.MA[3, 3] = -self.Kp_dot
        self.MA[4, 4] = -self.Mq_dot
        self.MA[5, 5] = -self.Nr_dot

        self.M = self.MRB + self.MA
        self.Minv = np.linalg.inv(self.M)

        # Damping coefficients
        self.Xu = -13.7
        self.Xu_abs = -141.0
        self.Yv = -0.0
        self.Yv_abs = -217.0
        self.Zw = -33.0
        self.Zw_abs = -190.0
        self.Kp = -0.0
        self.Kp_abs = -1.19
        self.Mq = -0.8
        self.Mq_abs = -0.47
        self.Nr = -0.0
        self.Nr_abs = -1.5

        # Constant inertial current velocity
        if current_speed is None:
            current_speed = np.zeros(3, dtype=float)
        self.current_speed = np.asarray(current_speed, dtype=float).reshape(3)

        # Actuator matrix for BlueROV2 Heavy
        self.B = self._build_actuator_matrix()

        # ============================================================
        # System dimensions and input constraints (for SimulationSystem)
        # ============================================================
        self.state_dim = 13  # [p(3), q(4), nu(6)]
        self.input_dim = 6  # [Fx, Fy, Fz, Mx, My, Mz]

        # Input wrench limits (body frame)
        # Force limits based on 8 thrusters with ~40N each = ~320N total
        # Conservative estimate: ~200N sustained force per direction
        # Torque limits scaled appropriately for underwater vehicle
        force_limit = 200.0  # Newtons
        torque_limit = 50.0  # Newton-meters

        self.u_min = np.array(
            [
                -force_limit,
                -force_limit,
                -force_limit,
                -torque_limit,
                -torque_limit,
                -torque_limit,
            ],
            dtype=float,
        )
        self.u_max = np.array(
            [
                force_limit,
                force_limit,
                force_limit,
                torque_limit,
                torque_limit,
                torque_limit,
            ],
            dtype=float,
        )
        self.u_norm_max = None  # No norm constraint; handled per-component

    # ============================================================
    # Public interface expected by the controller
    # ============================================================

    def mass_matrix(self, chi: np.ndarray, nu: np.ndarray) -> np.ndarray:
        return self.M.copy()

    def coriolis(self, chi: np.ndarray, nu: np.ndarray) -> np.ndarray:
        return self._coriolis(np.asarray(nu, dtype=float).reshape(6))

    def damping(self, chi: np.ndarray, nu: np.ndarray) -> np.ndarray:
        """
        Returns damping computed on the provided velocity `nu`.

        Note:
            This interface uses the passed velocity directly. In the full
            continuous-time dynamics, relative velocity w.r.t. current is used.
        """
        return self._damping(np.asarray(nu, dtype=float).reshape(6))

    def restoring(self, chi: np.ndarray, nu: np.ndarray) -> np.ndarray:
        """
        Hydrostatic restoring vector using quaternion attitude.
        """
        chi = np.asarray(chi, dtype=float).reshape(7)
        q = chi[3:7]
        return self._restoring_from_quaternion(q)

    def translational_partition(
        self, chi: np.ndarray, nu: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Reduced translational partition:

            M_v * dot(v) + h_v = tau_v + d_v

        using the translational block of the full model.
        """
        chi = np.asarray(chi, dtype=float).reshape(7)
        nu = np.asarray(nu, dtype=float).reshape(6)

        M = self.mass_matrix(chi, nu)
        C = self.coriolis(chi, nu)
        D = self.damping(chi, nu)
        g = self.restoring(chi, nu)

        h_full = C @ nu + D @ nu + g
        M_v = M[:3, :3].copy()
        h_v = h_full[:3].copy()
        return M_v, h_v

    def actuator_matrix(self) -> np.ndarray:
        return self.B.copy()

    # ============================================================
    # Internal model terms
    # ============================================================

    def _coriolis(self, nu: np.ndarray) -> np.ndarray:
        """
        Total Coriolis-centripetal matrix CRB + CA.
        """
        u, v, w, p, q, r = nu

        CRB = np.zeros((6, 6), dtype=float)

        CRB[0, 4] = self.m * w
        CRB[0, 5] = -self.m * v
        CRB[1, 3] = -self.m * w
        CRB[1, 5] = self.m * u
        CRB[2, 3] = self.m * v
        CRB[2, 4] = -self.m * u

        CRB[3, 1] = self.m * w
        CRB[3, 2] = -self.m * v
        CRB[3, 4] = self.Iz * r
        CRB[3, 5] = -self.Iy * q

        CRB[4, 0] = -self.m * w
        CRB[4, 2] = self.m * u
        CRB[4, 3] = -self.Iz * r
        CRB[4, 5] = self.Ix * p

        CRB[5, 0] = self.m * v
        CRB[5, 1] = -self.m * u
        CRB[5, 3] = self.Iy * q
        CRB[5, 4] = -self.Ix * p

        CA = np.zeros((6, 6), dtype=float)

        CA[0, 4] = -self.Zw_dot * w
        CA[0, 5] = self.Yv_dot * v
        CA[1, 3] = self.Zw_dot * w
        CA[1, 5] = -self.Xu_dot * u
        CA[2, 3] = -self.Yv_dot * v
        CA[2, 4] = self.Xu_dot * u

        CA[3, 1] = -self.Zw_dot * w
        CA[3, 2] = self.Yv_dot * v
        CA[3, 4] = -self.Nr_dot * r
        CA[3, 5] = self.Mq_dot * q

        CA[4, 0] = self.Zw_dot * w
        CA[4, 2] = -self.Xu_dot * u
        CA[4, 3] = self.Nr_dot * r
        CA[4, 5] = -self.Kp_dot * p

        CA[5, 0] = -self.Yv_dot * v
        CA[5, 1] = self.Xu_dot * u
        CA[5, 3] = -self.Mq_dot * q
        CA[5, 4] = self.Kp_dot * p

        return CRB + CA

    def _damping(self, nu_r: np.ndarray) -> np.ndarray:
        """
        Linear + quadratic damping using relative velocity nu_r.
        """
        u_r, v_r, w_r, p_r, q_r, r_r = nu_r

        D = np.zeros((6, 6), dtype=float)
        D[0, 0] = -self.Xu - self.Xu_abs * abs(u_r)
        D[1, 1] = -self.Yv - self.Yv_abs * abs(v_r)
        D[2, 2] = -self.Zw - self.Zw_abs * abs(w_r)
        D[3, 3] = -self.Kp - self.Kp_abs * abs(p_r)
        D[4, 4] = -self.Mq - self.Mq_abs * abs(q_r)
        D[5, 5] = -self.Nr - self.Nr_abs * abs(r_r)
        return D

    def _restoring_from_quaternion(self, q: np.ndarray) -> np.ndarray:
        """
        Hydrostatic restoring forces/moments from quaternion attitude.

        For compatibility with the previous model, this uses roll/pitch
        extracted from the quaternion.
        """
        phi, theta, psi = quaternion_to_roll_pitch_yaw(q)

        W_minus_B = self.W - self.Buoy
        sphi = np.sin(phi)
        cphi = np.cos(phi)
        sth = np.sin(theta)
        cth = np.cos(theta)

        gvec = np.zeros(6, dtype=float)
        gvec[0] = W_minus_B * sth
        gvec[1] = -W_minus_B * cth * sphi
        gvec[2] = -W_minus_B * cth * cphi
        gvec[3] = (self.yb * self.Buoy) * cth * cphi - (
            self.zb * self.Buoy
        ) * cth * sphi
        gvec[4] = -(self.zb * self.Buoy) * sth - (self.xb * self.Buoy) * cth * cphi
        gvec[5] = (self.xb * self.Buoy) * cth * sphi + (self.yb * self.Buoy) * sth
        return gvec

    def _build_actuator_matrix(self) -> np.ndarray:
        """
        BlueROV2 Heavy actuator configuration matrix B such that tau = B f.
        """
        lx = 0.150
        ly = 0.150
        lz = 0.010

        c45 = np.cos(np.pi / 4.0)
        s45 = np.sin(np.pi / 4.0)

        B = np.array(
            [
                [s45, s45, -s45, -s45, 0.0, 0.0, 0.0, 0.0],  # Fx
                [c45, -c45, c45, -c45, 0.0, 0.0, 0.0, 0.0],  # Fy
                [0.0, 0.0, 0.0, 0.0, -1.0, -1.0, -1.0, -1.0],  # Fz
                [0.0, 0.0, 0.0, 0.0, -ly, ly, -ly, ly],  # Mx
                [0.0, 0.0, 0.0, 0.0, lx, lx, -lx, -lx],  # My
                [lz, -lz, lz, -lz, 0.0, 0.0, 0.0, 0.0],  # Mz
            ],
            dtype=float,
        )
        return B

    # ============================================================
    # Dynamics
    # ============================================================

    def dynamics(
        self, x: np.ndarray, tau_body: np.ndarray, dt: float = 0.02
    ) -> np.ndarray:
        """
        Continuous-time dynamics:

            x = [p(3), q(4), nu(6)]
            tau_body = [Fx, Fy, Fz, Mx, My, Mz]

        Returns
        -------
        x_dot : np.ndarray, shape (13,)
        """
        x = np.asarray(x, dtype=float).reshape(13)
        tau_body = np.asarray(tau_body, dtype=float).reshape(6)

        # unpack
        p = x[0:3]
        q = normalize_vector(x[3:7])
        nu = x[7:13]

        v_body = nu[0:3]
        omega_body = nu[3:6]

        # transforms
        R_b2n = rotation_matrix_from_quaternion(q)
        R_n2b = R_b2n.T

        # current compensation in translation
        v_current_b = R_n2b @ self.current_speed
        nu_r = nu.copy()
        nu_r[0:3] -= v_current_b

        # dynamics terms
        C = self._coriolis(nu)
        D = self._damping(nu_r)
        gvec = self._restoring_from_quaternion(q)

        rhs = tau_body - C @ nu - D @ nu_r - gvec
        nu_dot = self.Minv @ rhs

        # kinematics
        p_dot = R_b2n @ v_body
        q_dot = quaternion_derivative_body_rates(q, omega_body)

        x_dot = np.concatenate([p_dot, q_dot, nu_dot])
        return x_dot

    # ============================================================
    # Optional helper
    # ============================================================

    def normalize_state_quaternion(self, x: np.ndarray) -> np.ndarray:
        """
        Return a copy of the state with a normalized quaternion.
        Useful after numerical integration.
        """
        x = np.asarray(x, dtype=float).reshape(13).copy()
        x[3:7] = normalize_vector(x[3:7])
        return x
