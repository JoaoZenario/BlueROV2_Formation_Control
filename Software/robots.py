import numpy as np


from systems import (
    LinearSystem,
    InputAffineNLSystem,
    SimulationSystem,
)
from rot_utils import quat2rotm, omega_matrix

"""This file implements general models and specific robot classes using the system interfaces defined in systems.py."""


# ----------------------------------------- #
# ------------- General Models ------------ #


class SingleIntegrator(LinearSystem):
    """n-D Single integrator: x_dot = u"""

    def __init__(
        self,
        dim: int,
        u_min: np.ndarray | None = None,
        u_max: np.ndarray | None = None,
        u_norm_max: float | None = None,
    ):
        A = np.zeros((dim, dim))
        B = np.eye(dim)
        super().__init__(A, B, dim, dim, u_min, u_max, u_norm_max)


class DoubleIntegrator(LinearSystem):
    """n-D Double integrator: p_dot = v, v_dot = u"""

    def __init__(
        self,
        dim: int,
        u_min: np.ndarray | None = None,
        u_max: np.ndarray | None = None,
        u_norm_max: float | None = None,
    ):
        A = np.block(
            [
                [np.zeros((dim, dim)), np.eye(dim)],
                [np.zeros((dim, dim)), np.zeros((dim, dim))],
            ]
        )
        B = np.block([[np.zeros((dim, dim))], [np.eye(dim)]])
        super().__init__(A, B, 2 * dim, dim, u_min, u_max, u_norm_max)


# ----------------------------------------- #
# ------------- Robot Models -------------- #


class Unicycle(InputAffineNLSystem):
    """Unicycle kinematic model."""

    def __init__(
        self,
        u_min: np.ndarray | None = None,
        u_max: np.ndarray | None = None,
    ):
        def f(x):
            return np.zeros(3)

        def G(x):
            theta = x[2]
            return np.array([[np.cos(theta), 0], [np.sin(theta), 0], [0, 1]])

        super().__init__(
            f,
            G,
            state_dim=3,
            input_dim=2,
            u_min=u_min,
            u_max=u_max,
        )


class ATMOSPlanar(InputAffineNLSystem):
    """
    Planar ATMOS robot model.

    State:
        x = [px, py, yaw, vx, vy, omega] ∈ R^2 x S^1 x R^3

    Input:
        u ∈ R^m  (thruster forces)

    Thruster mapping:
        wrench_world = R(yaw) @ B_body @ u
    """

    def __init__(
        self,
        m: float = 16.8,
        inertia: float = 0.1594,
        max_thrust=1.5,
        max_torque=0.5,
    ):
        """
        Parameters
        ----------
        m : float
            Mass
        inertia : float
            Moment of inertia (out-of-plane)
        """
        self.m = m
        self.I = inertia

        u_min = np.hstack((-max_thrust * np.ones(2), -max_torque * np.ones(1)))
        u_max = np.hstack((max_thrust * np.ones(2), max_torque * np.ones(1)))

        state_dim = 6
        input_dim = 3

        def f(x):
            px, py, yaw, vx, vy, omega = x
            return np.array(
                [
                    vx,  # ṗx
                    vy,  # ṗy
                    omega,  # θ̇
                    0.0,  # v̇x
                    0.0,  # v̇y
                    0.0,  # ω̇
                ]
            )

        def G(x):
            _, _, yaw, _, _, _ = x

            c, s = np.cos(yaw), np.sin(yaw)

            # World-from-body rotation for planar forces
            R = np.array(
                [
                    [c, -s],
                    [s, c],
                ]
            )

            Gmat = np.zeros((state_dim, input_dim))
            # Linear acceleration (rotation of forces to world frame)
            Gmat[3:5, 0:2] = (1.0 / self.m) * R

            # Angular acceleration
            Gmat[5, 2] = 1.0 / self.I

            return Gmat

        super().__init__(
            f, G, state_dim=state_dim, input_dim=input_dim, u_min=u_min, u_max=u_max
        )


class EulerLagrangeSystem(InputAffineNLSystem):
    """Generic Euler-Lagrange system: M(q) q_ddot + C(q, q_dot) q_dot + g(q) = tau."""

    def __init__(
        self,
        M: np.ndarray,
        C: np.ndarray,
        g: np.ndarray,
        dof: int,
        u_min=None,
        u_max=None,
        u_norm_max=None,
    ):
        def f(x):
            q, v = x[:dof], x[dof:]
            Minv = np.linalg.inv(M(q))
            return np.concatenate([v, -Minv @ (C(q, v) @ v + g(q))])

        def G(x):
            q = x[:dof]
            Minv = np.linalg.inv(M(q))
            return np.block([[np.zeros((dof, dof))], [Minv]])

        super().__init__(f, G, state_dim=2 * dof, input_dim=dof)


class PlanarQuadrotor(InputAffineNLSystem):
    """Planar quadrotor model (2D): state = [x, z, theta, xdot, zdot, thetadot]."""

    def __init__(self, m=1.0, I=0.01, l=0.25, g=9.81):
        self.m = m
        self.I = I
        self.l = l
        self.g = g

        def f(x):
            _, _, theta, xdot, zdot, thetadot = x
            return np.array([xdot, zdot, thetadot, 0.0, -g, 0.0])

        def G(x):
            _, _, theta, *_ = x
            s, c = np.sin(theta), np.cos(theta)
            return np.array(
                [
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [-s / m, -s / m],
                    [c / m, c / m],
                    [l / I, -l / I],
                ]
            )

        super().__init__(f, G, state_dim=6, input_dim=2)


class QuadrotorQuat(InputAffineNLSystem):
    """Full 3D quadrotor model with quaternion attitude.

    State: [p(3), v(3), q(4), ω(3)] ∈ ℝ¹³
    Input: [T, τx, τy, τz] ∈ ℝ⁴
    """

    def __init__(self, m=1.0, J=np.diag([0.01, 0.01, 0.02]), g=9.81):
        self.m = m
        self.J = J
        self.Jinv = np.linalg.inv(J)
        self.g = g

        def f(x):
            v = x[3:6]
            q = x[6:10]
            omega = x[10:13]

            dp = v
            dv = np.array([0, 0, -self.g])
            dq = 0.5 * omega_matrix(omega) @ q
            domega = -self.Jinv @ np.cross(omega, self.J @ omega)

            return np.concatenate([dp, dv, dq, domega])

        def G(x):
            q = x[6:10]
            R = quat2rotm(q)

            Gmat = np.zeros((13, 4))
            Gmat[3:6, 0] = (1 / self.m) * R @ np.array([0, 0, 1])
            Gmat[10:13, 1:4] = self.Jinv
            return Gmat

        super().__init__(f, G, state_dim=13, input_dim=4)


class QuadrotorSimulator(SimulationSystem):
    """Specialized simulator for quaternion-based quadrotor model."""

    def __init__(self, quadrotor: "QuadrotorQuat", integrator, x0, dt):
        super().__init__(quadrotor, integrator, x0, dt)
        assert quadrotor.state_dim == 13, "Expected 13D state for QuadrotorQuat."

    def step(self, u):
        """Integrate one step and renormalize quaternion."""
        super().step(u)

        # Normalize quaternion (elements 6:10)
        q = self.x[6:10]
        self.x[6:10] = q / np.linalg.norm(q)