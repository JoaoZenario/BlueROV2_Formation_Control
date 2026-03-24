from abc import ABC, abstractmethod
import numpy as np
from typing import Callable

"""This file implements generic systems models which are used by specific robot classes."""

# ----------------------------------------- #
# ------------- System Models ------------- #


class SystemBase(ABC):
    """Abstract base class for dynamical systems."""

    def __init__(
        self,
        state_dim: int,
        input_dim: int,
        u_min: np.ndarray | None = None,
        u_max: np.ndarray | None = None,
        u_norm_max: float | None = None,
    ):
        self.state_dim = state_dim
        self.input_dim = input_dim

        if u_min is not None:
            u_min = np.asarray(u_min, dtype=float)
            if u_min.shape != (input_dim,):
                raise ValueError(f"u_min shape {u_min.shape} != (input_dim,)")
        if u_max is not None:
            u_max = np.asarray(u_max, dtype=float)
            if u_max.shape != (input_dim,):
                raise ValueError(f"u_max shape {u_max.shape} != (input_dim,)")

        self.u_min = u_min.copy() if u_min is not None else None
        self.u_max = u_max.copy() if u_max is not None else None
        self.u_norm_max = float(u_norm_max) if u_norm_max is not None else None

    @abstractmethod
    def dynamics(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Compute the time derivative of the state."""
        pass

    def output(self, x: np.ndarray) -> np.ndarray:
        """Optionally define system output y = h(x)."""
        return x  # default: full state


class InputAffineNLSystem(SystemBase):
    """A generic input-affine nonlinear system.

    The system is described by the dynamics:
        x_dot = f(x) + G(x) u
    where x is the state, u is the input, f is the drift dynamics,
    and G collects the inputs vector fields.
    Attributes:
        f: A callable representing the drift dynamics f(x).
        G: A callable representing the input vector fields G(x).
    """

    def __init__(
        self,
        f: Callable[[np.ndarray], np.ndarray],
        G: Callable[[np.ndarray], np.ndarray],
        state_dim: int,
        input_dim: int,
        u_min: np.ndarray | None = None,
        u_max: np.ndarray | None = None,
        u_norm_max: float | None = None,
    ):
        """Initializes the InputAffineNLSystem with drift dynamics and input vector fields.

        Args:
            f: A callable representing the drift dynamics f(x).
            G: A callable representing the input vector fields G(x).
        """
        super().__init__(state_dim, input_dim, u_min, u_max, u_norm_max)
        self.f = f
        self.G = G

    def dynamics(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Computes the system dynamics at state x with input u.

        Args:
            x: The current state of the system.
            u: The current input to the system.

        Returns:
            The time derivative of the state (x_dot).
        """

        if x.shape != (self.state_dim,):
            raise ValueError(f"x shape {x.shape} != (state_dim,)")
        if u.shape != (self.input_dim,):
            raise ValueError(f"u shape {u.shape} != (input_dim,)")

        return self.f(x) + self.G(x) @ u


class LinearSystem(SystemBase):
    """A generic linear system.

    The system is described by the dynamics:
        x_dot = A x + B u
    where x is the state, u is the input, A is the state matrix, and B is the input matrix.
    Attributes:
        A: The state matrix.
        B: The input matrix.
    """

    def __init__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        state_dim: int,
        input_dim: int,
        u_min: np.ndarray | None = None,
        u_max: np.ndarray | None = None,
        u_norm_max: float | None = None,
    ):
        """Initializes the LinearSystem with state and input matrices.

        Args:
            A: The state matrix.
            B: The input matrix.
        """
        super().__init__(state_dim, input_dim, u_min, u_max, u_norm_max)
        self.A = A
        self.B = B

    def dynamics(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Computes the system dynamics at state x with input u.

        Args:
            x: The current state of the system.
            u: The current input to the system.

        Returns:
            The time derivative of the state (x_dot).
        """
        return self.A @ x + self.B @ u


# ----------------------------------------- #
# ------------- Integrators --------------- #


class Integrator(ABC):
    """Generic ODE integrator interface."""

    @abstractmethod
    def __call__(
        self, system: SystemBase, x: np.ndarray, u: np.ndarray, dt: float
    ) -> np.ndarray:
        raise NotImplementedError


class EulerIntegrator(Integrator):
    """Forward Euler integration."""

    def __call__(self, system, x, u, dt):
        return x + dt * system.dynamics(x, u)


class RK4Integrator(Integrator):
    """Fourth-order Runge-Kutta integration."""

    def __call__(self, system, x, u, dt):
        f = system.dynamics
        k1 = f(x, u)
        k2 = f(x + 0.5 * dt * k1, u)
        k3 = f(x + 0.5 * dt * k2, u)
        k4 = f(x + dt * k3, u)
        return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


# ----------------------------------------- #
# ---------- Simulation Model ------------- #


class SimulationSystem:
    """Wraps a system and an integrator, maintaining the internal state."""

    def __init__(
        self,
        system: InputAffineNLSystem,
        x0: np.ndarray,
        dt: float,
        integrator: Integrator = RK4Integrator(),
        u_min: np.ndarray | None = None,
        u_max: np.ndarray | None = None,
        u_norm_max: float | None = None,
        noise_fn: Callable[[np.ndarray, float], np.ndarray] | None = None,
    ):
        self.system = system
        self.integrator = integrator
        self.x = np.asarray(x0, dtype=float).copy()
        self.dt = dt
        self.t = 0.0

        self.u_min = (
            np.asarray(u_min, dtype=float).copy()
            if u_min is not None
            else (
                system.u_min.copy()
                if getattr(system, "u_min", None) is not None
                else None
            )
        )
        self.u_max = (
            np.asarray(u_max, dtype=float).copy()
            if u_max is not None
            else (
                system.u_max.copy()
                if getattr(system, "u_max", None) is not None
                else None
            )
        )
        self.u_norm_max = (
            u_norm_max
            if u_norm_max is not None
            else getattr(system, "u_norm_max", None)
        )

        self.noise_fn = noise_fn  # function handle for process noise

    def apply_input_constraints(self, u):
        """Clamp input within [u_min, u_max]."""
        if self.u_min is not None:
            u = np.maximum(u, self.u_min)
        if self.u_max is not None:
            u = np.minimum(u, self.u_max)
        if self.u_norm_max is not None:
            u_norm = np.linalg.norm(u)
            if u_norm > self.u_norm_max:
                u = u / u_norm * self.u_norm_max
        return u

    def add_process_noise(self, x):
        """Optionally add process noise after integration. TO BE CHANGED"""
        if self.noise_fn is not None:
            return x + self.noise_fn(x, self.t)
        return x

    def step(self, u):
        u = self.apply_input_constraints(u)
        self.x = self.integrator(self.system, self.x, u, self.dt)
        self.x = self.add_process_noise(self.x)
        self.t += self.dt

    def reset(self, x0: np.ndarray, t0: float = 0.0):
        """Reset the system state."""
        self.x = x0
        self.t = t0


# ----------------------------------------- #
# ------------- Noise Model --------------- #


def gaussian_process_noise(x, t, sigma=0.001):
    """Add Gaussian noise."""
    return np.random.randn(*x.shape) * sigma