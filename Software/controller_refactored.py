from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np


from math_utils import smooth_saturate, skew
from rot_utils import rotation_matrix_from_quaternion

from scipy import sparse
from scipy.spatial.transform import Rotation as R
from qpsolvers import solve_qp

Array = np.ndarray


@dataclass
class BarrierConfig:
    """
    Configuration parameters shared by all controllers.

    Attributes
    ----------
    dim : int
        Dimension of the relative-position space (typically 2 or 3).
    d_min : float
        Minimum admissible inter-agent distance for collision avoidance.
    d_max : float
        Maximum admissible inter-agent distance for connectivity maintenance.
    k_form : float
        Gain of the nominal quadratic formation potential.
    use_connectivity : bool
        Whether the connectivity barrier is active.
    use_collision : bool
        Whether the collision-avoidance barrier is active.
    use_recentered : bool
        Whether to use the recentered barrier or the raw barrier.
    eps : float
        Small positive constant used to prevent numerical division by zero.
    """

    dim: int
    d_min: float
    d_max: float
    k_form: float = 1.0
    use_connectivity: bool = True
    use_collision: bool = True
    use_recentered: bool = True
    eps: float = 1e-9

    def __post_init__(self) -> None:
        if self.dim <= 0:
            raise ValueError("dim must be positive.")
        if self.d_min < 0.0:
            raise ValueError("d_min must be nonnegative.")
        if self.d_max <= self.d_min:
            raise ValueError("d_max must be strictly greater than d_min.")
        if self.k_form < 0.0:
            raise ValueError("k_form must be nonnegative.")
        if self.eps <= 0.0:
            raise ValueError("eps must be positive.")


class ControllerBase(ABC):
    """
    Abstract base class for multi-agent controllers.

    This class only contains model-independent functionality, such as:
    - barrier functions for connectivity and collision avoidance,
    - gradients of the barriers,
    - recentering around a desired relative displacement,
    - nominal formation potential and gradient,
    - total edge potential and gradient,
    - optional command filters.

    Subclasses should implement `do_control(...)`.
    """

    def __init__(self, robot_id: int, config: BarrierConfig):
        self.robot_id = robot_id
        self.cfg = config

    # ============================================================
    # Utility methods
    # ============================================================

    def _as_vector(self, x: Array, name: str) -> Array:
        """Convert input to a flat NumPy vector and validate its dimension."""
        x = np.asarray(x, dtype=float).reshape(-1)
        if x.size != self.cfg.dim:
            raise ValueError(
                f"{name} must have dimension {self.cfg.dim}, got {x.size}."
            )
        return x

    def _safe_positive(self, value: float) -> float:
        """
        Ensure a positive denominator for numerically robust computations.

        This clips the value from below by `eps`.
        """
        return max(float(value), self.cfg.eps)

    def norm_sq(self, x: Array) -> float:
        """Return the squared Euclidean norm."""
        x = np.asarray(x, dtype=float).reshape(-1)
        return float(x @ x)

    def is_admissible(self, d: Array) -> bool:
        """
        Check whether a relative displacement lies in the admissible set

            d_min < ||d|| < d_max.
        """
        d = self._as_vector(d, "d")
        d_norm = np.linalg.norm(d)
        return (self.cfg.d_min < d_norm) and (d_norm < self.cfg.d_max)

    def validate_desired_relative_state(self, d_des: Array) -> None:
        """Raise if the desired relative displacement is not admissible."""
        d_des = self._as_vector(d_des, "d_des")
        if not self.is_admissible(d_des):
            raise ValueError(
                "Desired relative displacement is not admissible: "
                f"||d_des|| must satisfy {self.cfg.d_min} < ||d_des|| < {self.cfg.d_max}."
            )

    # ============================================================
    # Connectivity barrier
    # B_conn(d) = ln( d_max^2 / (d_max^2 - ||d||^2) )
    # ∇B_conn(d) = 2 d / (d_max^2 - ||d||^2)
    # ============================================================

    def connectivity_barrier(self, d: Array) -> float:
        d = self._as_vector(d, "d")
        if not self.cfg.use_connectivity:
            return 0.0

        denom = self._safe_positive(self.cfg.d_max**2 - self.norm_sq(d))
        return float(np.log(self.cfg.d_max**2 / denom))

    def grad_connectivity_barrier(self, d: Array) -> Array:
        d = self._as_vector(d, "d")
        if not self.cfg.use_connectivity:
            return np.zeros(self.cfg.dim)

        denom = self._safe_positive(self.cfg.d_max**2 - self.norm_sq(d))
        return (2.0 / denom) * d

    # ============================================================
    # Collision barrier
    # B_coll(d) = ln( d_min^2 / (||d||^2 - d_min^2) )
    # ∇B_coll(d) = - 2 d / (||d||^2 - d_min^2)
    # ============================================================

    def collision_barrier(self, d: Array) -> float:
        d = self._as_vector(d, "d")
        if not self.cfg.use_collision:
            return 0.0

        denom = self._safe_positive(self.norm_sq(d) - self.cfg.d_min**2)
        return float(np.log(self.cfg.d_min**2 / denom))

    def grad_collision_barrier(self, d: Array) -> Array:
        d = self._as_vector(d, "d")
        if not self.cfg.use_collision:
            return np.zeros(self.cfg.dim)

        denom = self._safe_positive(self.norm_sq(d) - self.cfg.d_min**2)
        return -(2.0 / denom) * d

    # ============================================================
    # Combined barrier
    # B(d) = B_conn(d) + B_coll(d)
    # ∇B(d) = ∇B_conn(d) + ∇B_coll(d)
    # ============================================================

    def barrier(self, d: Array) -> float:
        """Combined scalar barrier."""
        return self.connectivity_barrier(d) + self.collision_barrier(d)

    def grad_barrier(self, d: Array) -> Array:
        """Gradient of the combined scalar barrier."""
        return self.grad_connectivity_barrier(d) + self.grad_collision_barrier(d)

    # ============================================================
    # Recentered barrier
    #
    # B_rec(d) = B(d) - B(d_des) - ∇B(d_des)^T (d - d_des)
    # ∇B_rec(d) = ∇B(d) - ∇B(d_des)
    # ============================================================

    def recentered_barrier(self, d: Array, d_des: Array) -> float:
        """
        Scalar recentered barrier around the desired relative displacement.
        """
        d = self._as_vector(d, "d")
        d_des = self._as_vector(d_des, "d_des")
        self.validate_desired_relative_state(d_des)

        B_d = self.barrier(d)
        B_des = self.barrier(d_des)
        grad_B_des = self.grad_barrier(d_des)

        return float(B_d - B_des - grad_B_des @ (d - d_des))

    def grad_recentered_barrier(self, d: Array, d_des: Array) -> Array:
        """
        Gradient of the recentered barrier:

            ∇B_rec(d) = ∇B(d) - ∇B(d_des)
        """
        d = self._as_vector(d, "d")
        d_des = self._as_vector(d_des, "d_des")
        self.validate_desired_relative_state(d_des)

        return self.grad_barrier(d) - self.grad_barrier(d_des)

    # ============================================================
    # Nominal formation potential
    #
    # Phi_f(d) = 0.5 * k_form * ||d - d_des||^2
    # ∇Phi_f(d) = k_form * (d - d_des)
    # ============================================================

    def formation_potential(self, d: Array, d_des: Array) -> float:
        d = self._as_vector(d, "d")
        d_des = self._as_vector(d_des, "d_des")
        err = d - d_des
        return 0.5 * self.cfg.k_form * self.norm_sq(err)

    def grad_formation_potential(self, d: Array, d_des: Array) -> Array:
        d = self._as_vector(d, "d")
        d_des = self._as_vector(d_des, "d_des")
        return self.cfg.k_form * (d - d_des)

    # ============================================================
    # Total edge potential
    #
    # Option A in your derivation:
    # Phi(d) = Phi_f(d) + B_rec(d)     if use_recentered
    #        = Phi_f(d) + B(d)         otherwise
    # ============================================================

    def edge_potential(self, d: Array, d_des: Array) -> float:
        """
        Total scalar edge potential used at the outer loop.
        """
        if self.cfg.use_recentered:
            return self.formation_potential(d, d_des) + self.recentered_barrier(
                d, d_des
            )
        return self.formation_potential(d, d_des) + self.barrier(d)

    def grad_edge_potential(self, d: Array, d_des: Array) -> Array:
        """
        Gradient of the total edge potential.
        """
        grad_form = self.grad_formation_potential(d, d_des)
        if self.cfg.use_recentered:
            return grad_form + self.grad_recentered_barrier(d, d_des)
        return grad_form + self.grad_barrier(d)

    # ============================================================
    # Batch utilities over neighbors / edges
    # ============================================================

    def total_edge_potential(
        self,
        relative_states: Array,
        desired_relative_states: Array,
    ) -> float:
        """
        Sum of edge potentials over a set of relative states.
        """
        relative_states = np.asarray(relative_states, dtype=float)
        desired_relative_states = np.asarray(desired_relative_states, dtype=float)

        if relative_states.shape != desired_relative_states.shape:
            raise ValueError(
                "relative_states and desired_relative_states must have the same shape."
            )
        if relative_states.ndim != 2 or relative_states.shape[1] != self.cfg.dim:
            raise ValueError(
                f"relative_states must have shape (n_edges, {self.cfg.dim})."
            )

        total = 0.0
        for d, d_des in zip(relative_states, desired_relative_states):
            total += self.edge_potential(d, d_des)
        return float(total)

    def edge_gradients(
        self,
        relative_states: Array,
        desired_relative_states: Array,
    ) -> Array:
        """
        Return the gradient of the total edge potential for each edge.

        Output shape: (n_edges, dim)
        """
        relative_states = np.asarray(relative_states, dtype=float)
        desired_relative_states = np.asarray(desired_relative_states, dtype=float)

        if relative_states.shape != desired_relative_states.shape:
            raise ValueError(
                "relative_states and desired_relative_states must have the same shape."
            )
        if relative_states.ndim != 2 or relative_states.shape[1] != self.cfg.dim:
            raise ValueError(
                f"relative_states must have shape (n_edges, {self.cfg.dim})."
            )

        grads = np.zeros_like(relative_states, dtype=float)
        for k, (d, d_des) in enumerate(zip(relative_states, desired_relative_states)):
            grads[k] = self.grad_edge_potential(d, d_des)
        return grads

    # ============================================================
    # Model-independent command filters
    # ============================================================

    def first_order_filter_step(
        self,
        y_f: Array,
        y_des: Array,
        dt: float,
        lam: float,
    ) -> Tuple[Array, Array]:
        """
        One Euler step of the first-order filter

            ẏ_f = -λ y_f + λ y_des

        Returns
        -------
        y_f_next : np.ndarray
            Updated filtered signal.
        dy_f : np.ndarray
            Current derivative provided by the filter.
        """
        y_f = np.asarray(y_f, dtype=float).reshape(-1)
        y_des = np.asarray(y_des, dtype=float).reshape(-1)
        if y_f.shape != y_des.shape:
            raise ValueError("y_f and y_des must have the same shape.")
        if lam <= 0.0:
            raise ValueError("lam must be positive.")
        if dt <= 0.0:
            raise ValueError("dt must be positive.")

        dy_f = -lam * y_f + lam * y_des
        y_f_next = y_f + dt * dy_f
        return y_f_next, dy_f

    def second_order_filter_step(
        self,
        y_f: Array,
        dy_f: Array,
        y_des: Array,
        dt: float,
        wn: float,
        zeta: float = 1.0,
    ) -> Tuple[Array, Array, Array]:
        """
        One Euler step of the second-order filter

            ÿ_f + 2 ζ ω_n ẏ_f + ω_n² (y_f - y_des) = 0

        Returns
        -------
        y_f_next : np.ndarray
            Updated filtered signal.
        dy_f_next : np.ndarray
            Updated derivative state.
        ddy_f : np.ndarray
            Current second derivative.
        """
        y_f = np.asarray(y_f, dtype=float).reshape(-1)
        dy_f = np.asarray(dy_f, dtype=float).reshape(-1)
        y_des = np.asarray(y_des, dtype=float).reshape(-1)

        if y_f.shape != y_des.shape or dy_f.shape != y_des.shape:
            raise ValueError("y_f, dy_f, and y_des must have the same shape.")
        if wn <= 0.0:
            raise ValueError("wn must be positive.")
        if zeta <= 0.0:
            raise ValueError("zeta must be positive.")
        if dt <= 0.0:
            raise ValueError("dt must be positive.")

        ddy_f = -2.0 * zeta * wn * dy_f - (wn**2) * (y_f - y_des)
        dy_f_next = dy_f + dt * ddy_f
        y_f_next = y_f + dt * dy_f_next
        return y_f_next, dy_f_next, ddy_f

    # ============================================================
    # Abstract API
    # ============================================================

    @abstractmethod
    def do_control(self, *args, **kwargs) -> Array:
        """
        Compute the control input for the specific model.
        """
        raise NotImplementedError


# ============================================================
# Single-integrator controller
# ============================================================


@dataclass
class SingleIntegratorConfig(BarrierConfig):
    """
    Configuration for the single-integrator controller.

    Attributes
    ----------
    c_outer : float
        Gain multiplying the descent direction of the total edge potential.
    v_max : float | None
        Optional velocity saturation bound. If None, no saturation is applied.
    """

    c_outer: float = 1.0
    v_max: Optional[float] = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.c_outer <= 0.0:
            raise ValueError("c_outer must be positive.")
        if self.v_max is not None and self.v_max <= 0.0:
            raise ValueError("v_max must be positive when provided.")


class SingleIntegratorController(ControllerBase):
    """
    Controller for single-integrator dynamics

        ṗ_i = u_i

    using the gradient of the total edge potential. The control law is

        u_i = -c_outer * sum_k grad Phi_k(d_k),

    where Phi_k contains:
    - the nominal quadratic formation potential,
    - the connectivity/collision barrier contribution,
    - optionally the recentered barrier formulation.
    """

    def __init__(self, robot_id: int, config: SingleIntegratorConfig):
        super().__init__(robot_id, config)
        self.cfg: SingleIntegratorConfig = config

    # ============================================================
    # Helpers
    # ============================================================

    def saturate_velocity(self, u: Array) -> Array:
        """
        Apply Euclidean-norm saturation to the velocity command.
        """
        u = np.asarray(u, dtype=float).reshape(-1)
        if self.cfg.v_max is None:
            return u

        u = smooth_saturate(u, max_norm=self.cfg.v_max)
        return u

    def compute_outer_descent_direction(
        self,
        relative_states: Array,
        desired_relative_states: Array,
    ) -> Tuple[Array, Array]:
        """
        Compute the summed gradient of the total edge potentials.

        Parameters
        ----------
        relative_states : np.ndarray, shape (n_edges, dim)
            Relative displacements d_k = p_i - p_j for all edges involving agent i.
        desired_relative_states : np.ndarray, shape (n_edges, dim)
            Desired relative displacements d_k^des for the same edges.

        Returns
        -------
        summed_gradient : np.ndarray, shape (dim,)
            Sum of all edge gradients.
        edge_gradients : np.ndarray, shape (n_edges, dim)
            Individual edge gradients.
        """
        edge_grads = self.edge_gradients(relative_states, desired_relative_states)
        summed_grad = np.sum(edge_grads, axis=0)
        return summed_grad, edge_grads

    # ============================================================
    # Main control law
    # ============================================================

    def do_control(
        self,
        relative_states: Array,
        desired_relative_states: Array,
        is_leader: bool = False,
        leader_velocity: Optional[Array] = None,
    ) -> Array:
        """
        Compute the velocity input for the single-integrator model.

        Parameters
        ----------
        relative_states : np.ndarray, shape (n_edges, dim)
            Relative displacements d_k = p_i - p_j for all edges involving agent i.
        desired_relative_states : np.ndarray, shape (n_edges, dim)
            Desired relative displacements d_k^des.
        is_leader : bool, default False
            If True, the controller returns either zero or the provided leader velocity.
        leader_velocity : np.ndarray | None, default None
            Optional prescribed leader velocity.

        Returns
        -------
        u_i : np.ndarray, shape (dim,)
            Velocity command for agent i.
        """
        # --------------------------------------------------------
        # Leader handling
        # --------------------------------------------------------
        if is_leader:
            if leader_velocity is None:
                return np.zeros(self.cfg.dim)
            leader_velocity = self._as_vector(leader_velocity, "leader_velocity")
            return self.saturate_velocity(leader_velocity)

        # --------------------------------------------------------
        # Validate inputs
        # --------------------------------------------------------
        relative_states = np.asarray(relative_states, dtype=float)
        desired_relative_states = np.asarray(desired_relative_states, dtype=float)

        if relative_states.shape != desired_relative_states.shape:
            raise ValueError(
                "relative_states and desired_relative_states must have the same shape."
            )
        if relative_states.ndim != 2 or relative_states.shape[1] != self.cfg.dim:
            raise ValueError(
                f"relative_states must have shape (n_edges, {self.cfg.dim})."
            )

        # --------------------------------------------------------
        # Compute the total edge-potential gradient
        # --------------------------------------------------------
        summed_grad, _ = self.compute_outer_descent_direction(
            relative_states=relative_states,
            desired_relative_states=desired_relative_states,
        )

        # Gradient-descent control law
        u_i = -self.cfg.c_outer * summed_grad

        # Optional saturation
        u_i = self.saturate_velocity(u_i)
        return u_i


@dataclass
class DoubleIntegratorConfig(BarrierConfig):
    """
    Configuration for the double-integrator controller.

    Attributes
    ----------
    c_outer : float
        Gain used to generate the virtual velocity from the gradient of the
        total edge potential.
    k_vel : float
        Gain on the velocity tracking error in the second backstepping step.
    v_max : float | None
        Optional saturation bound for the virtual/actual velocity.
    a_max : float | None
        Optional saturation bound for the acceleration command.
    filter_type : {"first_order", "second_order"}
        Command-filter type.
    dt : float
        Sampling time.
    lam : float
        First-order filter bandwidth.
    wn : float
        Second-order filter natural frequency.
    zeta : float
        Second-order filter damping ratio.
    """

    c_outer: float = 1.0
    k_vel: float = 1.0
    v_max: Optional[float] = None
    a_max: Optional[float] = None

    filter_type: str = "second_order"  # "first_order" or "second_order"
    dt: float = 0.01
    lam: float = 5.0
    wn: float = 5.0
    zeta: float = 1.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.c_outer <= 0.0:
            raise ValueError("c_outer must be positive.")
        if self.k_vel <= 0.0:
            raise ValueError("k_vel must be positive.")
        if self.v_max is not None and self.v_max <= 0.0:
            raise ValueError("v_max must be positive when provided.")
        if self.a_max is not None and self.a_max <= 0.0:
            raise ValueError("a_max must be positive when provided.")
        if self.filter_type not in {"first_order", "second_order"}:
            raise ValueError("filter_type must be 'first_order' or 'second_order'.")
        if self.dt <= 0.0:
            raise ValueError("dt must be positive.")
        if self.lam <= 0.0:
            raise ValueError("lam must be positive.")
        if self.wn <= 0.0:
            raise ValueError("wn must be positive.")
        if self.zeta <= 0.0:
            raise ValueError("zeta must be positive.")


class DoubleIntegratorController(ControllerBase):
    """
    Controller for the double-integrator model

        ṗ_i = v_i
        v̇_i = u_i

    based on command-filtered backstepping.

    Structure
    ---------
    1) Outer loop:
           v_des_i = -c_outer * sum_k grad Phi_k(d_k)

       where Phi_k is the total edge potential containing:
       - the nominal quadratic formation term,
       - the connectivity/collision barrier contribution,
       - optionally the recentered barrier.

    2) Command filter:
           provides a filtered virtual velocity v_f and its derivative dv_f.

    3) Inner loop:
           e_v = v_i - v_f
           u_i = dv_f - k_vel * e_v - summed_grad

       This is the standard backstepping structure consistent with
       V = V1 + 0.5 * ||e_v||^2, where V1 is the reduced-system Lyapunov
       function based on the total edge potential.
    """

    def __init__(self, robot_id: int, config: DoubleIntegratorConfig):
        super().__init__(robot_id, config)
        self.cfg: DoubleIntegratorConfig = config

        # Command-filter state
        self.v_f = np.zeros(self.cfg.dim)
        self.dv_f = np.zeros(self.cfg.dim)

    # ============================================================
    # Helpers
    # ============================================================

    def reset_filter_state(self, value: Optional[Array] = None) -> None:
        """
        Reset the command-filter internal state.

        Parameters
        ----------
        value : np.ndarray | None
            Optional reset value for the filtered command.
            If None, the filter is reset to zero.
        """
        if value is None:
            self.v_f[:] = 0.0
            self.dv_f[:] = 0.0
            return

        value = self._as_vector(value, "value")
        self.v_f[:] = value
        self.dv_f[:] = 0.0

    def saturate_velocity(self, v: Array) -> Array:
        """Apply Euclidean-norm saturation to a velocity vector."""
        v = np.asarray(v, dtype=float).reshape(-1)
        if self.cfg.v_max is None:
            return v

        v = smooth_saturate(v, max_norm=self.cfg.v_max)
        return v

    def saturate_acceleration(self, u: Array) -> Array:
        """Apply Euclidean-norm saturation to an acceleration vector."""
        u = np.asarray(u, dtype=float).reshape(-1)
        if self.cfg.a_max is None:
            return u

        norm_u = np.linalg.norm(u)
        if norm_u <= self.cfg.a_max:
            return u
        return (self.cfg.a_max / max(norm_u, self.cfg.eps)) * u

    def compute_outer_descent_direction(
        self,
        relative_states: Array,
        desired_relative_states: Array,
    ) -> Tuple[Array, Array]:
        """
        Compute the summed gradient of the total edge potentials.

        Returns
        -------
        summed_gradient : np.ndarray, shape (dim,)
            Sum of all edge gradients.
        edge_gradients : np.ndarray, shape (n_edges, dim)
            Individual edge gradients.
        """
        edge_grads = self.edge_gradients(relative_states, desired_relative_states)
        summed_grad = np.sum(edge_grads, axis=0)
        return summed_grad, edge_grads

    def compute_virtual_velocity(
        self,
        relative_states: Array,
        desired_relative_states: Array,
    ) -> Tuple[Array, Array]:
        """
        Compute the outer-loop virtual velocity

            v_des = -c_outer * sum_k grad Phi_k(d_k)

        Returns
        -------
        v_des : np.ndarray, shape (dim,)
            Virtual velocity command.
        edge_gradients : np.ndarray, shape (n_edges, dim)
            Individual edge gradients.
        """
        summed_grad, edge_grads = self.compute_outer_descent_direction(
            relative_states=relative_states,
            desired_relative_states=desired_relative_states,
        )
        v_des = -self.cfg.c_outer * summed_grad
        v_des = self.saturate_velocity(v_des)
        return v_des, edge_grads

    def filter_virtual_velocity(self, v_des: Array) -> Tuple[Array, Array]:
        """
        Update the internal command filter and return the filtered virtual
        velocity and its derivative.
        """
        v_des = self._as_vector(v_des, "v_des")

        if self.cfg.filter_type == "first_order":
            self.v_f, self.dv_f = self.first_order_filter_step(
                y_f=self.v_f,
                y_des=v_des,
                dt=self.cfg.dt,
                lam=self.cfg.lam,
            )
            return self.v_f.copy(), self.dv_f.copy()

        # second-order filter
        self.v_f, self.dv_f, _ = self.second_order_filter_step(
            y_f=self.v_f,
            dy_f=self.dv_f,
            y_des=v_des,
            dt=self.cfg.dt,
            wn=self.cfg.wn,
            zeta=self.cfg.zeta,
        )
        return self.v_f.copy(), self.dv_f.copy()

    # ============================================================
    # Main control law
    # ============================================================

    def do_control(
        self,
        relative_states: Array,
        desired_relative_states: Array,
        velocity_i: Array,
        is_leader: bool = False,
        leader_velocity: Optional[Array] = None,
        leader_acceleration: Optional[Array] = None,
    ) -> Array:
        """
        Compute the acceleration input for the double-integrator model.

        Parameters
        ----------
        relative_states : np.ndarray, shape (n_edges, dim)
            Relative displacements d_k = p_i - p_j.
        desired_relative_states : np.ndarray, shape (n_edges, dim)
            Desired relative displacements d_k^des.
        velocity_i : np.ndarray, shape (dim,)
            Current velocity of agent i.
        is_leader : bool, default False
            If True, use the provided leader command or return zero.
        leader_velocity : np.ndarray | None, default None
            Optional prescribed leader velocity.
        leader_acceleration : np.ndarray | None, default None
            Optional prescribed leader acceleration.

        Returns
        -------
        u_i : np.ndarray, shape (dim,)
            Acceleration command.
        """
        velocity_i = self._as_vector(velocity_i, "velocity_i")

        # --------------------------------------------------------
        # Leader handling
        # --------------------------------------------------------
        if is_leader:
            if leader_velocity is not None:
                leader_velocity = self._as_vector(leader_velocity, "leader_velocity")
                self.filter_virtual_velocity(leader_velocity)

            if leader_acceleration is None:
                return np.zeros(self.cfg.dim)

            leader_acceleration = self._as_vector(
                leader_acceleration, "leader_acceleration"
            )
            return self.saturate_acceleration(leader_acceleration)

        # --------------------------------------------------------
        # Validate inputs
        # --------------------------------------------------------
        relative_states = np.asarray(relative_states, dtype=float)
        desired_relative_states = np.asarray(desired_relative_states, dtype=float)

        if relative_states.shape != desired_relative_states.shape:
            raise ValueError(
                "relative_states and desired_relative_states must have the same shape."
            )
        if relative_states.ndim != 2 or relative_states.shape[1] != self.cfg.dim:
            raise ValueError(
                f"relative_states must have shape (n_edges, {self.cfg.dim})."
            )

        # --------------------------------------------------------
        # 1) Outer loop: reduced-system descent direction
        # --------------------------------------------------------
        summed_grad, _ = self.compute_outer_descent_direction(
            relative_states=relative_states,
            desired_relative_states=desired_relative_states,
        )

        v_des = -self.cfg.c_outer * summed_grad
        v_des = self.saturate_velocity(v_des)

        # --------------------------------------------------------
        # 2) Command-filtered virtual velocity
        # --------------------------------------------------------
        v_f, dv_f = self.filter_virtual_velocity(v_des)

        # --------------------------------------------------------
        # 3) Inner-loop tracking error
        # --------------------------------------------------------
        e_v = velocity_i - v_f

        # --------------------------------------------------------
        # 4) Backstepping acceleration law
        #
        # Reduced-system Lyapunov:
        #     V1 = sum_k Phi_k(d_k)
        #
        # Since v_des = -c_outer * summed_grad,
        # the standard command-filtered backstepping form is
        #
        #     u = dv_f - k_vel * e_v - summed_grad
        #
        # The last term is the passivating/coupling term linked to the
        # gradient of V1.
        # --------------------------------------------------------
        u_i = dv_f - self.cfg.k_vel * e_v - summed_grad

        # Optional saturation
        u_i = self.saturate_acceleration(u_i)
        return u_i


@dataclass
class BlueROV2Config(BarrierConfig):
    c_outer: float = 1.0
    k_v: float = 2.0

    filter_type: Literal["first_order", "second_order"] = "second_order"
    dt: float = 0.01
    lam: float = 5.0
    wn: float = 5.0
    zeta: float = 1.0

    v_max: Optional[float] = None
    tau_v_max: Optional[float] = None

    clf_alpha: float = 2.0
    clf_slack_weight: float = 1000.0
    clf_force_weight: float = 1.0
    thrust_weight: float = 1.0
    lgv_epsilon: float = 1e-6

    # saturation mode for translational virtual velocity / force
    saturation_switch_ratio: float = 4.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.c_outer <= 0.0:
            raise ValueError("c_outer must be positive.")
        if self.k_v <= 0.0:
            raise ValueError("k_v must be positive.")
        if self.filter_type not in {"first_order", "second_order"}:
            raise ValueError("filter_type must be 'first_order' or 'second_order'.")
        if self.dt <= 0.0:
            raise ValueError("dt must be positive.")
        if self.lam <= 0.0:
            raise ValueError("lam must be positive.")
        if self.wn <= 0.0:
            raise ValueError("wn must be positive.")
        if self.zeta <= 0.0:
            raise ValueError("zeta must be positive.")
        if self.v_max is not None and self.v_max <= 0.0:
            raise ValueError("v_max must be positive when provided.")
        if self.tau_v_max is not None and self.tau_v_max <= 0.0:
            raise ValueError("tau_v_max must be positive when provided.")
        if self.clf_alpha <= 0.0:
            raise ValueError("clf_alpha must be positive.")
        if self.clf_slack_weight <= 0.0:
            raise ValueError("clf_slack_weight must be positive.")
        if self.clf_force_weight <= 0.0:
            raise ValueError("clf_force_weight must be positive.")
        if self.thrust_weight <= 0.0:
            raise ValueError("thrust_weight must be positive.")
        if self.lgv_epsilon <= 0.0:
            raise ValueError("lgv_epsilon must be positive.")
        if self.saturation_switch_ratio <= 1.0:
            raise ValueError("saturation_switch_ratio should be > 1.0.")


class BlueROV2Controller(ControllerBase):
    """
    Refactored BlueROV2 low-level controller.

    Main structure
    --------------
    1) Outer loop:
           v_des_I = -c_outer * sum_k grad Phi_k(d_k)

    2) Command filter:
           v_f_I, dv_f_I

    3) Two realizations:
           - feedback-linearization with backstepping/passivating term
           - CLF-QP on translational force
           - CLF-QP directly on thrusters

    Important
    ---------
    The passivating term is now computed internally as

        xi_b = R(q)^T * grad_{p_i} V1_i

    where grad_{p_i} V1_i is assembled from the incident edge gradients and
    the incidence signs provided by the caller.
    """

    def __init__(self, robot_id: int, config: BlueROV2Config, rov_model):
        super().__init__(robot_id, config)
        self.cfg: BlueROV2Config = config
        self.rov = rov_model

        # Command-filter state for inertial translational reference
        self.v_f_I = np.zeros(3)
        self.dv_f_I = np.zeros(3)

    # ============================================================
    # Basic utilities
    # ============================================================

    def reset_filter_state(self, value: Optional[Array] = None) -> None:
        if value is None:
            self.v_f_I[:] = 0.0
            self.dv_f_I[:] = 0.0
            return

        value = np.asarray(value, dtype=float).reshape(-1)
        if value.size != 3:
            raise ValueError("Filter reset value must have dimension 3.")
        self.v_f_I[:] = value
        self.dv_f_I[:] = 0.0

    def _sat_vector(self, x: Array, max_norm: Optional[float]) -> Array:
        """Apply Euclidean-norm smooth saturation to a velocity vector."""
        x = np.asarray(x, dtype=float).reshape(-1)
        if max_norm is None:
            return x

        x = smooth_saturate(x, max_norm=max_norm)
        return x

    def saturate_velocity(self, v: Array) -> Array:
        return self._sat_vector(v, self.cfg.v_max)

    def saturate_translational_force(self, tau_v: Array) -> Array:
        return self._sat_vector(tau_v, self.cfg.tau_v_max)

    # ============================================================
    # Outer-loop virtual translational velocity
    # ============================================================

    def compute_edge_gradients_and_virtual_velocity(
        self,
        relative_states: Array,
        desired_relative_states: Array,
    ) -> Tuple[Array, Array, Array]:
        """
        Returns
        -------
        summed_grad : np.ndarray, shape (3,)
            Sum of edge gradients.
        edge_grads : np.ndarray, shape (n_edges, 3)
            Individual edge gradients.
        v_des_I : np.ndarray, shape (3,)
            Virtual inertial translational velocity.
        """
        relative_states = np.asarray(relative_states, dtype=float)
        desired_relative_states = np.asarray(desired_relative_states, dtype=float)

        if relative_states.shape != desired_relative_states.shape:
            raise ValueError(
                "relative_states and desired_relative_states must have the same shape."
            )
        if relative_states.ndim != 2 or relative_states.shape[1] != 3:
            raise ValueError(
                "For BlueROV2, relative_states must have shape (n_edges, 3)."
            )

        edge_grads = self.edge_gradients(relative_states, desired_relative_states)
        summed_grad = np.sum(edge_grads, axis=0)
        v_des_I = -self.cfg.c_outer * summed_grad
        v_des_I = self.saturate_velocity(v_des_I)
        return summed_grad, edge_grads, v_des_I

    def compute_position_gradient_from_incidence(
        self,
        edge_gradients: Array,
        incidence_signs: Array,
    ) -> Array:
        """
        Compute grad_{p_i} V1_i = sum_k [E]_{ik} grad Phi_k(d_k)
        """
        edge_gradients = np.asarray(edge_gradients, dtype=float)
        incidence_signs = np.asarray(incidence_signs, dtype=float).reshape(-1)

        if edge_gradients.ndim != 2 or edge_gradients.shape[1] != 3:
            raise ValueError("edge_gradients must have shape (n_edges, 3).")
        if incidence_signs.size != edge_gradients.shape[0]:
            raise ValueError("incidence_signs must have one value per incident edge.")

        return np.sum(edge_gradients * incidence_signs[:, None], axis=0)

    def compute_passivating_term_body(
        self,
        q_i: Array,
        edge_gradients: Array,
        incidence_signs: Array,
    ) -> Tuple[Array, Array]:
        """
        Compute passivating term

            xi_I = grad_{p_i} V1_i
            xi_b = R(q)^T xi_I
        """
        q_i = np.asarray(q_i, dtype=float).reshape(4)
        grad_V1_pos_I = self.compute_position_gradient_from_incidence(
            edge_gradients=edge_gradients,
            incidence_signs=incidence_signs,
        )
        R_b2I = rotation_matrix_from_quaternion(q_i)
        xi_b = R_b2I.T @ grad_V1_pos_I
        return xi_b, grad_V1_pos_I

    # ============================================================
    # Command filter on inertial translational reference
    # ============================================================

    def update_virtual_velocity_filter(self, v_des_I: Array) -> Tuple[Array, Array]:
        v_des_I = np.asarray(v_des_I, dtype=float).reshape(3)

        if self.cfg.filter_type == "first_order":
            self.v_f_I, self.dv_f_I = self.first_order_filter_step(
                y_f=self.v_f_I,
                y_des=v_des_I,
                dt=self.cfg.dt,
                lam=self.cfg.lam,
            )
            return self.v_f_I.copy(), self.dv_f_I.copy()

        self.v_f_I, self.dv_f_I, _ = self.second_order_filter_step(
            y_f=self.v_f_I,
            dy_f=self.dv_f_I,
            y_des=v_des_I,
            dt=self.cfg.dt,
            wn=self.cfg.wn,
            zeta=self.cfg.zeta,
        )
        return self.v_f_I.copy(), self.dv_f_I.copy()

    def filtered_reference_in_body_frame(
        self,
        q_i: Array,
        omega_i: Array,
    ) -> Tuple[Array, Array]:
        """
        If v_b = R^T v_I, then

            dot(v_b) = R^T dot(v_I) - S(omega) v_b
        """
        q_i = np.asarray(q_i, dtype=float).reshape(4)
        omega_i = np.asarray(omega_i, dtype=float).reshape(3)

        R_b2I = rotation_matrix_from_quaternion(q_i)
        R_I2b = R_b2I.T

        v_f_b = R_I2b @ self.v_f_I
        dv_f_b = R_I2b @ self.dv_f_I - skew(omega_i) @ v_f_b
        return v_f_b, dv_f_b

    # ============================================================
    # Dynamics interface
    # ============================================================

    def _mass_matrix(self, chi_i: Array, nu_i: Array) -> Array:
        if hasattr(self.rov, "mass_matrix"):
            return np.asarray(self.rov.mass_matrix(chi_i, nu_i), dtype=float)
        if hasattr(self.rov, "M"):
            return np.asarray(self.rov.M, dtype=float)
        raise AttributeError(
            "rov_model must provide 'mass_matrix(chi, nu)' or attribute 'M'."
        )

    def _coriolis_matrix(self, chi_i: Array, nu_i: Array) -> Array:
        if hasattr(self.rov, "coriolis"):
            return np.asarray(self.rov.coriolis(chi_i, nu_i), dtype=float)
        if hasattr(self.rov, "_coriolis"):
            return np.asarray(self.rov._coriolis(nu_i), dtype=float)
        raise AttributeError(
            "rov_model must provide 'coriolis(chi, nu)' or '_coriolis(nu)'."
        )

    def _damping_matrix(self, chi_i: Array, nu_i: Array) -> Array:
        if hasattr(self.rov, "damping"):
            return np.asarray(self.rov.damping(chi_i, nu_i), dtype=float)
        if hasattr(self.rov, "_damping"):
            return np.asarray(self.rov._damping(nu_i), dtype=float)
        raise AttributeError(
            "rov_model must provide 'damping(chi, nu)' or '_damping(nu)'."
        )

    def _restoring_vector(self, chi_i: Array, nu_i: Array) -> Array:
        if hasattr(self.rov, "restoring"):
            return np.asarray(self.rov.restoring(chi_i, nu_i), dtype=float).reshape(6)
        raise AttributeError(
            "rov_model must provide 'restoring(chi, nu)' for quaternion-based usage."
        )

    def full_dynamics_matrices(
        self,
        p_i: Array,
        q_i: Array,
        nu_i: Array,
    ) -> Tuple[Array, Array, Array, Array]:
        p_i = np.asarray(p_i, dtype=float).reshape(3)
        q_i = np.asarray(q_i, dtype=float).reshape(4)
        nu_i = np.asarray(nu_i, dtype=float).reshape(6)
        chi_i = np.hstack((p_i, q_i))

        M = self._mass_matrix(chi_i, nu_i)
        C = self._coriolis_matrix(chi_i, nu_i)
        D = self._damping_matrix(chi_i, nu_i)
        g = self._restoring_vector(chi_i, nu_i)
        return M, C, D, g

    def translational_partition(
        self,
        p_i: Array,
        q_i: Array,
        nu_i: Array,
    ) -> Tuple[Array, Array]:
        """
        Returns an implementation-oriented reduced partition

            M_v dot(v) + h_v = tau_v + d_v

        Prefer rov_model.translational_partition(chi, nu) when available.
        """
        p_i = np.asarray(p_i, dtype=float).reshape(3)
        q_i = np.asarray(q_i, dtype=float).reshape(4)
        nu_i = np.asarray(nu_i, dtype=float).reshape(6)
        chi_i = np.hstack((p_i, q_i))

        if hasattr(self.rov, "translational_partition"):
            M_v, h_v = self.rov.translational_partition(chi_i, nu_i)
            return np.asarray(M_v, dtype=float), np.asarray(h_v, dtype=float).reshape(3)

        M, C, D, g = self.full_dynamics_matrices(p_i, q_i, nu_i)
        h_full = C @ nu_i + D @ nu_i + g
        return np.asarray(M[:3, :3], dtype=float), np.asarray(
            h_full[:3], dtype=float
        ).reshape(3)

    def actuator_matrix(self) -> Array:
        if hasattr(self.rov, "actuator_matrix"):
            return np.asarray(self.rov.actuator_matrix(), dtype=float)
        if hasattr(self.rov, "B"):
            return np.asarray(self.rov.B, dtype=float)
        raise AttributeError(
            "rov_model must provide 'actuator_matrix()' or attribute 'B'."
        )

    # ============================================================
    # Feedback-linearization realization
    # ============================================================

    def feedback_linearization_translational_force(
        self,
        p_i: Array,
        q_i: Array,
        nu_i: Array,
        relative_states: Array,
        desired_relative_states: Array,
        incidence_signs: Array,
    ) -> Tuple[Array, dict]:
        """
        Command-filtered backstepping with feedback linearization and
        internally computed passivating term.
        """
        p_i = np.asarray(p_i, dtype=float).reshape(3)
        q_i = np.asarray(q_i, dtype=float).reshape(4)
        nu_i = np.asarray(nu_i, dtype=float).reshape(6)

        v_body = nu_i[:3]
        omega_body = nu_i[3:]

        summed_grad, edge_grads, v_des_I = (
            self.compute_edge_gradients_and_virtual_velocity(
                relative_states=relative_states,
                desired_relative_states=desired_relative_states,
            )
        )

        v_f_I, dv_f_I = self.update_virtual_velocity_filter(v_des_I)

        R_b2I = rotation_matrix_from_quaternion(q_i)
        e_v_I = R_b2I @ v_body - v_f_I

        v_f_b, dv_f_b = self.filtered_reference_in_body_frame(q_i, omega_body)
        e_v_b = v_body - v_f_b

        xi_b, grad_V1_pos_I = self.compute_passivating_term_body(
            q_i=q_i,
            edge_gradients=edge_grads,
            incidence_signs=incidence_signs,
        )

        M_v, h_v = self.translational_partition(p_i, q_i, nu_i)

        tau_v = M_v @ dv_f_b + h_v - self.cfg.k_v * e_v_b - xi_b
        tau_v = self.saturate_translational_force(tau_v)

        debug = {
            "summed_grad": summed_grad,
            "edge_gradients": edge_grads,
            "grad_V1_pos_I": grad_V1_pos_I,
            "xi_b": xi_b,
            "v_des_I": v_des_I,
            "v_f_I": v_f_I,
            "dv_f_I": dv_f_I,
            "v_f_b": v_f_b,
            "dv_f_b": dv_f_b,
            "e_v_I": e_v_I,
            "e_v_b": e_v_b,
            "M_v": M_v,
            "h_v": h_v,
        }
        return tau_v, debug

    # ============================================================
    # CLF-QP realization on translational generalized force
    # ============================================================

    def translational_tracking_clf_terms(
        self,
        p_i: Array,
        q_i: Array,
        nu_i: Array,
    ) -> Tuple[float, Array, Array]:
        """
        Kinetic inner-loop CLF:

            V = 0.5 * e_v_b^T M_v e_v_b
        """
        p_i = np.asarray(p_i, dtype=float).reshape(3)
        q_i = np.asarray(q_i, dtype=float).reshape(4)
        nu_i = np.asarray(nu_i, dtype=float).reshape(6)

        v_body = nu_i[:3]
        omega_body = nu_i[3:]

        M_v, _ = self.translational_partition(p_i, q_i, nu_i)
        v_f_b, _ = self.filtered_reference_in_body_frame(q_i, omega_body)

        e_v_b = v_body - v_f_b
        V = 0.5 * float(e_v_b @ (M_v @ e_v_b))
        grad_V = M_v @ e_v_b
        return V, grad_V, e_v_b

    def clf_qp_translational_force(
        self,
        p_i: Array,
        q_i: Array,
        nu_i: Array,
        relative_states: Array,
        desired_relative_states: Array,
        local_fallback: Optional[Array] = None,
        include_position_gradient_term: bool = False,
        incidence_signs: Optional[Array] = None,
    ) -> Tuple[Array, dict]:
        """
        CLF-QP over translational generalized force.

        By default uses the inner-loop kinetic CLF only.
        Optionally adds the position-gradient contribution from V1_i
        in a simple implementation-oriented way.
        """
        p_i = np.asarray(p_i, dtype=float).reshape(3)
        q_i = np.asarray(q_i, dtype=float).reshape(4)
        nu_i = np.asarray(nu_i, dtype=float).reshape(6)

        summed_grad, edge_grads, v_des_I = (
            self.compute_edge_gradients_and_virtual_velocity(
                relative_states=relative_states,
                desired_relative_states=desired_relative_states,
            )
        )
        self.update_virtual_velocity_filter(v_des_I)

        M_v, h_v = self.translational_partition(p_i, q_i, nu_i)
        V, grad_V, e_v_b = self.translational_tracking_clf_terms(p_i, q_i, nu_i)

        f_sys = -np.linalg.solve(M_v, h_v)
        LfV = float(grad_V @ f_sys)

        LgV = np.linalg.solve(M_v.T, grad_V).reshape(1, 3)

        extra_info = {}

        if include_position_gradient_term:
            if incidence_signs is None:
                raise ValueError(
                    "incidence_signs must be provided when include_position_gradient_term=True."
                )
            _, grad_V1_pos_I = self.compute_passivating_term_body(
                q_i=q_i,
                edge_gradients=edge_grads,
                incidence_signs=incidence_signs,
            )
            extra_info["grad_V1_pos_I"] = grad_V1_pos_I

        if np.linalg.norm(LgV) < self.cfg.lgv_epsilon:
            tau_v = (
                np.zeros(3)
                if local_fallback is None
                else np.asarray(local_fallback, dtype=float).reshape(3)
            )
            tau_v = self.saturate_translational_force(tau_v)
            return tau_v, {
                "mode": "fallback",
                "reason": "small_LgV",
                "V": V,
                "LfV": LfV,
                "LgV": LgV,
                "v_des_I": v_des_I,
                "edge_gradients": edge_grads,
                "e_v_b": e_v_b,
                **extra_info,
            }

        P = np.diag(
            [
                self.cfg.clf_force_weight,
                self.cfg.clf_force_weight,
                self.cfg.clf_force_weight,
                self.cfg.clf_slack_weight,
            ]
        )
        q_obj = np.zeros(4)

        A = np.hstack([LgV, np.array([[-1.0]])]).reshape(1, 4)
        b = np.array([-self.cfg.clf_alpha * V - LfV])

        if self.cfg.tau_v_max is None:
            lb_tau = np.full(3, -np.inf)
            ub_tau = np.full(3, np.inf)
        else:
            lb_tau = np.full(3, -self.cfg.tau_v_max)
            ub_tau = np.full(3, self.cfg.tau_v_max)

        lb = np.hstack([lb_tau, 0.0])
        ub = np.hstack([ub_tau, np.inf])

        sol = solve_qp(
            P=sparse.csc_matrix(P),
            q=q_obj,
            G=sparse.csc_matrix(A),
            h=b,
            A=None,
            b=None,
            lb=lb,
            ub=ub,
            solver="osqp",
        )

        if sol is None:
            tau_v = (
                np.zeros(3)
                if local_fallback is None
                else np.asarray(local_fallback, dtype=float).reshape(3)
            )
            tau_v = self.saturate_translational_force(tau_v)
            return tau_v, {
                "mode": "fallback",
                "reason": "qp_failed",
                "V": V,
                "LfV": LfV,
                "LgV": LgV,
                "v_des_I": v_des_I,
                "edge_gradients": edge_grads,
                "e_v_b": e_v_b,
                **extra_info,
            }

        tau_v = sol[:3]
        return tau_v, {
            "mode": "qp",
            "delta": sol[3],
            "V": V,
            "LfV": LfV,
            "LgV": LgV,
            "v_des_I": v_des_I,
            "edge_gradients": edge_grads,
            "e_v_b": e_v_b,
            **extra_info,
        }

    # ============================================================
    # CLF-QP with direct thruster allocation
    # ============================================================

    def clf_qp_thrusters(
        self,
        p_i: Array,
        q_i: Array,
        nu_i: Array,
        relative_states: Array,
        desired_relative_states: Array,
        f_min: Array,
        f_max: Array,
    ) -> Tuple[Array, dict]:
        p_i = np.asarray(p_i, dtype=float).reshape(3)
        q_i = np.asarray(q_i, dtype=float).reshape(4)
        nu_i = np.asarray(nu_i, dtype=float).reshape(6)
        f_min = np.asarray(f_min, dtype=float).reshape(-1)
        f_max = np.asarray(f_max, dtype=float).reshape(-1)

        summed_grad, edge_grads, v_des_I = (
            self.compute_edge_gradients_and_virtual_velocity(
                relative_states=relative_states,
                desired_relative_states=desired_relative_states,
            )
        )
        self.update_virtual_velocity_filter(v_des_I)

        M_v, h_v = self.translational_partition(p_i, q_i, nu_i)
        V, grad_V, e_v_b = self.translational_tracking_clf_terms(p_i, q_i, nu_i)

        f_sys = -np.linalg.solve(M_v, h_v)
        LfV = float(grad_V @ f_sys)

        LgV_tau = np.linalg.solve(M_v.T, grad_V).reshape(1, 3)

        B = self.actuator_matrix()
        B_v = B[:3, :]
        LgV_f = LgV_tau @ B_v

        if np.linalg.norm(LgV_f) < self.cfg.lgv_epsilon:
            return np.zeros(B.shape[1]), {
                "mode": "fallback",
                "reason": "small_LgV",
                "V": V,
                "LfV": LfV,
                "LgV_f": LgV_f,
                "v_des_I": v_des_I,
                "summed_grad": summed_grad,
                "edge_gradients": edge_grads,
                "e_v_b": e_v_b,
            }

        m = B.shape[1]
        P = np.diag([self.cfg.thrust_weight] * m + [self.cfg.clf_slack_weight])
        q_obj = np.zeros(m + 1)

        A = np.hstack([LgV_f, np.array([[-1.0]])]).reshape(1, m + 1)
        b = np.array([-self.cfg.clf_alpha * V - LfV])

        lb = np.hstack([f_min, 0.0])
        ub = np.hstack([f_max, np.inf])

        sol = solve_qp(
            P=sparse.csc_matrix(P),
            q=q_obj,
            G=sparse.csc_matrix(A),
            h=b,
            A=None,
            b=None,
            lb=lb,
            ub=ub,
            solver="osqp",
        )

        if sol is None:
            return np.zeros(m), {
                "mode": "fallback",
                "reason": "qp_failed",
                "V": V,
                "LfV": LfV,
                "LgV_f": LgV_f,
                "v_des_I": v_des_I,
                "summed_grad": summed_grad,
                "edge_gradients": edge_grads,
                "e_v_b": e_v_b,
            }

        f_star = sol[:m]
        return f_star, {
            "mode": "qp",
            "delta": sol[m],
            "V": V,
            "LfV": LfV,
            "LgV_f": LgV_f,
            "v_des_I": v_des_I,
            "summed_grad": summed_grad,
            "edge_gradients": edge_grads,
            "e_v_b": e_v_b,
        }

    # ============================================================
    # Allocation helper
    # ============================================================

    def allocate_thrusters_min_norm(self, tau: Array) -> Array:
        tau = np.asarray(tau, dtype=float).reshape(6)
        B = self.actuator_matrix()
        return np.linalg.pinv(B) @ tau

    # ============================================================
    # Main interface
    # ============================================================

    def do_control(
        self,
        p_i: Array,
        q_i: Array,
        nu_i: Array,
        relative_states: Array,
        desired_relative_states: Array,
        incidence_signs: Array,
        mode: Literal[
            "feedback_linearization", "clf_qp_tau", "clf_qp_thrusters"
        ] = "feedback_linearization",
        tau_omega: Optional[Array] = None,
        local_fallback_tau_v: Optional[Array] = None,
        f_min: Optional[Array] = None,
        f_max: Optional[Array] = None,
    ) -> Tuple[Array, dict]:
        """
        Parameters
        ----------
        incidence_signs : np.ndarray, shape (n_edges,)
            Values [E]_{ik} for the edges involving agent i.
            Needed for the passivating term.
        """
        tau_omega = (
            np.zeros(3)
            if tau_omega is None
            else np.asarray(tau_omega, dtype=float).reshape(3)
        )

        if mode == "feedback_linearization":
            tau_v, debug = self.feedback_linearization_translational_force(
                p_i=p_i,
                q_i=q_i,
                nu_i=nu_i,
                relative_states=relative_states,
                desired_relative_states=desired_relative_states,
                incidence_signs=incidence_signs,
            )
            tau = np.hstack([tau_v, tau_omega])
            debug["mode"] = mode
            debug["tau"] = tau
            return tau, debug

        if mode == "clf_qp_tau":
            tau_v, debug = self.clf_qp_translational_force(
                p_i=p_i,
                q_i=q_i,
                nu_i=nu_i,
                relative_states=relative_states,
                desired_relative_states=desired_relative_states,
                local_fallback=local_fallback_tau_v,
                include_position_gradient_term=False,
                incidence_signs=incidence_signs,
            )
            tau = np.hstack([tau_v, tau_omega])
            debug["mode"] = mode
            debug["tau"] = tau
            return tau, debug

        if mode == "clf_qp_thrusters":
            if f_min is None or f_max is None:
                raise ValueError(
                    "f_min and f_max must be provided in 'clf_qp_thrusters' mode."
                )
            f_star, debug = self.clf_qp_thrusters(
                p_i=p_i,
                q_i=q_i,
                nu_i=nu_i,
                relative_states=relative_states,
                desired_relative_states=desired_relative_states,
                f_min=f_min,
                f_max=f_max,
            )
            debug["mode"] = mode
            debug["f"] = f_star
            return f_star, debug

        raise ValueError(f"Unknown mode '{mode}'.")
