#!/usr/bin/env python3

##########################################################
#                                                        #
# Parameters to the Formation Controller                 #
# Code Author: Joao Zenario                              #
# Refactored architecture: config + scenario + logs      #
#                                                        #
##########################################################

from dataclasses import dataclass, field, replace
import argparse
import numpy as np

from graphs_utils import (
    build_directed_tree_adjacency,
    compute_directed_edge_list,
    generate_random_connected_graph,
    is_rooted_directed_tree,
)


@dataclass
class SimulationConfig:
    """User-selected simulation settings.

    Parameters are grouped by topic. CLI equivalents are exposed in
    Parameters.function_arguments(); programmatic overrides are passed as
    keyword arguments to Parameters().
    """

    # ------------------------------------------------------------------
    # Swarm topology
    # ------------------------------------------------------------------
    N: int = 5
    """Number of agents (including the leader at index 0)."""
    dim: int = 3
    """Spatial dimension: 2 or 3."""
    mode_graph: int = 1
    """Graph type: 0 = undirected, 1 = directed rooted spanning tree."""
    mode_dynamics: int = 1
    """Legacy dynamics selector passed to graph utilities (0 = SI, 1 = DI)."""
    tree_type: str = "leader_star"
    """Directed tree topology: 'leader_star', 'leader_chain', or 'random_tree'."""

    # ------------------------------------------------------------------
    # Active constraint flags
    # ------------------------------------------------------------------
    conn: bool = True
    """Enable connectivity-maintenance CBF (upper distance bound)."""
    coll: bool = True
    """Enable collision-avoidance CBF (lower distance bound)."""
    fov: bool = True
    """Enable field-of-view CBF (pyramidal camera frustum)."""
    rec: bool = True
    """Use recentered (smooth) barrier functions."""

    # ------------------------------------------------------------------
    # Time stepping
    # ------------------------------------------------------------------
    dt: float = 0.01
    """Simulation time step [s]."""
    steps: int = 3000
    """Total number of simulation steps."""

    # ------------------------------------------------------------------
    # Controller / simulation mode
    # ------------------------------------------------------------------
    simulation_mode: str = "bluerov_feedback_linearization"
    """Controller backend:
      'single_integrator'              – first-order SI
      'double_integrator'              – second-order DI
      'bluerov_feedback_linearization' – feedback-linearised BlueROV2
      'bluerov_clf_qp_tau'             – CLF-QP wrench (force + torque) output
      'bluerov_clf_qp_thrusters'       – CLF-QP direct thruster allocation
    """
    use_quaternions: bool = True
    """Represent orientation as unit quaternions (True) or Euler angles (False)."""

    # ------------------------------------------------------------------
    # Formation control gains
    # c1: outer-loop virtual velocity gain
    # c2: inner-loop tracking gain
    # c3: velocity error damping gain
    # c4_h/c4_v: FoV barrier weight (horizontal / vertical channel)
    # k_form: scalar multiplier applied to the formation gradient
    # ------------------------------------------------------------------
    c1: float = 1.0
    """Outer-loop (virtual velocity) gain."""
    c2: float = 1.0
    """Inner-loop (velocity tracking) gain."""
    c3: float = 1.0
    """Velocity error damping gain."""
    c4_h: float = 0.5
    """FoV barrier weight for the horizontal camera channel."""
    c4_v: float = 0.5
    """FoV barrier weight for the vertical camera channel."""
    k_form: float = 1.0
    """Global scalar multiplier on the formation gradient."""

    # ------------------------------------------------------------------
    # Command filter (inner-loop reference model)
    # ------------------------------------------------------------------
    filter_type: str = "second_order"
    """Filter type: 'first_order' (bandwidth lam) or 'second_order' (wn, zeta)."""
    wn: float = 10.0
    """Natural frequency of the second-order command filter [rad/s]."""
    zeta: float = 1.0
    """Damping ratio of the second-order command filter (1 = critically damped)."""
    lam: float = 5.0
    """Bandwidth of the first-order command filter [rad/s]."""

    # ------------------------------------------------------------------
    # Velocity and force limits
    # ------------------------------------------------------------------
    v_max: float = 1.0
    """Maximum admissible translational speed [m/s]."""
    a_max: float = 1.0
    """Maximum admissible translational acceleration [m/s²]."""
    tau_v_max: float | None = 20.0
    """Maximum translational generalised force norm [N]; None = unconstrained."""
    r_max: float = 0.5
    """Maximum admissible angular rate [rad/s] (used by SI/DI modes)."""

    # ------------------------------------------------------------------
    # Inter-agent distance constraints
    # ------------------------------------------------------------------
    d_min: float = 0.5
    """Collision-avoidance lower bound on inter-agent distance [m]."""
    d_max: float = 5.0
    """Connectivity upper bound on inter-agent distance [m]."""

    # ------------------------------------------------------------------
    # Attitude stabilisation gains (used in the BlueROV2 when no FoV control is active)
    # ------------------------------------------------------------------
    k_att: float = 5.0
    """Proportional gain for attitude stabilisation."""
    k_att_d: float = 2.0
    """Derivative (damping) gain for attitude stabilisation."""
    kp_yaw: float = 2.0
    """Proportional gain for yaw error in the FoV attitude controller."""
    kp_pitch: float = 1.0
    """Proportional gain for pitch error in the FoV attitude controller."""

    # ------------------------------------------------------------------
    # Field-of-view (FoV) controller
    # ------------------------------------------------------------------
    theta_max_x: float = float(np.radians(45))
    """Half-angle of the horizontal FoV cone [rad]."""
    theta_max_y: float = float(np.radians(30))
    """Half-angle of the vertical FoV cone [rad]."""
    alpha_h_des: float = 0.0
    """Desired normalised horizontal camera coordinate (0 = centred); |α| < 1."""
    alpha_v_des: float = 0.0
    """Desired normalised vertical camera coordinate (0 = centred); |α| < 1."""
    fov_eps: float = 1e-6
    """Numerical regularisation added inside the FoV barrier log terms."""
    k_fov_omega: float = 0.10
    """Backstepping gain: maps FoV attitude gradient to desired angular rate."""
    k_fov_backstep: float = 0.10
    """Backstepping gain: penalises angular-rate tracking error."""
    omega_fov_max: float = 0.10
    """Component-wise saturation applied to the desired angular rate [rad/s]."""
    fov_R_cb: np.ndarray | None = None
    """Rotation matrix from camera frame to body frame (3×3). None → eye(3)."""
    fov_p_cb: np.ndarray | None = None
    """Translation from camera origin to body origin in body frame [m]. None → zeros(3)."""

    # ------------------------------------------------------------------
    # CLF-QP (bluerov_clf_qp_tau mode)
    # ------------------------------------------------------------------
    clf_alpha: float = 0.1
    """Class-K decay rate α for the CLF constraint: V̇ ≤ -α·V + δ."""
    clf_alpha_sat: float = 0.5
    """Saturation ceiling for the class-K function; set None for pure linear."""
    clf_slack_weight: float = 1000.0
    """Penalty weight on the CLF slack variable δ (large → hard CLF constraint)."""
    clf_force_weight: float = 1.0
    """Regularisation weight on translational forces in the CLF-QP cost."""
    clf_torque_weight: float = 10.0
    """Regularisation weight on torques in the CLF-QP cost.
    High value → QP drives τ_ω ≈ 0 when FoV is inactive."""
    lgv_epsilon: float = 1e-6
    """Minimum LgV norm below which the CLF constraint is skipped (numerical safety)."""
    saturation_mode: str = "smooth"
    """Force saturation strategy: 'hard', 'smooth', or 'hybrid'."""
    saturation_switch_ratio: float = 4.0
    """Ratio τ_max/τ_nom at which the hybrid saturation mode switches [> 1]."""

    # ------------------------------------------------------------------
    # Thruster allocation (bluerov_clf_qp_thrusters mode)
    # ------------------------------------------------------------------
    thrust_weight: float = 1.0
    """Regularisation weight on individual thruster forces in the allocation QP."""
    thruster_limit: float = 40.0
    """Per-thruster force saturation limit [N]."""

    # ------------------------------------------------------------------
    # Adaptive barrier-domain relaxation
    # ------------------------------------------------------------------
    use_adaptive_barrier: bool = False
    """Enable time-varying relaxation of the CBF safety margins."""
    lambda_delta: float = 2.0
    """Decay rate of the lower-margin relaxation σ_δ."""
    gamma_delta: float = 1.0
    """Activation gain for the lower-margin relaxation dynamics."""
    sigma_delta_max: float = 0.15
    """Maximum allowed lower-margin relaxation [m]. Must satisfy
    sigma_delta_max + sigma_Delta_max < d_max - d_min."""
    lambda_Delta: float = 2.0
    """Decay rate of the upper-margin relaxation σ_Δ."""
    gamma_Delta: float = 1.0
    """Activation gain for the upper-margin relaxation dynamics."""
    sigma_Delta_max: float = 0.15
    """Maximum allowed upper-margin relaxation [m]."""
    adaptive_dt: float = 0.01
    """Integration step for the adaptive barrier ODE [s]."""

    # ------------------------------------------------------------------
    # Leader motion profile
    # ------------------------------------------------------------------
    leader_vel_amp: float = 0.5
    """Amplitude of the sinusoidal leader velocity profile [m/s]."""
    leader_freq: float = 0.5
    """Frequency of the sinusoidal leader velocity profile [Hz]."""
    leader_accel_steps: int = 200
    """Number of steps over which the leader accelerates at startup."""
    leader_brake_steps: int = 200
    """Number of steps over which the leader decelerates back to rest."""

    # ------------------------------------------------------------------
    # Scenario generation
    # ------------------------------------------------------------------
    R: float = 2.0
    """Desired formation radius (distance from leader to each follower offset) [m].
    Must satisfy d_min < R < d_max."""
    l: float = 7.0
    """Side length of the cubic spawn workspace [m]."""
    random_seed: int | None = 0
    """RNG seed for scenario generation; None = non-deterministic."""
    init_noise: float = 0.25
    """Standard deviation of position noise applied to initial conditions [m]."""
    max_scenario_tries: int = 500
    """Maximum rejection-sampling attempts when building a feasible scenario."""

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    counter: int = 0
    """Step counter incremented by the simulation loop (do not set manually)."""

    def validate(self) -> None:
        if self.dim not in (2, 3):
            raise ValueError("dim must be 2 or 3.")
        if self.N < 1:
            raise ValueError("N must be positive.")
        if self.d_max <= self.d_min:
            raise ValueError("d_max must be strictly greater than d_min.")
        if self.dt <= 0.0:
            raise ValueError("dt must be positive.")
        if self.steps < 1:
            raise ValueError("steps must be positive.")
        if self.v_max <= 0.0:
            raise ValueError("v_max must be positive.")
        if self.a_max <= 0.0:
            raise ValueError("a_max must be positive.")
        if self.wn <= 0.0:
            raise ValueError("wn must be positive.")
        if self.lam <= 0.0:
            raise ValueError("lam must be positive.")
        if self.zeta <= 0.0:
            raise ValueError("zeta must be positive.")
        if self.tau_v_max is not None and self.tau_v_max <= 0.0:
            raise ValueError("tau_v_max must be positive when provided.")
        if self.clf_alpha <= 0.0:
            raise ValueError("clf_alpha must be positive.")
        if self.clf_alpha_sat is not None and self.clf_alpha_sat <= 0.0:
            raise ValueError("clf_alpha_sat must be positive when provided.")
        if self.clf_slack_weight <= 0.0:
            raise ValueError("clf_slack_weight must be positive.")
        if self.clf_force_weight <= 0.0:
            raise ValueError("clf_force_weight must be positive.")
        if self.clf_torque_weight <= 0.0:
            raise ValueError("clf_torque_weight must be positive.")
        if self.thrust_weight <= 0.0:
            raise ValueError("thrust_weight must be positive.")
        if self.lgv_epsilon <= 0.0:
            raise ValueError("lgv_epsilon must be positive.")
        if self.saturation_mode not in {"hard", "smooth", "hybrid"}:
            raise ValueError("saturation_mode must be 'hard', 'smooth', or 'hybrid'.")
        if self.saturation_switch_ratio <= 1.0:
            raise ValueError("saturation_switch_ratio must be > 1.0.")
        if self.thruster_limit <= 0.0:
            raise ValueError("thruster_limit must be positive.")
        if self.k_form < 0.0:
            raise ValueError("k_form must be nonnegative.")
        if self.k_att < 0.0:
            raise ValueError("k_att must be nonnegative.")
        if self.k_att_d < 0.0:
            raise ValueError("k_att_d must be nonnegative.")
        if self.lambda_delta <= 0.0:
            raise ValueError("lambda_delta must be positive.")
        if self.gamma_delta < 0.0:
            raise ValueError("gamma_delta must be nonnegative.")
        if self.sigma_delta_max < 0.0:
            raise ValueError("sigma_delta_max must be nonnegative.")
        if self.lambda_Delta <= 0.0:
            raise ValueError("lambda_Delta must be positive.")
        if self.gamma_Delta < 0.0:
            raise ValueError("gamma_Delta must be nonnegative.")
        if self.sigma_Delta_max < 0.0:
            raise ValueError("sigma_Delta_max must be nonnegative.")
        if self.adaptive_dt <= 0.0:
            raise ValueError("adaptive_dt must be positive.")
        if self.use_adaptive_barrier:
            if self.sigma_delta_max + self.sigma_Delta_max >= self.d_max - self.d_min:
                raise ValueError(
                    "sigma_delta_max + sigma_Delta_max must be strictly smaller than d_max - d_min."
                )
        if self.leader_accel_steps < 0:
            raise ValueError("leader_accel_steps must be nonnegative.")
        if self.leader_brake_steps < 0:
            raise ValueError("leader_brake_steps must be nonnegative.")
        if abs(self.alpha_h_des) >= 1.0:
            raise ValueError("alpha_h_des must satisfy |alpha_h_des| < 1.")
        if abs(self.alpha_v_des) >= 1.0:
            raise ValueError("alpha_v_des must satisfy |alpha_v_des| < 1.")
        if self.fov_eps <= 0.0:
            raise ValueError("fov_eps must be positive.")
        if self.k_fov_omega <= 0.0:
            raise ValueError("k_fov_omega must be positive.")
        if self.k_fov_backstep <= 0.0:
            raise ValueError("k_fov_backstep must be positive.")
        if self.omega_fov_max <= 0.0:
            raise ValueError("omega_fov_max must be positive.")


@dataclass
class Scenario:
    """Initial/structural simulation data."""

    x: np.ndarray
    A: np.ndarray
    q: np.ndarray
    o: np.ndarray
    relative_pos_desired: np.ndarray
    target_dist: np.ndarray
    E: np.ndarray
    NE: int
    dist: np.ndarray
    x_lf: np.ndarray
    dot_x_lf: np.ndarray

    @staticmethod
    def _make_offsets(cfg) -> np.ndarray:
        """
        Desired leader-centered formation offsets.
        """
        o = np.zeros((cfg.N, cfg.dim), dtype=float)

        if cfg.N <= 1:
            return o

        for i in range(1, cfg.N):
            angle = (2.0 * np.pi / max(cfg.N - 1, 1)) * (i - 1)
            if cfg.dim >= 2:
                o[i, 0] = cfg.R * np.cos(angle)
                o[i, 1] = cfg.R * np.sin(angle)
            if cfg.dim == 3:
                o[i, 2] = 0.0

        return o

    @staticmethod
    def _pairwise_distances(x: np.ndarray) -> np.ndarray:
        return np.linalg.norm(x[:, None, :] - x[None, :, :], axis=2)

    @staticmethod
    def _all_pairwise_collision_constraints_satisfied(
        dist: np.ndarray,
        d_min: float,
    ) -> bool:
        N = dist.shape[0]
        for i in range(N):
            for j in range(i + 1, N):
                if dist[i, j] <= d_min:
                    return False
        return True

    @staticmethod
    def _all_pairwise_upper_bound_satisfied(dist: np.ndarray, d_max: float) -> bool:
        N = dist.shape[0]
        for i in range(N):
            for j in range(i + 1, N):
                if dist[i, j] >= d_max:
                    return False
        return True

    @staticmethod
    def _edge_connectivity_constraints_satisfied(
        x: np.ndarray,
        A: np.ndarray,
        d_min: float,
        d_max: float,
    ) -> bool:
        """
        For each active directed edge A[i,j]=1 require:
            d_min < ||x_i - x_j|| < d_max
        """
        N = A.shape[0]
        for i in range(N):
            for j in range(N):
                if A[i, j] == 1:
                    dij = np.linalg.norm(x[i] - x[j])
                    if not (d_min < dij < d_max):
                        return False
        return True

    @classmethod
    def _build_positions_for_directed_tree(
        cls, cfg, o: np.ndarray, A: np.ndarray
    ) -> np.ndarray:
        """
        Generate positions by sampling directed-tree edge lengths uniformly
        in [d_min, d_max], while enforcing:
        - pairwise collision avoidance for all robots
        - connectivity bounds on all active directed edges
        """
        rng = np.random.default_rng(getattr(cfg, "random_seed", None))

        max_tries = getattr(cfg, "max_scenario_tries", 500)
        safety_eps = 1e-3

        def sample_unit_vector(dim: int) -> np.ndarray:
            v = rng.normal(size=dim)
            n = np.linalg.norm(v)
            if n < 1e-12:
                v = np.zeros(dim)
                v[0] = 1.0
                return v
            return v / n

        def inside_workspace(x: np.ndarray) -> bool:
            return bool(np.all(x >= 0.0) and np.all(x <= cfg.l))

        for _ in range(max_tries):
            # Keep the leader away from borders so followers can spread around.
            margin = 1.1 * cfg.d_max
            low = max(0.0, margin)
            high = min(cfg.l, cfg.l - margin)

            if high <= low:
                leader = np.full(cfg.dim, 0.5 * cfg.l, dtype=float)
            else:
                leader = rng.uniform(low=low, high=high, size=cfg.dim)

            x = np.zeros((cfg.N, cfg.dim), dtype=float)
            x[0] = leader

            feasible = True
            for i in range(1, cfg.N):
                parents = np.where(A[i] == 1)[0]
                if len(parents) != 1:
                    feasible = False
                    break

                p = int(parents[0])
                r = rng.uniform(cfg.d_min + safety_eps, cfg.d_max - safety_eps)
                direction = sample_unit_vector(cfg.dim)
                x[i] = x[p] + r * direction

                if not inside_workspace(x[i]):
                    feasible = False
                    break

            if not feasible:
                continue

            dist = cls._pairwise_distances(x)

            if not cls._all_pairwise_collision_constraints_satisfied(dist, cfg.d_min):
                continue

            if not cls._all_pairwise_upper_bound_satisfied(dist, cfg.d_max):
                continue

            if not cls._edge_connectivity_constraints_satisfied(
                x, A, cfg.d_min, cfg.d_max
            ):
                continue

            return x

        # deterministic fallback close to desired offsets around workspace center
        leader = np.full(cfg.dim, 0.5 * cfg.l, dtype=float)
        return leader + o

    @classmethod
    def _build_directed_scenario(cls, cfg, o: np.ndarray):
        """
        Directed case:
        - build rooted directed spanning tree
        - generate feasible positions for that tree
        """
        tree_type = getattr(cfg, "tree_type", "leader_star")
        A = build_directed_tree_adjacency(
            N=cfg.N,
            tree_type=tree_type,
            random_seed=getattr(cfg, "random_seed", None),
        )

        if not is_rooted_directed_tree(A, root=0):
            raise RuntimeError(
                "Generated adjacency is not a valid rooted directed spanning tree."
            )

        x = cls._build_positions_for_directed_tree(cfg, o, A)
        dist = cls._pairwise_distances(x)

        if not cls._all_pairwise_collision_constraints_satisfied(dist, cfg.d_min):
            raise RuntimeError(
                "Generated directed scenario violates collision-avoidance constraints."
            )

        if not cls._all_pairwise_upper_bound_satisfied(dist, cfg.d_max):
            raise RuntimeError(
                "Generated directed scenario violates all-pairs upper distance bound."
            )

        if not cls._edge_connectivity_constraints_satisfied(x, A, cfg.d_min, cfg.d_max):
            raise RuntimeError(
                "Generated directed scenario violates connectivity constraints on active edges."
            )

        E, NE = compute_directed_edge_list(A)
        return x, A, E, NE, dist

    @classmethod
    def _build_undirected_scenario(cls, cfg, o: np.ndarray):
        """
        Undirected case:
        use the existing graph-generation utility provided by the student.
        """
        x, A, E, NE, dist = generate_random_connected_graph(
            N=cfg.N,
            d=cfg.dim,
            d_max=cfg.d_max,
            d_min=cfg.d_min,
            l=cfg.l,
            mode=bool(cfg.mode_graph),
            fov=bool(cfg.fov),
        )

        x = np.asarray(x, dtype=float)
        A = np.asarray(A, dtype=int)
        E = np.asarray(E, dtype=np.int64)
        dist = np.asarray(dist, dtype=float)

        return x, A, E, NE, dist

    @classmethod
    def build(cls, cfg) -> "Scenario":
        o = cls._make_offsets(cfg)

        # Quick feasibility check for the desired formation around the leader
        if cfg.N > 1 and not (cfg.d_min < cfg.R < cfg.d_max):
            raise ValueError(
                "The desired formation radius R should satisfy d_min < R < d_max."
            )

        if cfg.mode_graph == 1:
            x, A, E, NE, dist = cls._build_directed_scenario(cfg, o)
        elif cfg.mode_graph == 0:
            x, A, E, NE, dist = cls._build_undirected_scenario(cfg, o)
        else:
            raise ValueError("mode_graph must be 0 (undirected) or 1 (directed).")

        q = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (cfg.N, 1))

        relative_pos_desired = np.zeros((cfg.N, cfg.N, cfg.dim), dtype=float)
        target_dist = np.zeros(cfg.N, dtype=float)

        for i in range(cfg.N):
            for j in range(cfg.N):
                relative_pos_desired[i, j] = o[i] - o[j]
            if i:
                target_dist[i] = np.linalg.norm(relative_pos_desired[i, 0])

        x_lf = np.zeros((cfg.N, cfg.dim), dtype=float)
        dot_x_lf = np.zeros((cfg.N, cfg.dim), dtype=float)

        return cls(
            x=x,
            A=A,
            q=q,
            o=o,
            relative_pos_desired=relative_pos_desired,
            target_dist=target_dist,
            E=E,
            NE=NE,
            dist=dist,
            x_lf=x_lf,
            dot_x_lf=dot_x_lf,
        )


@dataclass
class SimulationData:
    """Data generated during simulation."""

    history: np.ndarray
    vel_history: np.ndarray
    accel_history: np.ndarray
    err_history: list
    fov_history: np.ndarray
    v_history_full: list
    tau_history: list
    q_history: list
    debug_history: list = field(default_factory=list)

    @classmethod
    def empty(cls, cfg: SimulationConfig) -> "SimulationData":
        return cls(
            history=np.empty((0, cfg.N, cfg.dim)),
            vel_history=np.empty((0, cfg.N)),
            accel_history=np.empty((0, cfg.N)),
            err_history=[],
            fov_history=np.empty((0, cfg.N, 2)),
            v_history_full=[],
            tau_history=[],
            q_history=[],
            debug_history=[],
        )


class Parameters:
    """
    Container for clean architecture components:
    - config: SimulationConfig
    - scenario: Scenario
    - data: SimulationData
    """

    config: SimulationConfig
    scenario: Scenario
    data: SimulationData

    def __init__(self, config: SimulationConfig | None = None, **overrides):
        if config is not None and not isinstance(config, SimulationConfig):
            raise TypeError("config must be a SimulationConfig instance or None.")

        field_names = set(SimulationConfig.__dataclass_fields__.keys())
        unknown = set(overrides) - field_names
        if unknown:
            unknown_list = ", ".join(sorted(unknown))
            raise TypeError(f"Unknown SimulationConfig fields: {unknown_list}")

        bool_keys = {
            "conn",
            "coll",
            "fov",
            "rec",
            "use_quaternions",
            "use_adaptive_barrier",
        }
        normalized = dict(overrides)
        for key in bool_keys:
            if key in normalized:
                normalized[key] = bool(normalized[key])

        base = config if config is not None else SimulationConfig()
        cfg = replace(base, **normalized) if normalized else replace(base)
        cfg.validate()

        self.config = cfg
        self.scenario = Scenario.build(cfg)
        self.data = SimulationData.empty(cfg)

    def get_leader_acceleration(self, step: int) -> np.ndarray:
        """
        Accelerate then brake so leader velocity returns close to zero.
        """
        cfg = self.config
        accel = np.zeros(cfg.dim)
        accel_end = cfg.leader_accel_steps
        brake_end = cfg.leader_accel_steps + cfg.leader_brake_steps

        if step < accel_end:
            accel[0] = 0.1
            accel[1] = 0.1
        elif step < brake_end:
            accel[0] = -0.1
            accel[1] = -0.1

        return accel

    @staticmethod
    def function_arguments():
        parser = argparse.ArgumentParser(
            description="Formation Control Simulation Parameters"
        )

        parser.add_argument(
            "--N", type=int, default=SimulationConfig.N, help="Number of agents"
        )
        parser.add_argument(
            "--dim",
            type=int,
            default=SimulationConfig.dim,
            help="Dimension (2D or 3D)",
        )
        parser.add_argument(
            "--graph",
            type=int,
            default=SimulationConfig.mode_graph,
            help="Graph type (0: Undirected, 1: Directed)",
        )
        parser.add_argument(
            "--conn",
            type=int,
            default=int(SimulationConfig.conn),
            help="Connectivity (0/1)",
        )
        parser.add_argument(
            "--coll",
            type=int,
            default=int(SimulationConfig.coll),
            help="Collision (0/1)",
        )
        parser.add_argument(
            "--fov",
            type=int,
            default=int(SimulationConfig.fov),
            help="Field of View (0/1)",
        )
        parser.add_argument(
            "--dynamics",
            type=int,
            default=SimulationConfig.mode_dynamics,
            help="Legacy dynamics selector (0: Single, 1: Double)",
        )
        parser.add_argument(
            "--rec",
            type=int,
            default=int(SimulationConfig.rec),
            help="Recentered barrier",
        )
        parser.add_argument(
            "--dt",
            type=float,
            default=SimulationConfig.dt,
            help="Simulation step",
        )
        parser.add_argument(
            "--steps",
            type=int,
            default=SimulationConfig.steps,
            help="Simulation steps",
        )

        parser.add_argument(
            "--c1", type=float, default=SimulationConfig.c1, help="Outer-loop gain"
        )
        parser.add_argument(
            "--c2", type=float, default=SimulationConfig.c2, help="Inner-loop gain"
        )
        parser.add_argument(
            "--c3", type=float, default=SimulationConfig.c3, help="Velocity gain"
        )
        parser.add_argument(
            "--c4_h",
            type=float,
            default=SimulationConfig.c4_h,
            help="FOV horiz gain",
        )
        parser.add_argument(
            "--c4_v",
            type=float,
            default=SimulationConfig.c4_v,
            help="FOV vert gain",
        )

        parser.add_argument(
            "--wn", type=float, default=SimulationConfig.wn, help="Filter wn"
        )
        parser.add_argument(
            "--v_max",
            type=float,
            default=SimulationConfig.v_max,
            help="Max velocity",
        )
        parser.add_argument(
            "--a_max",
            type=float,
            default=SimulationConfig.a_max,
            help="Max acceleration",
        )
        parser.add_argument(
            "--d_max",
            type=float,
            default=SimulationConfig.d_max,
            help="Connectivity limit",
        )
        parser.add_argument(
            "--d_min",
            type=float,
            default=SimulationConfig.d_min,
            help="Collision limit",
        )
        parser.add_argument(
            "--r_max",
            type=float,
            default=SimulationConfig.r_max,
            help="Max rotation rate",
        )
        parser.add_argument(
            "--R", type=float, default=SimulationConfig.R, help="Formation radius"
        )
        parser.add_argument(
            "--l", type=float, default=SimulationConfig.l, help="Spawn area side"
        )

        parser.add_argument(
            "--kp_yaw", type=float, default=SimulationConfig.kp_yaw, help="Yaw gain"
        )
        parser.add_argument(
            "--kp_pitch",
            type=float,
            default=SimulationConfig.kp_pitch,
            help="Pitch gain",
        )
        parser.add_argument(
            "--theta_max_x",
            type=float,
            default=SimulationConfig.theta_max_x,
            help="FOV range horizontal",
        )
        parser.add_argument(
            "--theta_max_y",
            type=float,
            default=SimulationConfig.theta_max_y,
            help="FOV range vertical",
        )
        parser.add_argument(
            "--alpha_h_des",
            type=float,
            default=SimulationConfig.alpha_h_des,
            help="Desired normalised horizontal camera coordinate",
        )
        parser.add_argument(
            "--alpha_v_des",
            type=float,
            default=SimulationConfig.alpha_v_des,
            help="Desired normalised vertical camera coordinate",
        )
        parser.add_argument(
            "--fov_eps",
            type=float,
            default=SimulationConfig.fov_eps,
            help="FoV numerical regularisation",
        )
        parser.add_argument(
            "--k_fov_omega",
            type=float,
            default=SimulationConfig.k_fov_omega,
            help="FoV rotational backstepping gain",
        )
        parser.add_argument(
            "--k_fov_backstep",
            type=float,
            default=SimulationConfig.k_fov_backstep,
            help="Angular-rate tracking error gain",
        )
        parser.add_argument(
            "--omega_fov_max",
            type=float,
            default=SimulationConfig.omega_fov_max,
            help="Componentwise saturation on omega_des (rad/s)",
        )

        parser.add_argument(
            "--simulation_mode",
            type=str,
            default=SimulationConfig.simulation_mode,
            choices=[
                "single_integrator",
                "double_integrator",
                "bluerov_feedback_linearization",
                "bluerov_clf_qp_tau",
                "bluerov_clf_qp_thrusters",
            ],
            help="Simulation/controller mode",
        )
        parser.add_argument(
            "--filter_type",
            type=str,
            default=SimulationConfig.filter_type,
            choices=["first_order", "second_order"],
            help="Command filter type",
        )
        parser.add_argument(
            "--lam",
            type=float,
            default=SimulationConfig.lam,
            help="First-order bandwidth",
        )
        parser.add_argument(
            "--zeta",
            type=float,
            default=SimulationConfig.zeta,
            help="Second-order damping",
        )
        parser.add_argument(
            "--use_quaternions",
            type=int,
            default=int(SimulationConfig.use_quaternions),
            help="Use quaternion state representation (0/1)",
        )
        parser.add_argument(
            "--tau_v_max",
            type=float,
            default=SimulationConfig.tau_v_max,
            help="Max translational generalized force norm",
        )
        parser.add_argument(
            "--clf_alpha",
            type=float,
            default=SimulationConfig.clf_alpha,
            help="CLF decay (linear slope of the class K function)",
        )
        parser.add_argument(
            "--clf_alpha_sat",
            type=float,
            default=SimulationConfig.clf_alpha_sat,
            help="Saturation ceiling of the class K function; None = linear",
        )
        parser.add_argument(
            "--clf_slack_weight",
            type=float,
            default=SimulationConfig.clf_slack_weight,
            help="CLF-QP slack penalty",
        )
        parser.add_argument(
            "--clf_force_weight",
            type=float,
            default=SimulationConfig.clf_force_weight,
            help="CLF-QP force penalty",
        )
        parser.add_argument(
            "--clf_torque_weight",
            type=float,
            default=SimulationConfig.clf_torque_weight,
            help="CLF-QP torque penalty (high → QP sets τ_ω≈0 when FoV inactive)",
        )
        parser.add_argument(
            "--thrust_weight",
            type=float,
            default=SimulationConfig.thrust_weight,
            help="Direct-thruster QP penalty",
        )
        parser.add_argument(
            "--lgv_epsilon",
            type=float,
            default=SimulationConfig.lgv_epsilon,
            help="Small LgV",
        )
        parser.add_argument(
            "--saturation_mode",
            type=str,
            default=SimulationConfig.saturation_mode,
            choices=["hard", "smooth", "hybrid"],
            help="Saturation mode",
        )
        parser.add_argument(
            "--saturation_switch_ratio",
            type=float,
            default=SimulationConfig.saturation_switch_ratio,
            help="Hybrid saturation switch ratio",
        )
        parser.add_argument(
            "--thruster_limit",
            type=float,
            default=SimulationConfig.thruster_limit,
            help="Thruster force limit",
        )
        parser.add_argument(
            "--k_form",
            type=float,
            default=SimulationConfig.k_form,
            help="Formation gain",
        )
        parser.add_argument(
            "--k_att",
            type=float,
            default=SimulationConfig.k_att,
            help="Attitude proportional gain",
        )
        parser.add_argument(
            "--k_att_d",
            type=float,
            default=SimulationConfig.k_att_d,
            help="Attitude derivative (damping) gain",
        )
        parser.add_argument(
            "--use_adaptive_barrier",
            type=int,
            default=int(SimulationConfig.use_adaptive_barrier),
            help="Enable adaptive barrier-domain relaxation (0/1)",
        )
        parser.add_argument(
            "--lambda_delta",
            type=float,
            default=SimulationConfig.lambda_delta,
            help="Adaptive decay rate for lower margin",
        )
        parser.add_argument(
            "--gamma_delta",
            type=float,
            default=SimulationConfig.gamma_delta,
            help="Adaptive activation gain for lower margin",
        )
        parser.add_argument(
            "--sigma_delta_max",
            type=float,
            default=SimulationConfig.sigma_delta_max,
            help="Maximum relaxation for lower margin",
        )
        parser.add_argument(
            "--lambda_Delta",
            type=float,
            default=SimulationConfig.lambda_Delta,
            help="Adaptive decay rate for upper margin",
        )
        parser.add_argument(
            "--gamma_Delta",
            type=float,
            default=SimulationConfig.gamma_Delta,
            help="Adaptive activation gain for upper margin",
        )
        parser.add_argument(
            "--sigma_Delta_max",
            type=float,
            default=SimulationConfig.sigma_Delta_max,
            help="Maximum relaxation for upper margin",
        )
        parser.add_argument(
            "--adaptive_dt",
            type=float,
            default=SimulationConfig.adaptive_dt,
            help="Integration step for adaptive barrier dynamics",
        )
        parser.add_argument(
            "--leader_vel_amp",
            type=float,
            default=SimulationConfig.leader_vel_amp,
            help="Leader velocity profile amplitude",
        )
        parser.add_argument(
            "--leader_freq",
            type=float,
            default=SimulationConfig.leader_freq,
            help="Leader velocity profile frequency",
        )
        parser.add_argument(
            "--leader_accel_steps",
            type=int,
            default=SimulationConfig.leader_accel_steps,
            help="Number of steps to accelerate leader",
        )
        parser.add_argument(
            "--leader_brake_steps",
            type=int,
            default=SimulationConfig.leader_brake_steps,
            help="Number of steps to brake leader back to zero velocity",
        )
        parser.add_argument(
            "--tree_type",
            type=str,
            default=SimulationConfig.tree_type,
            choices=["leader_star", "leader_chain", "random_tree"],
            help="Directed tree topology",
        )
        parser.add_argument(
            "--random_seed",
            type=int,
            default=SimulationConfig.random_seed,
            help="Random seed for scenario generation",
        )
        parser.add_argument(
            "--init_noise",
            type=float,
            default=SimulationConfig.init_noise,
            help="Initial position noise around desired offsets",
        )
        parser.add_argument(
            "--max_scenario_tries",
            type=int,
            default=SimulationConfig.max_scenario_tries,
            help="Maximum attempts to generate a feasible scenario",
        )

        args = vars(parser.parse_args())
        args["mode_graph"] = args.pop("graph")
        args["mode_dynamics"] = args.pop("dynamics")
        return Parameters(**args)
