#!/usr/bin/env python3

##########################################################
#                                                        #
# Parameters to the Formation Controller                 #
# Code Author: Joao Zenario                              #
# Refactored architecture: config + scenario + logs      #
#                                                        #
##########################################################

from dataclasses import dataclass, field
import argparse
import numpy as np


@dataclass
class SimulationConfig:
    """User-selected simulation settings."""

    N: int = 5
    dim: int = 3
    mode_graph: int = 1
    conn: bool = True
    coll: bool = True
    fov: bool = True
    rec: bool = True
    mode_dynamics: int = 1

    dt: float = 0.01
    steps: int = 1000

    c1: float = 1.0
    c2: float = 1.0
    c3: float = 1.0
    c4_h: float = 0.5
    c4_v: float = 0.5

    wn: float = 10.0
    zeta: float = 1.0
    lam: float = 5.0

    v_max: float = 1.0
    a_max: float = 1.0
    d_max: float = 5.0
    d_min: float = 0.5
    r_max: float = 1.5

    R: float = 2.0
    l: float = 7.0

    kp_yaw: float = 2.0
    kp_pitch: float = 1.0
    theta_max_x: float = float(np.radians(45))
    theta_max_y: float = float(np.radians(30))

    simulation_mode: str = "bluerov_feedback_linearization"
    filter_type: str = "second_order"
    use_quaternions: bool = True

    tau_v_max: float | None = None
    clf_alpha: float = 2.0
    clf_slack_weight: float = 1000.0
    clf_force_weight: float = 1.0
    thrust_weight: float = 1.0
    lgv_epsilon: float = 1e-6
    saturation_mode: str = "smooth"
    saturation_switch_ratio: float = 4.0
    thruster_limit: float = 40.0
    k_form: float = 1.0

    leader_vel_amp: float = 0.5
    leader_freq: float = 0.5
    leader_accel_steps: int = 200
    leader_brake_steps: int = 200

    counter: int = 0

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
        if self.clf_slack_weight <= 0.0:
            raise ValueError("clf_slack_weight must be positive.")
        if self.clf_force_weight <= 0.0:
            raise ValueError("clf_force_weight must be positive.")
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
        if self.leader_accel_steps < 0:
            raise ValueError("leader_accel_steps must be nonnegative.")
        if self.leader_brake_steps < 0:
            raise ValueError("leader_brake_steps must be nonnegative.")


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
    def _default_positions_adjacency(
        cfg: SimulationConfig,
    ) -> tuple[np.ndarray, np.ndarray]:
        if cfg.N == 5:
            x = np.array(
                [
                    [3.39655962, 0.29968507, 4.17778228],
                    [3.30965058, 1.82937878, 0.09709306],
                    [4.46324661, 3.24636054, 2.73822641],
                    [0.13316123, 1.37122818, 4.99218486],
                    [0.04021556, 4.23317497, 2.23731517],
                ],
                dtype=float,
            )
            x = x[:, : cfg.dim]
            A = np.array(
                [
                    [0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0],
                ],
                dtype=int,
            )
            return x, A

        x = cfg.l * np.random.rand(cfg.N, cfg.dim)
        A = np.zeros((cfg.N, cfg.N), dtype=int)
        for i in range(1, cfg.N):
            A[i, i - 1] = 1
        return x, A

    @classmethod
    def build(cls, cfg: SimulationConfig) -> "Scenario":
        x, A = cls._default_positions_adjacency(cfg)
        q = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (cfg.N, 1))

        o = np.zeros((cfg.N, cfg.dim))
        for i in range(1, cfg.N):
            angle = (2.0 * np.pi / max(cfg.N - 1, 1)) * (i - 1)
            if cfg.dim >= 2:
                o[i, 0] = cfg.R * np.cos(angle)
                o[i, 1] = cfg.R * np.sin(angle)
            if cfg.dim == 3:
                o[i, 2] = 0.0

        relative_pos_desired = np.zeros((cfg.N, cfg.N, cfg.dim))
        target_dist = np.zeros(cfg.N)
        for i in range(cfg.N):
            for j in range(cfg.N):
                relative_pos_desired[i, j] = o[i] - o[j]
            if i:
                target_dist[i] = np.linalg.norm(relative_pos_desired[i, 0])

        E = np.empty((0, 2), dtype=np.int64)
        NE = 0

        dist = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=2)

        x_lf = np.zeros((cfg.N, cfg.dim))
        dot_x_lf = np.zeros((cfg.N, cfg.dim))

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
    Backward-compatible facade over clean architecture components:
    - config: SimulationConfig
    - scenario: Scenario
    - data: SimulationData
    """

    _INTERNAL = {"config", "scenario", "data"}

    def __init__(
        self,
        N=5,
        dim=3,
        mode_graph=1,
        conn=1,
        coll=1,
        fov=1,
        rec=1,
        dt=0.01,
        steps=1000,
        c1=1.0,
        c2=1.0,
        c3=1.0,
        c4_h=0.5,
        c4_v=0.5,
        wn=10.0,
        v_max=1.0,
        a_max=1.0,
        d_max=5.0,
        d_min=0.5,
        r_max=1.5,
        R=2.0,
        l=7.0,
        kp_yaw=2.0,
        kp_pitch=1.0,
        theta_max_x=np.radians(45),
        theta_max_y=np.radians(30),
        simulation_mode="bluerov_feedback_linearization",
        filter_type="second_order",
        lam=5.0,
        zeta=1.0,
        use_quaternions=True,
        tau_v_max=None,
        clf_alpha=2.0,
        clf_slack_weight=1000.0,
        clf_force_weight=1.0,
        thrust_weight=1.0,
        lgv_epsilon=1e-6,
        saturation_mode="smooth",
        saturation_switch_ratio=4.0,
        thruster_limit=40.0,
        k_form=1.0,
        mode_dynamics=1,
        leader_accel_steps=200,
        leader_brake_steps=200,
    ):
        cfg = SimulationConfig(
            N=N,
            dim=dim,
            mode_graph=mode_graph,
            conn=bool(conn),
            coll=bool(coll),
            fov=bool(fov),
            rec=bool(rec),
            mode_dynamics=mode_dynamics,
            dt=dt,
            steps=steps,
            c1=c1,
            c2=c2,
            c3=c3,
            c4_h=c4_h,
            c4_v=c4_v,
            wn=wn,
            zeta=zeta,
            lam=lam,
            v_max=v_max,
            a_max=a_max,
            d_max=d_max,
            d_min=d_min,
            r_max=r_max,
            R=R,
            l=l,
            kp_yaw=kp_yaw,
            kp_pitch=kp_pitch,
            theta_max_x=theta_max_x,
            theta_max_y=theta_max_y,
            simulation_mode=simulation_mode,
            filter_type=filter_type,
            use_quaternions=bool(use_quaternions),
            tau_v_max=tau_v_max,
            clf_alpha=clf_alpha,
            clf_slack_weight=clf_slack_weight,
            clf_force_weight=clf_force_weight,
            thrust_weight=thrust_weight,
            lgv_epsilon=lgv_epsilon,
            saturation_mode=saturation_mode,
            saturation_switch_ratio=saturation_switch_ratio,
            thruster_limit=thruster_limit,
            k_form=k_form,
            leader_accel_steps=leader_accel_steps,
            leader_brake_steps=leader_brake_steps,
        )
        cfg.validate()

        object.__setattr__(self, "config", cfg)
        object.__setattr__(self, "scenario", Scenario.build(cfg))
        object.__setattr__(self, "data", SimulationData.empty(cfg))

    def __getattr__(self, name):
        for container_name in ("config", "scenario", "data"):
            container = object.__getattribute__(self, container_name)
            if hasattr(container, name):
                return getattr(container, name)
        raise AttributeError(f"{type(self).__name__} has no attribute '{name}'")

    def __setattr__(self, name, value):
        if name in self._INTERNAL or name.startswith("_"):
            object.__setattr__(self, name, value)
            return

        for container_name in ("config", "scenario", "data"):
            if container_name in self.__dict__:
                container = self.__dict__[container_name]
                if hasattr(container, name):
                    setattr(container, name, value)
                    return

        object.__setattr__(self, name, value)

    def get_leader_acceleration(self, step: int) -> np.ndarray:
        """
        Accelerate then brake so leader velocity returns close to zero.
        """
        accel = np.zeros(self.dim)
        accel_end = self.leader_accel_steps
        brake_end = self.leader_accel_steps + self.leader_brake_steps

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

        parser.add_argument("--N", type=int, default=5, help="Number of agents")
        parser.add_argument("--dim", type=int, default=3, help="Dimension (2D or 3D)")
        parser.add_argument(
            "--graph",
            type=int,
            default=1,
            help="Graph type (0: Undirected, 1: Directed)",
        )
        parser.add_argument("--conn", type=int, default=1, help="Connectivity (0/1)")
        parser.add_argument("--coll", type=int, default=1, help="Collision (0/1)")
        parser.add_argument("--fov", type=int, default=1, help="Field of View (0/1)")
        parser.add_argument(
            "--dynamics",
            type=int,
            default=1,
            help="Legacy dynamics selector (0: Single, 1: Double)",
        )
        parser.add_argument("--rec", type=int, default=1, help="Recentered barrier")
        parser.add_argument("--dt", type=float, default=0.01, help="Simulation step")
        parser.add_argument("--steps", type=int, default=1000, help="Simulation steps")

        parser.add_argument("--c1", type=float, default=1.0, help="Outer-loop gain")
        parser.add_argument("--c2", type=float, default=1.0, help="Inner-loop gain")
        parser.add_argument("--c3", type=float, default=1.0, help="Velocity gain")
        parser.add_argument("--c4_h", type=float, default=0.5, help="FOV horiz gain")
        parser.add_argument("--c4_v", type=float, default=0.5, help="FOV vert gain")

        parser.add_argument("--wn", type=float, default=10.0, help="Filter wn")
        parser.add_argument("--v_max", type=float, default=1.0, help="Max velocity")
        parser.add_argument("--a_max", type=float, default=1.0, help="Max acceleration")
        parser.add_argument(
            "--d_max", type=float, default=5.0, help="Connectivity limit"
        )
        parser.add_argument("--d_min", type=float, default=0.5, help="Collision limit")
        parser.add_argument(
            "--r_max", type=float, default=1.5, help="Max rotation rate"
        )
        parser.add_argument("--R", type=float, default=2.0, help="Formation radius")
        parser.add_argument("--l", type=float, default=5.0, help="Spawn area side")

        parser.add_argument("--kp_yaw", type=float, default=2.0, help="Yaw gain")
        parser.add_argument("--kp_pitch", type=float, default=1.0, help="Pitch gain")
        parser.add_argument(
            "--theta_max_x",
            type=float,
            default=np.radians(45),
            help="FOV range horizontal",
        )
        parser.add_argument(
            "--theta_max_y",
            type=float,
            default=np.radians(30),
            help="FOV range vertical",
        )

        parser.add_argument(
            "--simulation_mode",
            type=str,
            default="bluerov_feedback_linearization",
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
            default="second_order",
            choices=["first_order", "second_order"],
            help="Command filter type",
        )
        parser.add_argument(
            "--lam", type=float, default=5.0, help="First-order bandwidth"
        )
        parser.add_argument(
            "--zeta", type=float, default=1.0, help="Second-order damping"
        )
        parser.add_argument(
            "--use_quaternions",
            type=int,
            default=1,
            help="Use quaternion state representation (0/1)",
        )
        parser.add_argument(
            "--tau_v_max",
            type=float,
            default=None,
            help="Max translational generalized force norm",
        )
        parser.add_argument("--clf_alpha", type=float, default=2.0, help="CLF decay")
        parser.add_argument(
            "--clf_slack_weight",
            type=float,
            default=1000.0,
            help="CLF-QP slack penalty",
        )
        parser.add_argument(
            "--clf_force_weight", type=float, default=1.0, help="CLF-QP force penalty"
        )
        parser.add_argument(
            "--thrust_weight",
            type=float,
            default=1.0,
            help="Direct-thruster QP penalty",
        )
        parser.add_argument("--lgv_epsilon", type=float, default=1e-6, help="Small LgV")
        parser.add_argument(
            "--saturation_mode",
            type=str,
            default="smooth",
            choices=["hard", "smooth", "hybrid"],
            help="Saturation mode",
        )
        parser.add_argument(
            "--saturation_switch_ratio",
            type=float,
            default=4.0,
            help="Hybrid saturation switch ratio",
        )
        parser.add_argument(
            "--thruster_limit", type=float, default=40.0, help="Thruster force limit"
        )
        parser.add_argument("--k_form", type=float, default=1.0, help="Formation gain")
        parser.add_argument(
            "--leader_accel_steps",
            type=int,
            default=200,
            help="Number of steps to accelerate leader",
        )
        parser.add_argument(
            "--leader_brake_steps",
            type=int,
            default=200,
            help="Number of steps to brake leader back to zero velocity",
        )

        args = parser.parse_args()

        return Parameters(
            N=args.N,
            dim=args.dim,
            mode_graph=args.graph,
            conn=args.conn,
            coll=args.coll,
            fov=args.fov,
            rec=args.rec,
            mode_dynamics=args.dynamics,
            dt=args.dt,
            steps=args.steps,
            c1=args.c1,
            c2=args.c2,
            c3=args.c3,
            c4_h=args.c4_h,
            c4_v=args.c4_v,
            wn=args.wn,
            v_max=args.v_max,
            a_max=args.a_max,
            d_max=args.d_max,
            d_min=args.d_min,
            r_max=args.r_max,
            R=args.R,
            l=args.l,
            kp_yaw=args.kp_yaw,
            kp_pitch=args.kp_pitch,
            theta_max_x=args.theta_max_x,
            theta_max_y=args.theta_max_y,
            simulation_mode=args.simulation_mode,
            filter_type=args.filter_type,
            lam=args.lam,
            zeta=args.zeta,
            use_quaternions=bool(args.use_quaternions),
            tau_v_max=args.tau_v_max,
            clf_alpha=args.clf_alpha,
            clf_slack_weight=args.clf_slack_weight,
            clf_force_weight=args.clf_force_weight,
            thrust_weight=args.thrust_weight,
            lgv_epsilon=args.lgv_epsilon,
            saturation_mode=args.saturation_mode,
            saturation_switch_ratio=args.saturation_switch_ratio,
            thruster_limit=args.thruster_limit,
            k_form=args.k_form,
            leader_accel_steps=args.leader_accel_steps,
            leader_brake_steps=args.leader_brake_steps,
        )
