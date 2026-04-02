#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Tuple

import numpy as np

from robot_simulator import RobotSimulator
from graphs_utils import get_neighbor_lists_from_adjacency
from systems import InputAffineNLSystem, RK4Integrator
from parameters_refactored import Parameters
from blue_rov2_refactored import BlueROV2

from controller_refactored import (
    SingleIntegratorConfig,
    DoubleIntegratorConfig,
    BlueROV2Config,
    SingleIntegratorController,
    DoubleIntegratorController,
    BlueROV2Controller,
)


DO_ANIMATION = False
DO_TOPOLOGY_PLOT = True
DO_BARRIER_PLOT = True


SimulationMode = Literal[
    "single_integrator",
    "double_integrator",
    "bluerov_feedback_linearization",
    "bluerov_clf_qp_tau",
    "bluerov_clf_qp_thrusters",
]


@dataclass
class SimulationOptions:
    mode: SimulationMode
    show_progress: bool = True


class FormationControlSimulation:
    def __init__(self, params: Parameters, options: SimulationOptions):
        self.params = params
        self.cfg = params.config
        self.scenario = params.scenario
        self.data = params.data
        self.options = options
        self.mode = options.mode

        print("\n" + "=" * 72)
        print("                FORMATION CONTROL SIMULATION")
        print(f"                Mode: {self.mode}")
        print("=" * 72)

        self.neighbors_dict = get_neighbor_lists_from_adjacency(self.scenario.A)

        self.simulators = self._initialize_simulators()
        self.controllers = self._initialize_controllers()

        self.leader_velocity_ref = np.zeros(3)

        self._initialize_state_tracking()
        print("Initialization complete.\n")

    # ============================================================
    # Config builders
    # ============================================================

    def _single_integrator_config(self) -> SingleIntegratorConfig:
        return SingleIntegratorConfig(
            dim=self.cfg.dim,
            d_min=self.cfg.d_min,
            d_max=self.cfg.d_max,
            k_form=self.cfg.k_form,
            use_connectivity=bool(self.cfg.conn),
            use_collision=bool(self.cfg.coll),
            use_recentered=bool(self.cfg.rec),
            c_outer=self.cfg.c1,
            v_max=self.cfg.v_max,
        )

    def _double_integrator_config(self) -> DoubleIntegratorConfig:
        return DoubleIntegratorConfig(
            dim=self.cfg.dim,
            d_min=self.cfg.d_min,
            d_max=self.cfg.d_max,
            k_form=self.cfg.k_form,
            use_connectivity=bool(self.cfg.conn),
            use_collision=bool(self.cfg.coll),
            use_recentered=bool(self.cfg.rec),
            c_outer=self.cfg.c1,
            k_vel=self.cfg.c2,
            v_max=self.cfg.v_max,
            a_max=self.cfg.a_max,
            filter_type=self.cfg.filter_type,
            dt=self.cfg.dt,
            lam=self.cfg.lam,
            wn=self.cfg.wn,
            zeta=self.cfg.zeta,
        )

    def _bluerov_config(self) -> BlueROV2Config:
        return BlueROV2Config(
            dim=3,
            d_min=self.cfg.d_min,
            d_max=self.cfg.d_max,
            k_form=self.cfg.k_form,
            use_connectivity=bool(self.cfg.conn),
            use_collision=bool(self.cfg.coll),
            use_recentered=bool(self.cfg.rec),
            c_outer=self.cfg.c1,
            k_v=self.cfg.c2,
            filter_type=self.cfg.filter_type,
            dt=self.cfg.dt,
            lam=self.cfg.lam,
            wn=self.cfg.wn,
            zeta=self.cfg.zeta,
            v_max=self.cfg.v_max,
            tau_v_max=self.cfg.tau_v_max,
            clf_alpha=self.cfg.clf_alpha,
            clf_slack_weight=self.cfg.clf_slack_weight,
            clf_force_weight=self.cfg.clf_force_weight,
            thrust_weight=self.cfg.thrust_weight,
            lgv_epsilon=self.cfg.lgv_epsilon,
            saturation_switch_ratio=self.cfg.saturation_switch_ratio,
        )

    # ============================================================
    # Initialization
    # ============================================================

    def _initialize_simulators(self) -> List[RobotSimulator]:
        simulators = []
        for i in range(self.cfg.N):
            if self.mode == "single_integrator":
                si_model = InputAffineNLSystem(
                    f=lambda x: np.zeros(3),
                    G=lambda x: np.eye(3),
                    state_dim=3,
                    input_dim=3,
                )
                x0 = self.scenario.x[i].astype(float).copy()
                sim = RobotSimulator(i, si_model, x0, self.cfg.dt, RK4Integrator())

            elif self.mode == "double_integrator":
                di_model = InputAffineNLSystem(
                    f=lambda x: np.hstack([x[3:6], np.zeros(3)]),
                    G=lambda x: np.vstack([np.zeros((3, 3)), np.eye(3)]),
                    state_dim=6,
                    input_dim=3,
                )
                x0 = np.zeros(6, dtype=float)
                x0[0:3] = self.scenario.x[i]
                sim = RobotSimulator(i, di_model, x0, self.cfg.dt, RK4Integrator())

            else:
                rov_model = BlueROV2()
                if self.cfg.use_quaternions:
                    x0 = np.zeros(13)
                    x0[0:3] = self.scenario.x[i]
                    x0[3:7] = self.scenario.q[i]  # [x, y, z, w]
                    x0[7:13] = 0.0  # [u, v, w, wx, wy, wz]
                else:
                    x0 = np.zeros(12)
                    x0[0:3] = self.scenario.x[i]
                    x0[3:6] = 0.0
                    x0[6:12] = 0.0
                sim = RobotSimulator(i, rov_model, x0, self.cfg.dt, RK4Integrator())

            simulators.append(sim)

        return simulators

    def _initialize_controllers(self) -> List:
        if self.mode == "single_integrator":
            cfg = self._single_integrator_config()
            return [SingleIntegratorController(i, cfg) for i in range(self.cfg.N)]

        if self.mode == "double_integrator":
            cfg = self._double_integrator_config()
            return [DoubleIntegratorController(i, cfg) for i in range(self.cfg.N)]

        if self.mode in {
            "bluerov_feedback_linearization",
            "bluerov_clf_qp_tau",
            "bluerov_clf_qp_thrusters",
        }:
            cfg = self._bluerov_config()
            return [BlueROV2Controller(i, cfg, BlueROV2()) for i in range(self.cfg.N)]

        raise ValueError(f"Unsupported simulation mode: {self.mode}")

    def _initialize_state_tracking(self) -> None:
        self.x = np.array([self._get_position(sim.state) for sim in self.simulators])
        self.v = np.array(
            [self._get_linear_velocity(sim.state) for sim in self.simulators]
        )

        if self.mode.startswith("bluerov") and self.cfg.use_quaternions:
            self.q = np.array(
                [self._get_quaternion(sim.state) for sim in self.simulators]
            )
        else:
            self.q = None

        self.data.history = self.x[np.newaxis, :, :]
        self.data.vel_history = [[np.linalg.norm(vi) for vi in self.v]]
        self.data.v_history_full = [self.v.copy()]
        self.data.accel_history = [[0.0] * self.cfg.N]
        self.data.err_history = [self._compute_formation_errors()]
        self.data.tau_history = [[0.0] * self.cfg.N]

        if self.cfg.fov:
            # Keep a consistent shape: (time, N, 2) for [horizontal, vertical] errors in deg.
            self.data.fov_history = [[[0.0, 0.0] for _ in range(self.cfg.N)]]

        if self.q is not None:
            self.data.q_history = [self.q.copy()]

    # ============================================================
    # State readers
    # ============================================================

    def _get_position(self, state: np.ndarray) -> np.ndarray:
        if self.mode == "single_integrator":
            return state[0:3]
        if self.mode == "double_integrator":
            return state[0:3]
        return state[0:3]

    def _get_quaternion(self, state: np.ndarray) -> np.ndarray:
        return state[3:7]

    def _get_linear_velocity(self, state: np.ndarray) -> np.ndarray:
        if self.mode == "single_integrator":
            return np.zeros(3)
        if self.mode == "double_integrator":
            return state[3:6]
        if self.cfg.use_quaternions:
            return state[7:10]
        return state[6:9]

    def _get_body_velocity6(self, state: np.ndarray) -> np.ndarray:
        if self.mode in {"single_integrator", "double_integrator"}:
            return np.zeros(6)
        if self.cfg.use_quaternions:
            return state[7:13]
        return state[6:12]

    # ============================================================
    # Local graph data
    # ============================================================

    def _local_relative_data(self, i: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns relative displacements, desired relative displacements,
        and local incidence signs for agent i.

        Since we use the local convention d_k = p_i - p_j, the incidence signs
        are simply +1 for all local edges.
        """
        neighbors_i = self.neighbors_dict[i]

        if len(neighbors_i) == 0:
            rel_state = np.zeros((0, self.cfg.dim))
            desired_rel = np.zeros((0, self.cfg.dim))
            incidence_signs = np.zeros((0,))
            return rel_state, desired_rel, incidence_signs

        p_i = self._get_position(self.simulators[i].state)
        neighbors_pos = np.array(
            [self._get_position(self.simulators[j].state) for j in neighbors_i]
        )

        rel_state = p_i - neighbors_pos
        desired_rel = self.scenario.relative_pos_desired[i, neighbors_i]
        incidence_signs = np.ones(len(neighbors_i))

        return rel_state, desired_rel, incidence_signs

    # ============================================================
    # Leader reference
    # ============================================================

    def _leader_command(self, step: int) -> Tuple[np.ndarray, np.ndarray]:
        accel_leader = np.asarray(
            self.params.get_leader_acceleration(step), dtype=float
        ).reshape(3)

        accel_norm = np.linalg.norm(accel_leader)
        if accel_norm > self.cfg.a_max and accel_norm > 0.0:
            accel_leader *= self.cfg.a_max / accel_norm

        self.leader_velocity_ref += accel_leader * self.cfg.dt
        return self.leader_velocity_ref.copy(), accel_leader.copy()

    # ============================================================
    # Control computation
    # ============================================================

    def _compute_control_single_integrator(self, i: int, step: int) -> np.ndarray:
        rel_state, desired_rel, _ = self._local_relative_data(i)

        if i == 0:
            v_lead, _ = self._leader_command(step)
            return self.controllers[i].do_control(
                relative_states=rel_state,
                desired_relative_states=desired_rel,
                is_leader=True,
                leader_velocity=v_lead,
            )

        return self.controllers[i].do_control(
            relative_states=rel_state,
            desired_relative_states=desired_rel,
            is_leader=False,
        )

    def _compute_control_double_integrator(self, i: int, step: int) -> np.ndarray:
        rel_state, desired_rel, _ = self._local_relative_data(i)
        vel_i = self._get_linear_velocity(self.simulators[i].state)

        if i == 0:
            v_lead, a_lead = self._leader_command(step)
            return self.controllers[i].do_control(
                relative_states=rel_state,
                desired_relative_states=desired_rel,
                velocity_i=vel_i,
                is_leader=True,
                leader_velocity=v_lead,
                leader_acceleration=a_lead,
            )

        return self.controllers[i].do_control(
            relative_states=rel_state,
            desired_relative_states=desired_rel,
            velocity_i=vel_i,
            is_leader=False,
        )

    def _compute_control_bluerov(self, i: int, step: int):
        rel_state, desired_rel, incidence_signs = self._local_relative_data(i)

        state_i = self.simulators[i].state
        p_i = self._get_position(state_i)
        q_i = self._get_quaternion(state_i)
        nu_i = self._get_body_velocity6(state_i)

        ctrl: BlueROV2Controller = self.controllers[i]

        if i == 0:
            v_lead, _ = self._leader_command(step)
            ctrl.update_virtual_velocity_filter(v_lead)

        if self.mode == "bluerov_feedback_linearization":
            tau_i, debug = ctrl.do_control(
                p_i=p_i,
                q_i=q_i,
                nu_i=nu_i,
                relative_states=rel_state,
                desired_relative_states=desired_rel,
                incidence_signs=incidence_signs,
                mode="feedback_linearization",
            )
            return tau_i, debug

        if self.mode == "bluerov_clf_qp_tau":
            tau_i, debug = ctrl.do_control(
                p_i=p_i,
                q_i=q_i,
                nu_i=nu_i,
                relative_states=rel_state,
                desired_relative_states=desired_rel,
                incidence_signs=incidence_signs,
                mode="clf_qp_tau",
                local_fallback_tau_v=np.zeros(3),
            )
            return tau_i, debug

        if self.mode == "bluerov_clf_qp_thrusters":
            B = ctrl.actuator_matrix()
            m = B.shape[1]
            f_min = -self.cfg.thruster_limit * np.ones(m)
            f_max = +self.cfg.thruster_limit * np.ones(m)

            f_i, debug = ctrl.do_control(
                p_i=p_i,
                q_i=q_i,
                nu_i=nu_i,
                relative_states=rel_state,
                desired_relative_states=desired_rel,
                incidence_signs=incidence_signs,
                mode="clf_qp_thrusters",
                f_min=f_min,
                f_max=f_max,
            )

            tau_i = B @ f_i
            debug["tau_from_thrusters"] = tau_i
            return tau_i, debug

        raise ValueError(f"Unsupported BlueROV2 mode: {self.mode}")

    # ============================================================
    # Main run loop
    # ============================================================

    def run(self) -> Dict:
        print("Starting simulation...\n")

        self.u_history = []
        self.debug_history = []

        for step in range(self.cfg.steps):
            u_cmds = []
            step_debug = []

            # ---------- STEP 1: CONTROL ----------
            for i in range(self.cfg.N):
                if self.mode == "single_integrator":
                    u_i = self._compute_control_single_integrator(i, step)
                    debug_i = {}

                elif self.mode == "double_integrator":
                    u_i = self._compute_control_double_integrator(i, step)
                    debug_i = {}

                else:
                    u_i, debug_i = self._compute_control_bluerov(i, step)

                u_cmds.append(u_i)
                step_debug.append(debug_i)

            u_cmds = np.array(u_cmds, dtype=float)
            self.u_history.append(u_cmds.copy())
            self.debug_history.append(step_debug)

            # ---------- STEP 2: INTEGRATION ----------
            old_v = self.v.copy()
            for i in range(self.cfg.N):
                new_state = self.simulators[i].integrate_dynamics(u_cmds[i])
                self.x[i] = self._get_position(new_state)
                if self.mode == "single_integrator":
                    self.v[i] = u_cmds[i]
                else:
                    self.v[i] = self._get_linear_velocity(new_state)

                if self.q is not None:
                    self.q[i] = self._get_quaternion(new_state)

            # ---------- STEP 3: LOGGING ----------
            self.data.history = np.vstack([self.data.history, self.x[np.newaxis, :, :]])
            self.data.vel_history.append([np.linalg.norm(vi) for vi in self.v])
            self.data.v_history_full.append(self.v.copy())

            accel_step = np.linalg.norm((self.v - old_v) / self.cfg.dt, axis=1)
            self.data.accel_history.append(list(accel_step))
            self.data.err_history.append(self._compute_formation_errors())

            if self.mode == "bluerov_clf_qp_thrusters":
                # max absolute per-thruster force
                self.data.tau_history.append(
                    [float(np.max(np.abs(ui))) for ui in u_cmds]
                )
            elif self.mode.startswith("bluerov"):
                # norm of translational wrench (first 3 components)
                self.data.tau_history.append(
                    [
                        float(np.linalg.norm(np.asarray(ui, dtype=float)[:3]))
                        for ui in u_cmds
                    ]
                )
            else:
                # SI / DI: norm of the applied command vector
                self.data.tau_history.append(
                    [float(np.linalg.norm(ui)) for ui in u_cmds]
                )

            if self.q is not None:
                self.data.q_history.append(self.q.copy())

            if self.cfg.fov:
                # Placeholder FOV history until a dedicated FOV module populates real angles.
                self.data.fov_history.append([[0.0, 0.0] for _ in range(self.cfg.N)])

            # ---------- STEP 4: PROGRESS ----------
            if self.options.show_progress:
                progress_step = max(1, self.cfg.steps // 10)
                if (step + 1) % progress_step == 0 or step == self.cfg.steps - 1:
                    mean_err = (
                        np.mean(self.data.err_history[-1][1:])
                        if self.cfg.N > 1
                        else 0.0
                    )
                    print(
                        f"Step {step+1}/{self.cfg.steps} - Mean Formation Error: {mean_err:.3f} m"
                    )
                    for i in range(1, self.cfg.N):
                        print(
                            f"  Agent {i}: Error = {self.data.err_history[-1][i]:.3f} m"
                        )

                    print("-" * 60)

        # ---------- FINALIZE ----------
        self.data.vel_history = np.array(self.data.vel_history)
        self.data.v_history_full = np.array(self.data.v_history_full)
        self.data.accel_history = np.array(self.data.accel_history)
        self.data.err_history = np.array(self.data.err_history)
        self.data.tau_history = np.array(self.data.tau_history)

        if self.cfg.fov:
            self.data.fov_history = np.array(self.data.fov_history)

        if self.q is not None:
            self.data.q_history = np.array(self.data.q_history)

        return {
            "finished": True,
            "mode": self.mode,
            "u_history": self.u_history,
            "debug_history": self.debug_history,
        }

    # ============================================================
    # Metrics
    # ============================================================

    def _compute_formation_errors(self) -> np.ndarray:
        """
        Formation error in meters.

        For each follower i, the error is the Euclidean distance between the
        current position and the desired position relative to the leader:
            e_i = || x_i - (x_0 + o_i) ||.

        The leader error is set to zero.
        """
        errors = np.zeros(self.cfg.N)

        leader_pos = self.x[0]
        for i in range(1, self.cfg.N):
            desired_pos_i = leader_pos + self.scenario.o[i]
            errors[i] = np.linalg.norm(self.x[i] - desired_pos_i)

        return errors


def build_options_from_params(params: Parameters) -> SimulationOptions:
    """
    Decide the simulation mode.

    Priority:
    1) params.simulation_mode, if present
    2) fallback from old mode_dynamics
    """
    cfg = params.config
    if cfg.simulation_mode:
        mode = cfg.simulation_mode
    else:
        # Legacy fallback
        if cfg.mode_dynamics == 0:
            mode = "single_integrator"
        elif cfg.mode_dynamics == 1:
            mode = "double_integrator"
        else:
            mode = "bluerov_feedback_linearization"

    return SimulationOptions(mode=mode, show_progress=True)


def main():
    params = Parameters.function_arguments()
    options = build_options_from_params(params)

    sim = FormationControlSimulation(params, options)
    result = sim.run()

    from plot_animator_refactored import Animation

    animator = Animation(params)
    if DO_ANIMATION:
        animator.replay_animation(
            params,
            sim.u_history,
            show_topology=DO_TOPOLOGY_PLOT,
            show_barrier=DO_BARRIER_PLOT,
        )
    else:
        animator.show_static_plots(
            params,
            sim.u_history,
            show_topology=DO_TOPOLOGY_PLOT,
            show_barrier=DO_BARRIER_PLOT,
        )

    return result


if __name__ == "__main__":
    main()
