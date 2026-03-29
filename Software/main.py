#!/usr/bin/env python3

import numpy as np
from typing import List, Dict
from robot_simulator import RobotSimulator
from graphs_utils import get_neighbor_lists_from_adjacency
from controller import (
    DoubleIntegratorController,
    FieldOfViewController,
    BlueROV2Controller
)
from systems import RK4Integrator
from parameters import Parameters
from BlueROV2_thrust import BlueROV2

class FormationControlSimulation:
    def __init__(self, params: Parameters):
        self.params = params

        print("\n" + "="*60)
        print("        BLUE ROV2 FORMATION CONTROL SIMULATION")
        print("="*60)

        self.neighbors_dict = get_neighbor_lists_from_adjacency(self.params.A) # Topology.
        self.simulators = self._initialize_simulators() # Simulators.

        # Controllers
        self.formation_controllers = self._initialize_formation_controller() # Formation controller (single/double integrator).
        self.fov_controllers = self._initialize_fov_controller() if self.params.fov else None # FOV controller.
        self.controllers = self._initialize_BROV_controller() # BlueROV2 controller.

        self._initialize_state_tracking() # State Tracking and History Logging.
        
        # Internal state for leader velocity integration
        self.leader_v_star = np.zeros(3) 
        print("      Initialization Complete\n")

    def _initialize_simulators(self) -> List[RobotSimulator]:
        simulators = []
        for i in range(self.params.N):
            rov_model = BlueROV2() 
            x0 = np.zeros(12)
            x0[0:3] = self.params.x[i] 
            sim = RobotSimulator(i, rov_model, x0, self.params.dt, RK4Integrator())
            simulators.append(sim)
        return simulators

    def _initialize_formation_controller(self) -> List:
        f_controllers = []
        for i in range(self.params.N):
            ctrl = DoubleIntegratorController(i, self.params)
            f_controllers.append(ctrl)
        return f_controllers

    def _initialize_fov_controller(self) -> List[FieldOfViewController]:
        return [FieldOfViewController(i, self.params) for i in range(self.params.N)]

    def _initialize_BROV_controller(self) -> List[BlueROV2Controller]:
        controllers = []
        for i in range(self.params.N):
            rov_model = BlueROV2() 
            ctrl = BlueROV2Controller(i, self.params, rov_model)
            controllers.append(ctrl)
        return controllers

    def _initialize_state_tracking(self):
        """Initializes history and logs the initial state."""
        self.x = np.array([sim.state[0:3] for sim in self.simulators])
        self.v = np.array([sim.state[6:9] for sim in self.simulators])
        self.q = self.params.q.copy()
        
        self.params.history = self.x[np.newaxis, :, :] # (1, N, 3)
        self.params.vel_history = [[np.linalg.norm(vi) for vi in self.v]]
        self.params.v_history_full = [self.v.copy()]
        self.params.accel_history = [[0.0] * self.params.N]
        self.params.err_history = [self._compute_formation_errors()]
        self.params.tau_history = [[0.0] * self.params.N]
        
        if self.params.fov:
            self.params.fov_history = [[[0.0, 0.0]] * self.params.N]
            self.params.q_history = [self.q.copy()]

    def run(self) -> Dict:
        print("Starting Cascaded ROV Simulation...\n")
        u_history = []
        
        for step in range(self.params.steps):
            u_wrenches = []
            
            # --- STEP 1: COMPUTE CONTROLS ---
            for i in range(self.params.N):
                state_full = self.simulators[i].state
                neighbors_i = self.neighbors_dict[i]
                
                if len(neighbors_i) > 0:
                    neighbors_pos = np.array([self.simulators[j].state[0:3] for j in neighbors_i])
                    rel_state = state_full[0:3] - neighbors_pos
                else:
                    rel_state = np.array([]).reshape(0, self.params.dim)
                
                desired_rel = self.params.relative_pos_desired[i, neighbors_i]

                if i == 0: 
                    accel_leader = self.params.get_leader_acceleration(step)
                    # Saturate leader acceleration to a_max
                    accel_leader_norm = np.linalg.norm(accel_leader)
                    if accel_leader_norm > self.params.a_max:
                        accel_leader *= self.params.a_max / accel_leader_norm
                    self.leader_v_star += accel_leader * self.params.dt
                    v_star_3d = self.leader_v_star
                else:
                    v_star_3d = self.formation_controllers[i].doControl(
                        relative_states=rel_state,
                        desired_relative_states=desired_rel,
                        velocity_i=state_full[6:9]
                    )
                
                v_star_6d = np.zeros(6); v_star_6d[0:3] = v_star_3d

                tau_i = self.controllers[i].controller1(state_full, v_star_6d) # Controller1
                #tau_i = self.controllers[i].controller2(state_full, v_star_6d) # Controller2
                # Controller3
                #f_thrusters = self.controllers[i].controller3(state_full, v_star_6d)
                #tau_i = self.controllers[i].get_actuator_matrix() @ f_thrusters  # TRANSLATION STEP: Convert 8 forces into 1 6D Wrench. tau = B @ f

                # Now send the 6D Wrench to the simulator
                u_wrenches.append(tau_i)
            
            u_history.append(np.array(u_wrenches).copy())
            
            # --- STEP 2: INTEGRATE PHYSICS ---
            old_v = self.v.copy()
            for i in range(self.params.N):
                new_state = self.simulators[i].integrate_dynamics(u_wrenches[i])
                self.x[i] = new_state[0:3]
                self.v[i] = new_state[6:9]

                # Hard cap actual agent acceleration using a_max (post-dynamics safety clamp).
                dv = self.v[i] - old_v[i]
                dv_norm = np.linalg.norm(dv)
                max_dv = self.params.a_max * self.params.dt
                if dv_norm > max_dv and dv_norm > 0:
                    dv = dv * (max_dv / dv_norm)
                    corrected_v = old_v[i] + dv
                    self.v[i] = corrected_v
                    # Apply corrected velocity back to simulator state vector as well.
                    self.simulators[i].sim.x[6:9] = corrected_v

            # --- STEP 3: FOV (Optional) ---
            fov_errors_step = [[0.0, 0.0]] * self.params.N
            if self.params.fov:
                for i in range(1, self.params.N):
                    neighbors_i = self.neighbors_dict[i]
                    if len(neighbors_i) > 0:
                        _, ah, av, q_new = self.fov_controllers[i].compute_orientation_input(
                            self.x[i],
                            self.x[neighbors_i[0]],
                            self.q[i]
                        )
                        self.q[i] = q_new
                        fov_errors_step[i] = [np.degrees(ah), np.degrees(av)]

            # --- STEP 4: DATA LOGGING ---
            self.params.history = np.vstack([self.params.history, self.x[np.newaxis, :, :]])
            self.params.vel_history.append([np.linalg.norm(vi) for vi in self.v])
            self.params.v_history_full.append(self.v.copy())
            accel_step = np.linalg.norm((self.v - old_v) / self.params.dt, axis=1)
            self.params.accel_history.append(list(accel_step))
            self.params.err_history.append(self._compute_formation_errors())
            self.params.tau_history.append([np.linalg.norm(ti) for ti in u_wrenches])
            if self.params.fov:
                self.params.fov_history.append(fov_errors_step)
                self.params.q_history.append(self.q.copy())
            
            progress_step = max(1, self.params.steps // 10)
            # Print meand error and each agent error every 10% of the simulation steps
            if (step + 1) % progress_step == 0 or step == self.params.steps - 1:
                mean_err = np.mean(self.params.err_history[-1][1:]) 
                print(f"Step {step+1}/{self.params.steps} - Mean Formation Error: {mean_err:.2f}%")
                for i in range(1, self.params.N):
                    print(f"  Agent {i}: Error = {self.params.err_history[-1][i]:.2f}%")
                print("-"*50)
                      
        # --- STEP 5: NUMPY CONVERSION ---
        self.params.vel_history = np.array(self.params.vel_history)
        self.params.v_history_full = np.array(self.params.v_history_full)
        self.params.accel_history = np.array(self.params.accel_history)
        self.params.err_history = np.array(self.params.err_history)
        self.params.tau_history = np.array(self.params.tau_history)
        if self.params.fov:
            self.params.fov_history = np.array(self.params.fov_history)
            self.params.q_history = np.array(self.params.q_history)

        self.u_history = u_history
        return {"finished": True}

    def _compute_formation_errors(self) -> np.ndarray:
        errors = np.zeros(self.params.N)
        for i in range(1, self.params.N):
            target_pos = self.x[0] + self.params.relative_pos_desired[i, 0]
            dist_err = np.linalg.norm(self.x[i] - target_pos)
            target_dist = self.params.target_dist[i] if hasattr(self.params, 'target_dist') else np.linalg.norm(self.params.relative_pos_desired[i, 0])
            if target_dist > 1e-6:
                errors[i] = (dist_err / target_dist) * 100.0
            else:
                errors[i] = 0.0
        return errors

def main():
    params = Parameters.function_arguments()
    sim = FormationControlSimulation(params)
    sim.run()
    from plot_animator import Animation
    animator = Animation(params)
    animator.replay_animation(params, sim.u_history)

if __name__ == "__main__":
    main()