#!/usr/bin/env python3

##########################################################
#                                                        #
# Formation Controller                                   #
# Code Author: João Zenário                              #
# Equations and concepts based on: Esteban's thesis      #
#                                                        #     
##########################################################

import numpy as np
from single_integrator import Single_integrator
from double_integrator import Double_integrator
from constraints import Constraints
import mpl_toolkits.mplot3d.art3d as art3d
from scipy.spatial.transform import Rotation as R

class BROV_Formation(Single_integrator, Double_integrator, Constraints):
    def __init__(self):
        Single_integrator.__init__(self)
        Double_integrator.__init__(self)
        Constraints.__init__(self)

    #* Function to update the pose of each agent after the step (moving at u acceleration for dt time).
    def update_step(self, frame, params, animator):
        t = params.steps * params.dt
        params.counter += 1
        u = np.zeros_like(params.x) # u will store each agent's acceleration.
        current_fov_errors = [0.0] * params.N

        if params.mode_dynamics:
            u = self.leader_acceleration(params, u) # Leader acceleration.
            u = self.compute_acceleration(params, u) # Every other agent acceleration.

            params.v += u * params.dt # Update velocities (v̇ = u <=> v = u * dt).
            params.x += params.v * params.dt # Update positions (ẋ = v <=> x = v * dt) .
        else:
            u = self.leader_velocity(params, u) # Leader velocity.
            u = self.compute_velocity(params, u) # Every other agent velocity.

            params.x += u * params.dt # Update positions (ẋ = u <=> x = u * dt).
        
        if params.fov:
            current_fov_errors = self.field_of_view(params) # Apply Field-of-View constraint.

        self.pose_history(params)
        self.velocity_history(params, u)
        self.acceleration_history(params, u)
        self.formation_error_history(params)
        self.fov_error_history(params, current_fov_errors)
        animator.plot_handle(frame, params)

        return animator.dots + animator.trails + animator.dist_lines + animator.err_lines + [animator.telemetry_text] # For efficiency, returning these variables is crucial since they were the only ones that changed.

    #* Function to update the quaternion orientation for agent i (omega: [p, q, r] angular velocity vector in body frame)
    def update_quaternion(self, i, omega, dt, params):
        # Get current quaternion [x, y, z, w].
        q_current = params.q[i]
        
        # Convert angular velocity to a quaternion rate of change.
        rotation_vec = omega * dt
        angle = np.linalg.norm(rotation_vec)
        
        if angle > 1e-8:
            delta_rot = R.from_rotvec(rotation_vec) # Create a rotation object from the rotation vector (exponential map).
            q_obj_current = R.from_quat(q_current) # Current orientation as a Scipy Rotation object.
            q_new_obj = q_obj_current * delta_rot # Apply the rotation: q_new = q_current * delta_q.
            params.q[i] = q_new_obj.as_quat() # Extract back to array and ensure normalization.
        
        # Normalize to prevent numerical drift.
        params.q[i] /= np.linalg.norm(params.q[i])

    #* Function to store all agent's pose history.
    def pose_history(self, params):
        # Add agent's latest pose to history.
        params.history.append(params.x.copy())
        params.history_np = np.array(params.history)
        pass

    #* Function to store all agent's velocity history.
    def velocity_history(self, params, u):
        # Store velocity of each agent over time for the animation.
        if params.mode_dynamics:
            current_speeds = [np.linalg.norm(v) for v in params.v]
        else:
            current_speeds = [np.linalg.norm(v) for v in u]

        params.vel_history.append(current_speeds)
        params.vel_hist_np = np.array(params.vel_history)
        pass

    #* Function to store all agent's acceleration history.
    def acceleration_history(self, params, u):
        # Store acceleration of each agent over time for the animation.
        current_accels = [np.linalg.norm(a) for a in u]
        params.accel_history.append(current_accels)
        params.accel_hist_np = np.array(params.accel_history)
        pass

    #* Function to store all agent's formation error history.
    def formation_error_history(self, params):
        current_errors = []
        # Formation distance error.
        for i in range(params.N):
            if i == 0:
                current_errors.append(0.0)
            else:
                # Target = Leader Position + Desired Relative Offset from Leader.
                target_pos = params.x[0] + params.relative_pos_desired[i, 0]
                dist_err = np.linalg.norm(params.x[i] - target_pos)
                # Normalize by formation radius R.
                percent_err = (dist_err / params.R) * 100
                current_errors.append(percent_err)


        # Store formation distance error of each agent over time for the animation.
        params.err_history.append(current_errors)
        params.err_hist_np = np.array(params.err_history)
        pass

    #* Function to store all agent's field of view error history.
    def fov_error_history(self, params, current_fov_errors):
        # Store FOV error for each agent over time for the simulation.
        params.fov_history.append(current_fov_errors)
        params.fov_hist_np = np.array(params.fov_history) 
        pass