#!/usr/bin/env python3

##########################################################
#                                                        #
# Formation Simulation                                   #
# Code Author: João Zenário                              #
#                                                        #     
##########################################################

from formation_controller import BROV_Formation 
from parameters import Parameters
from plot_animator import Animation
import numpy as np

class Simulation:
    #* Function to setup all classes in order to run the simulation.
    def __init__(self, params):
        # 1. Initialize all parameters and variables.
        self.params = params
        
        # 2. Setup the controller.
        self.controller = BROV_Formation()

        # 3. Setup the animation.
        self.animator = Animation(self.params)

    #* Function to run the animation.
    def run(self):
        # Start the Matplotlib event loop.
        self.animator.plot_animation(self.params, self.controller)
        #return self.animator.plot_animation(self.params, self.controller) # Uncomment when running on jupyter notebook.

#* Main.
def main():
    params = Parameters.function_arguments() # Set parameters from terminal.

    print("\n" + "="*50)
    print("      BLUE ROV2 FORMATION CONTROL SIMULATION")
    print("="*50)
    dashboard = (
        f"--- CONFIGURATION ---\n"
        f"Agents (N):        {params.N}\n"
        f"Dimension:         {params.dim}D\n"
        f"Dynamics:          {'Double' if params.mode_dynamics else 'Single'} Integrator\n"
        f"Graph Type:        {'Directed' if params.mode_graph else 'Undirected'}\n"
        f"Connectivity Maintenance:       {['Off', 'On'][params.conn]}\n"
        f"Collision Avoidance:       {['Off', 'On'][params.coll]}\n"
        f"Field of View:       {['Off', 'On'][params.fov]}\n"
        f"Recentered Barrier:       {['Off', 'On'][params.rec]}\n"
        f"\n--- PHYSICS & CONTROL ---\n"
        f"Time Step (dt):    {params.dt}s ({params.steps} steps)\n"
        f"Gains:             c1={params.c1}, c2={params.c2}, c3={params.c3}, c4_h={params.c4_h}, c4_v={params.c4_v}, ωn={params.wn if params.mode_dynamics else 'N/A'}, kp_yaw={params.kp_yaw}, kp_pitch={params.kp_pitch}\n"
        f"Limits:            v_max={params.v_max}m/s, a_max={params.a_max}m/s², r_max={params.r_max}rad/s\n"
        f"\n--- GEOMETRY ---\n"
        f"Thresholds:        d_max={params.d_max}m, d_min={params.d_min}m, theta_max_x={np.degrees(params.theta_max_x)}°, theta_max_y={np.degrees(params.theta_max_y)}°\n"
        f"Formation:         Radius={params.R}m, Spawn Area={params.l}m²"
    )
    print(dashboard)
    print("="*50 + "\n")

    sim = Simulation(params) # Create the simulation object.
    sim.run() # Run it.

if __name__ == "__main__":
    main()