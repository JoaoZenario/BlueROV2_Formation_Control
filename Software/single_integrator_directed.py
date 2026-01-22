#!/usr/bin/env python3

##########################################################
#                                                        #
# Single Integrator Formation Control - Directed Graph   #
# Code Author: João Zenário                              #
# Equations and concepts based on: Esteban's thesis      #
#                                                        #     
##########################################################

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import random

class BROV_Formation:

    #* Function to Initialize Variables.
    def __init__(self):
        self.dt = 0.01 # Size of simulation steps.
        self.steps = 1000 # Number of simulation steps.
        self.counter = 0 # Variable to count the steps (iterations) throughout time.
        self.c = 1.0  # Control gain.
        self.N = 10 # Number of agents in the system. Change to desired value.
        self.x = np.zeros((self.N, 3)) # Initial random positions, N agents in 3D space (x, y, z).
        self.x[0] = np.random.uniform(1, 100, size=3) # Define leader's pose.
        # Distance matrix. Since this is an directed graph and the agents are only connected to the leader, the distance from the agents to the leader is the only one known.
        self.d = np.zeros_like(self.x)

        # Fillinf matrix d with random distances from agents to leader.
        for i in range(1, self.N):
            self.d[i, 0] = random.uniform(1, 100)

        # Filling matrix x with random coordinates accordingly with the distance matrix.
        for i in range (1, self.N):
            # Compute the coordinates in order to fullfill the requisite: d_i = √((x_i - x_0)² + (y_i - y_0)² + (z_i - z_0)²) -> Sphere with radius d_i and center x_0 (Leader's pose).
            phi = random.uniform(0, 2 * np.pi)
            costheta = random.uniform(-1, 1)
            theta = np.arccos(costheta)
            # Unit vector on sphere
            x = np.sin(theta) * np.cos(phi)
            y = np.sin(theta) * np.sin(phi)
            z = np.cos(theta)
            
            # Final Position = Leader + (Distance * Direction)
            self.x[i] = self.x[0] + self.d[i, 0] * np.array([x, y, z])

        # Colors to distinguish each agent.
        self.colors = [
            '#{:02X}{:02X}{:02X}'.format(
                int(r * 255), int(g * 255), int(b * 255)
            )
            for r, g, b, _ in plt.cm.hsv([i / self.N for i in range(self.N)])
        ]
        
        self.history = [self.x.copy()] # Store history of agent's pose over time for the trails.

        R = 30.0  # Distance from leader. Change to desired value.
        # Offset formation relative to Leader.
        self.o = np.zeros((self.N, 3))

        # Agent 0 is the leader at (0,0,0)
        # Agents 1 to N form a shape around the leader, examples:
        # N = 4: Triangle with the leader in the center.
        # N = 5: Square with the leader in the center.
        # N = 6: Pentagon with the leader in the center.
        # N = 11:  Decagon with the leader in the center.
        # N -> +∞ Shape -> Circle with the leader in the center.
        for i in range(1, self.N):
            angle = (2 * np.pi / (self.N - 1)) * (i - 1)
            self.o[i, 0] = R * np.cos(angle) # X offset
            self.o[i, 1] = R * np.sin(angle) # Y offset
            self.o[i, 2] = 0.0               # Z offset (same depth)

        self.relative_pos_desired = np.zeros((self.N, self.N, 3)) # Stores desired relative positions between agents.

        for i in range(self.N):
            for j in range(3):
                self.relative_pos_desired[i, j] = self.o[i] - self.o[j] # Fills the array with the desired relative positions.
            if i:
                self.target_dist = np.linalg.norm(self.relative_pos_desired[i, 0]) # Calculate the target distance (from desired offsets).

        # Adjacency Matrix.
        # This represents who can "see" whom. Since this is a sirected system, the leader doesn't communicate with anyone and the others only comunicate with the leader.
        # A = [a_ij], where a_ij = 1 if agent i and j are connected, else 0.
        self.A = np.zeros((self.N, self.N))

        for i in range(1, self.N):
            self.A[i, 0] = 1

        # Degree Matrix.
        # D is a diagonal matrix where each entry is the sum of each matrix A row (d_ii = ∑_j a_ij).
        #self.D = np.diag(np.sum(self.A, axis=1))

        # Laplacian Matrix.
        # The Laplacian matrix is the difference between the Degree matrix and the Adjacency matrix (L = D - A, from equation 2.15).
        #self.L = self.D - self.A

    #* Function to compute each agent's velocity in the exact moment.
    # The agents are in an initial pose (with a certain relative pose to each others)
    # and the goal is the agents moving to a pose where the relative pose to each others is the desired one (relative_pose_desired).   
    def consensus_control(self):

        t = self.steps * self.dt
        self.counter += 1
        # To compute the velocity we'll use equation 2.11 and 2.13 (with a slight change):
        # ̇ẋ_i = u_i = -c * ∑_j [(x_i - x_j) - (o_i - o_j)] for all j connected to i.
        u = np.zeros_like(self.x) # u will store each agent's velocity.

        for i in range(self.N):
            if i == 0:
                if (self.counter < 0.5 * self.steps):
                    u[i, 0] = 5  # Leader's x speed.
                    u[i, 1] = 20 # Leader's y speed.
                    u[i, 2] = 5  # Leader's z speed.
                else:
                    u[i, 0] = -10.0 # Leader's x speed.
                    u[i, 1] = -10.0 # Leader's y speed.
                    u[i, 2] = 5.0   # Leader's z speed.

            else:
                agent_i_velocity = np.array([0.0, 0.0, 0.0]) # Initialize velocity vector.
                #j = 0 # Since the leader is the only one taken into account.
                # Generalized.
                for j in range(self.N):
                    if self.A[i, j] == 1:
                        relative_pos_real = self.x[i] - self.x[j] # Computes the real relative pose.
                        agent_i_velocity -= (relative_pos_real - self.relative_pos_desired[i, j]) # Add the contribution of neighbor j to agent i's velocity.
                        u_position = self.c * agent_i_velocity # Velocity needed for agent i to reach desired position.
                        u_leader = u[j].copy() # Store leader's velocity vector.
                # It's crucial to add leader's velocity to the velocity that agent i would initially need to achieve the desired pose,
                # because leader is in movement and that must be taken into account.
                u[i] = u_position + u_leader # Agent's i velocity.
        return u # Used in the update_step function.

    #* Function to update the pose of each agent after the step (moving at u velocity for dt time).
    def update_step(self, frame):
        u = self.consensus_control()
        self.x += u * self.dt # Update positions using Euler's Method (x_i+1 = x_i + u * dt).
        self.history.append(self.x.copy()) # Add agent's latest pose to history.
        history_np = np.array(self.history)

        current_time = frame * self.dt
        stats = f"Time: {current_time:5.2f}s\n"
        stats += "-" * 15 + "\n"
        
        for i in range(self.N):
            # Calculate speed (magnitude of velocity vector).
            speed = np.linalg.norm(u[i]) # Compute the speed using the Euclidean norm: |v| = √(u_x² + u_y² + u_z²).  
            stats += f"ROV {i+1} Vel: {speed:5.2f} m/s\n"

            if i: # Skip if i=0.
                dist_to_leader = np.linalg.norm(self.x[i] - self.x[0]) # Calculate actual distance to leader using the same method (Euclidean distance): d = √[(u_x_i - u_x_0)² + (u_y_i - u_y_0)² + (u_z_i - u_z_0)²].
                stats += f"Distance to Leader: {dist_to_leader:5.2f}m (Target: {self.target_dist:5.2f}m)\n"
        
        self.telemetry_text.set_text(stats) # Store the statistics for the animation.

        for i in range(self.N):
            # Update the dots on the plot.
            self.dots[i].set_data([self.x[i, 0]], [self.x[i, 1]]) # For x and y.
            self.dots[i].set_3d_properties([self.x[i, 2]]) # For z.
            # Stores all time steps of each agent until this moment.
            self.trails[i].set_data(history_np[:, i, 0], history_np[:, i, 1]) # For x and y.
            self.trails[i].set_3d_properties(history_np[:, i, 2]) # For z.

        return self.dots + self.trails + [self.telemetry_text] # For efficiency, returning these variables is crucial since they were the only ones that changed.

    #* Function to run and plot the animation that shows the robot's movement and formation process throughout time.
    def plot_animation(self):
        # Animation setup.
        self.fig = plt.figure(figsize=(9, 9))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.dots = [self.ax.plot([], [], 'o', color=self.colors[i], markersize=5, label=f"ROV {i+1}")[0] for i in range(self.N)] # Create 3 dots, one for each agent.
        self.trails = [self.ax.plot([], [], '-', color=self.colors[i], alpha=0.2)[0] for i in range(self.N)] # Create the path trail for each agent.
        self.ax.set_xlim(-200, 200) # Plot limits based on the agent's x pose.
        self.ax.set_ylim(-200, 200) # Plot limits based on the agent's y pose.
        self.ax.set_zlim(-200, 200) # Plot limits based on the agent's z pose.
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_zlabel('Z (m)')
        self.ax.grid(True)
        self.ax.set_title("ROV Formation Evolution Over Time")
        self.ax.legend(loc='upper right', shadow=True, fontsize='medium')
        # Create a Text Box to show the time and agent's velocity.
        self.telemetry_text = self.fig.text(0.02, 0.95, '', 
                                    verticalalignment='top', 
                                    family='monospace',
                                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
        self.animation = FuncAnimation(self.fig, self.update_step, frames=self.steps, interval=10, blit=False, repeat = False) # Run animation.
        #self.animation.save('rov_formation.gif', writer='pillow', fps=30) # Save animation.
        plt.show() # Plot animation.

if __name__ == "__main__":
    sim = BROV_Formation()
    sim.plot_animation()