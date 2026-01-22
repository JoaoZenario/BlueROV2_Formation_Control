#!/usr/bin/env python3

##########################################################
#                                                        #
# Parameters to the Formation Controller                 #
# Code Author: João Zenário                              #
# Equations and concepts based on: Esteban's thesis      #
#                                                        #     
##########################################################

import numpy as np
import random 
import matplotlib.pyplot as plt
   
class Parameters:
    #* Function to initialize all variables and parameters.
    def __init__(self):
        self.dt = 0.01 # Size of simulation steps.
        self.steps = 1000 # Number of simulation steps.
        self.counter = 0 # Variable to count the steps (iterations) throughout time.
        self.c = 1.0  # Control gain.
        self.N = random.randint(2, 20) # Number of agents in the system. Change to desired value.
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
            for j in range(self.N):
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