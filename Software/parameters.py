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

from graphs_utils import generate_random_connected_graph
from graphs_utils import plot_graph_from_adjacency
   
class Parameters:
    #* Function to initialize all variables and parameters.
    def __init__(self):
        # New added code for random graph generation and distance between agents.
        self.N = random.randint(2, 20) # Number of agents in the system. Change to desired value.
        self.dim = 3 # 3D. It's possible to switch between 2D and 3D.
        self.d_max = 40  # Connection radius.
        self.d_min = 0.0  # Minimum distance requirement.
        self.l = 100 # Square side length (unit square).
        self.mode = 1 # Parameter to specifiy which graph is desired, directed (1) or undirected (0).
        self.x, self.A, self.E, self.NE, self.dist = generate_random_connected_graph(self.N, self.dim, self.d_max, self.d_min, self.l, self.mode)
        plot_graph_from_adjacency(self.x, self.A, self.mode)

        self.dt = 0.01 # Size of simulation steps.
        self.steps = 1000 # Number of simulation steps.
        self.counter = 0 # Variable to count the steps (iterations) throughout time.
        self.c1 = 1.0  # Control gain.
        self.c2 = 1.0  # Control gain.
        self.v_max = 50 # Maximum velocity a robot can reach.

        # Colors to distinguish each agent.
        self.colors = [
            '#{:02X}{:02X}{:02X}'.format(
                int(r * 255), int(g * 255), int(b * 255)
            )
            for r, g, b, _ in plt.cm.hsv([i / self.N for i in range(self.N)])
        ]

        self.history = [self.x.copy()] # Store history of agent's pose over time for the trails.

        R = 20  # Distance from leader. Change to desired value.
        # Offset formation relative to Leader.
        self.o = np.zeros((self.N, self.dim))

        # Agent 0 is the leader at (0,0,0)
        # Agents 1 to N form a shape around the leader, examples:
        # N = 4: Triangle with the leader in the center.
        # N = 5: Square with the leader in the center.
        # N = 6: Pentagon with the leader in the center.
        # N = 11:  Decagon with the leader in the center.
        # N -> +∞ Shape -> Circle with the leader in the center.
        for i in range(1, self.N):
            angle = (2 * np.pi / (self.N - 1)) * (i - 1)
            if self.dim == 2:
                self.o[i, 0] = R * np.sin(angle) # X offset.
                self.o[i, 1] = 0.0               # Y offset (same Y).
            if self.dim == 3:
                self.o[i, 0] = R * np.cos(angle) # X offset.
                self.o[i, 1] = R * np.sin(angle) # Y offset.
                self.o[i, 2] = 0.0               # Z offset (same depth).

        self.relative_pos_desired = np.zeros((self.N, self.N, self.dim)) # Stores desired relative positions between agents.

        for i in range(self.N):
            for j in range(self.N):
                self.relative_pos_desired[i, j] = self.o[i] - self.o[j] # Fills the array with the desired relative positions.
            if i:
                self.target_dist = np.linalg.norm(self.relative_pos_desired[i, 0]) # Calculate the target distance (from desired offsets).