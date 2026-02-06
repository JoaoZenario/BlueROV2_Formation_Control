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

import argparse
import sys
   
class Parameters:
    #* Function to initialize all variables and parameters.
    """
     Arguments:
            N -> Number of agents in the system.
            dim -> Simulation dimension (2D or 3D).
            mode_graph -> Graph type, undirected (0) or directed (1).
            mode_constraints -> Desired constraints (No Constraints(0); Only Connectivity Maintenance(1); Connectivity Maintenance and Collision Avoidance(2)).
            mode_dynamics -> Dynamics type, single-integrator dynamics (0) or double-integrator dynamics (1).
            dt -> Size of simulation steps.
            steps -> Number of simulation steps.
            c1 -> Control gain for Lyapunov's barrier function.
            c2 -> Control gain for control input (u).
            gamma -> Positive gain for velocity offset regarding double integrator agents.
            v_max -> Maximum velocity a robot can reach.
            a_max -> Maximum acceleration a robot can reach.
            d_max -> Threshold for Connectivity Maintenance.
            d_min -> Threshold for Collision Avoidance.
            R -> Formation radius (distance from leader).
            l -> Square side length (space where agents are spawned).
    """
    def __init__(self,
                 N = 10,
                 dim = 3,
                 mode_graph = 1,
                 mode_constraint = 2,
                 mode_dynamics = 1,
                 dt = 0.01,
                 steps = 1000,
                 c1 = 1.0,
                 c2 = 1.0,
                 gamma = 2.0,
                 v_max = 100,
                 a_max = 200,
                 d_max = 50.0,
                 d_min = 5.0,
                 R = 20,
                 l = 100):

        self.N = N
        self.dim = dim
        self.mode_graph = mode_graph
        self.mode_constraint = mode_constraint
        self.mode_dynamics = mode_dynamics
        self.dt = dt
        self.steps = steps
        self.c1 = c1
        self.c2 = c2
        self.gamma = gamma
        self.v_max = v_max
        self.a_max = a_max
        self.d_max = d_max
        self.d_min = d_min
        self.R = R
        self.l = l
        self.counter = 0

        # Generate random graph and compute distance between agents and all usefull information.
        self.x, self.A, self.E, self.NE, self.dist = generate_random_connected_graph(self.N, self.dim, self.d_max, self.d_min, self.l, self.mode_graph)
        plot_graph_from_adjacency(self.x, self.A, self.mode_graph)

        # Colors to distinguish each agent.
        self.colors = [
            '#{:02X}{:02X}{:02X}'.format(
                int(r * 255), int(g * 255), int(b * 255)
            )
            for r, g, b, _ in plt.cm.hsv([i / self.N for i in range(self.N)])
        ]

        self.history = [self.x.copy()] # Store history of agent's pose over time for the trails.
        self.o = np.zeros((self.N, self.dim)) # Offset formation relative to Leader.
        self.v = np.zeros((self.N, self.dim)) # Velocity of all agents.
        self.vel_history = [] # Store history of agent's velocity over time.

        if self.mode_dynamics:
            self.accel_history = [] # Store history of agent's acceleration over time.

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
                self.o[i, 0] = 0.0 # X offset.
                self.o[i, 1] = i*5 # Y offset (same Y).
            if self.dim == 3:
                self.o[i, 0] = self.R * np.cos(angle) # X offset.
                self.o[i, 1] = self.R * np.sin(angle) # Y offset.
                self.o[i, 2] = 0.0               # Z offset (same depth).

        self.relative_pos_desired = np.zeros((self.N, self.N, self.dim)) # Stores desired relative positions between agents.
        self.target_dist = np.zeros(self.N) # Stores desired distance between agents.

        for i in range(self.N):
            for j in range(self.N):
                self.relative_pos_desired[i, j] = self.o[i] - self.o[j] # Fills the array with the desired relative positions.
            if i:
                self.target_dist[i] = np.linalg.norm(self.relative_pos_desired[i, 0]) # Calculate the target distance (from desired offsets).

    @staticmethod
    def function_arguments():
        """Creates a Parameters instance using command line arguments."""
        parser = argparse.ArgumentParser(description="Formation Control Simulation Parameters")
        
        # Add arguments matching your __init__ parameters
        parser.add_argument('--N', type=int, default=10, help='Number of agents')
        parser.add_argument('--dim', type=int, default=3, help='Dimension (2D or 3D)')
        parser.add_argument('--graph', type=int, default=1, help='Graph type (0: Undirected, 1: Directed)')
        parser.add_argument('--constraint', type=int, default=2, help='Constraints (0: None, 1: Conn, 2: Both)')
        parser.add_argument('--dynamics', type=int, default=1, help='Dynamics (0: Single, 1: Double)')
        parser.add_argument('--dt', type=float, default=0.01, help='Size of simulation steps')
        parser.add_argument('--steps', type=int, default=1000, help='Number of simulation steps')
        parser.add_argument('--c1', type=float, default=1.0, help='Barrier function gain')
        parser.add_argument('--c2', type=float, default=1.0, help='Control gain')
        parser.add_argument('--gamma', type=float, default=2.0, help='Velocity offset gain')
        parser.add_argument('--v_max', type=int, default=100, help='Maximum velocity')
        parser.add_argument('--a_max', type=int, default=200, help='Maximum acceleration')
        parser.add_argument('--d_max', type=int, default=50, help='Connectivity Maintenance threshold')
        parser.add_argument('--d_min', type=int, default=5, help='Collision Avoidance threshold')       
        parser.add_argument('--R', type=int, default=20.0, help='Formation radius')
        parser.add_argument('--l', type=int, default=100, help='Square side length') 
        
        # Parse the arguments from sys.argv
        args = parser.parse_args()
        
        # Return a NEW instance of Parameters using the parsed values
        return Parameters(
            N=args.N,
            dim=args.dim,
            mode_graph=args.graph,
            mode_constraint=args.constraint,
            mode_dynamics=args.dynamics,
            dt=args.dt,
            steps=args.steps,
            c1=args.c1,
            c2=args.c2,
            gamma=args.gamma,
            v_max=args.v_max,
            a_max=args.a_max,
            d_max=args.d_max,
            d_min=args.d_min,
            R=args.R,
            l=args.l
        )
