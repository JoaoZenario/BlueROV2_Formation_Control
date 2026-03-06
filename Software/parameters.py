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
            conn -> Conectivity Maintenance constraint (Off (0); On (1)).
            coll -> Collision Avoidance constraint (Off (0); On (1)).
            fov -> Field of View constraint (Off (0); On (1)).
            mode_dynamics -> Dynamics type, single-integrator dynamics (0) or double-integrator dynamics (1).
            dt -> Size of simulation steps.
            steps -> Number of simulation steps.
            c1 -> Control gain for Lyapunov's barrier function regarding Connectivity Maintenance and Collision Avoidance.
            c2 -> Control gain for control input (u).
            c3 -> Control gain for Lyapunov's barrier function regarding Field of View.
            wn -> Filters' natural frequency.
            v_max -> Maximum velocity a robot can reach.
            a_max -> Maximum acceleration a robot can reach.
            d_max -> Threshold for Connectivity Maintenance.
            d_min -> Threshold for Collision Avoidance.
            r_max -> Maximum rotation a robot can reach.
            R -> Formation radius (distance from leader).
            l -> Square side length (space where agents are spawned).
            kp_yaw -> Proportional gain for rotation.
            delta_alpha -> Field of View limit (range sideways).
    """
    def __init__(self,
                 N = 5,
                 dim = 3,
                 mode_graph = 1,
                 conn = 1,
                 coll = 1,
                 fov = 1,
                 mode_dynamics = 1,
                 dt = 0.01,
                 steps = 1000,
                 c1 = 1.0,
                 c2 = 1.0,
                 c3 = 0.5,
                 wn = 10.0,
                 v_max = 1.0,
                 a_max = 1.0,
                 d_max = 5.0,
                 d_min = 0.5,
                 r_max = 1.5,
                 R = 2.0,
                 l = 5.0,
                 kp_yaw = 2.0,
                 delta_alpha = np.radians(45)):

        self.N = N
        self.dim = dim
        self.mode_graph = mode_graph
        self.conn = conn
        self.coll = coll
        self.fov = fov
        self.mode_dynamics = mode_dynamics
        self.dt = dt
        self.steps = steps
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.wn = wn
        self.v_max = v_max
        self.a_max = a_max
        self.d_max = d_max
        self.d_min = d_min
        self.r_max = r_max
        self.R = R
        self.l = l
        self.counter = 0
        self.kp_yaw = kp_yaw
        self.delta_alpha = delta_alpha

        self.theta_max_x = np.radians(45) # 45 degrees left/right
        self.theta_max_y = np.radians(30) # 30 degrees up/down
        self.c3_h = 0.5 # Horizontal barrier gain
        self.c3_v = 0.5 # Vertical barrier gain
        self.kp_pitch = 1.0
        self.damping = 3.0


        self.x = np.array([
            [3.39655962, 0.29968507, 4.17778228],
            [3.30965058, 1.82937878, 0.09709306],
            [4.46324661, 3.24636054, 2.73822641],
            [0.13316123, 1.37122818, 4.99218486],
            [0.04021556, 4.23317497, 2.23731517]
        ])

        self.A = np.array([
            [0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [0, 1, 1, 1, 0]
        ])

        self.E = np.empty((0, 2), dtype=np.int64) 

        self.NE = 0 

        self.dist = np.array([
            [0.        , 4.35884624, 3.4486285 , 3.53004607, 5.52293408],
            [4.35884624, 0.        , 3.21157384, 5.85336743, 4.58780908],
            [3.4486285 , 3.21157384, 0.        , 5.22934886, 4.55937698],
            [3.53004607, 5.85336743, 5.22934886, 0.        , 3.97349787],
            [5.52293408, 4.58780908, 4.55937698, 3.97349787, 0.        ]
        ])

        # Generate random graph and compute distance between agents and all usefull information.
        #self.x, self.A, self.E, self.NE, self.dist = generate_random_connected_graph(self.N, self.dim, self.d_max, self.d_min, self.l, self.mode_graph)
        #plot_graph_from_adjacency(self.x, self.A, self.mode_graph)

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
        self.err_history = [] # Store history of agent's formation error over time.
        self.fov_history = [] # Store history of agent's FOV over time.
        self.vel_history = [] # Store history of agent's velocity over time.
        self.history_np = np.empty((0, self.N, self.dim))
        self.vel_hist_np = np.empty((0, self.N))
        self.accel_hist_np = np.empty((0, self.N))
        self.err_hist_np = np.empty((0, self.N))
        self.fov_hist_np = np.empty((0, self.N))
        self.q = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (self.N, 1)) # Create an (N, 4) array to store [x, y, z, w] for each of the N agents

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

        self.x_lf = np.zeros((self.N, self.dim))
        self.ẋ_lf = np.zeros((self.N, self.dim))

    @staticmethod
    def function_arguments():
        """Creates a Parameters instance using command line arguments."""
        parser = argparse.ArgumentParser(description="Formation Control Simulation Parameters")
        
        # Add arguments matching your __init__ parameters
        parser.add_argument('--N', type=int, default=5, help='Number of agents')
        parser.add_argument('--dim', type=int, default=3, help='Dimension (2D or 3D)')
        parser.add_argument('--graph', type=int, default=1, help='Graph type (0: Undirected, 1: Directed)')
        parser.add_argument('--conn', type=int, default=1, help='Conectivity Maintenance (Off (0); On (1))')
        parser.add_argument('--coll', type=int, default=1, help='Collision Avoidance (Off (0); On (1))')
        parser.add_argument('--fov', type=int, default=1, help='Field of View (Off (0); On (1))')
        parser.add_argument('--dynamics', type=int, default=1, help='Dynamics (0: Single, 1: Double)')
        parser.add_argument('--dt', type=float, default=0.01, help='Size of simulation steps')
        parser.add_argument('--steps', type=int, default=1000, help='Number of simulation steps')
        parser.add_argument('--c1', type=float, default=1.0, help='Barrier function gain')
        parser.add_argument('--c2', type=float, default=1.0, help='Control gain')
        parser.add_argument('--c3', type=float, default=0.5, help='FOV barrier function gain')
        parser.add_argument('--wn', type=float, default=10.0, help='Natural frequency')
        parser.add_argument('--v_max', type=float, default=1.0, help='Maximum velocity')
        parser.add_argument('--a_max', type=float, default=1.0, help='Maximum acceleration')
        parser.add_argument('--d_max', type=float, default=5.0, help='Connectivity Maintenance threshold')
        parser.add_argument('--d_min', type=float, default=0.5, help='Collision Avoidance threshold')
        parser.add_argument('--r_max', type=float, default=1.5, help='Maximum rotation rate')       
        parser.add_argument('--R', type=float, default=2.0, help='Formation radius')
        parser.add_argument('--l', type=float, default=5.0, help='Square side length') 
        parser.add_argument('--kp_yaw', type=float, default=5.0, help='Rotation gain') 
        parser.add_argument('--delta_alpha', type=float, default=np.radians(45), help='FOV range')  

        # Parse the arguments from sys.argv
        args = parser.parse_args()
        
        # Return a NEW instance of Parameters using the parsed values
        return Parameters(
            N=args.N,
            dim=args.dim,
            mode_graph=args.graph,
            conn=args.conn,
            coll=args.coll,
            fov=args.fov,
            mode_dynamics=args.dynamics,
            dt=args.dt,
            steps=args.steps,
            c1=args.c1,
            c2=args.c2,
            c3=args.c3,
            wn=args.wn,
            v_max=args.v_max,
            a_max=args.a_max,
            d_max=args.d_max,
            d_min=args.d_min,
            r_max=args.r_max,
            R=args.R,
            l=args.l,
            kp_yaw=args.kp_yaw,
            delta_alpha=args.delta_alpha
        )
