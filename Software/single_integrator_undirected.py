#!/usr/bin/env python3

##########################################################
#                                                        #
# Single Integrator Formation Control - Undirected Graph #
# Code Author: João Zenário                              #
# Equations and concepts based on: Esteban's thesis      #
#                                                        #     
##########################################################

import numpy as np
import matplotlib.pyplot as plt

def simulate():

    dt = 0.01 # Size of simulation steps.
    steps = 1000 # Number of simulation steps.
    c = 1.0  # Control gain.

    #* Initial Random Positions.
    # 3 agents in 2D space (x, y).
    x = np.array([[0.0, 0.0], [2.0, 5.0], [5.0, 1.0]]) 
    
    # Number of agents in the system.
    N = x.shape[0]

    #* Ofset Formation Relative to Centroid (in this case (2.33,2.0)).
    # Agent 0 at (0, 2), Agent 1 at (-1.5, -1), Agent 2 at (1.5, -1) relative to the formation centroid.
    o = np.array([[0.0, 2.0], [-1.5, -1.0], [1.5, -1.0]])

    desired_matrix = np.zeros((3, 3, 2)) # Store desired relative positions between agents.

    for i in range(N):
        for j in range(N):
            desired_matrix[i, j] = o[i] - o[j]

    #* Adjacency Matrix.
    # This represents who can "see" whom. Since this is an undirected system, every agent can communicate with the others.
    # A = [a_ij], where a_ij = 1 if agent i and j are connected, else 0
    A = np.array([[0, 1, 1],
                  [1, 0, 1],
                  [1, 1, 0]])

    #* Degree Matrix.
    # D is a diagonal matrix where each entry is the sum of each matrix A row (d_ii = ∑_j a_ij).
    D = np.diag(np.sum(A, axis=1))

    #* Laplacian Matrix.
    # The Laplacian matrix is the difference between the Degree matrix and the Adjacency matrix.
    L = D - A

    history = [x.copy()]

    #* Control Loop.
    # To compute the velocity we use the following equation:
    # ̇ẋ_i = u_i = -c * ∑_j [(x_i - x_j) - (o_i - o_j)] for all j connected to i.
    for _ in range(steps):

        u = np.zeros_like(x) # u will store each agent's velocity.

        for i in range(N):
            agent_i_velocity = np.array([0.0, 0.0])
            
            for j in range(N):
                if A[i, j] == 1: # If agents i and j are connected.
                    relative_pos_real = x[i] - x[j] # Computes the real relative position.
                    relative_pos_desired = desired_matrix[i, j] # Desired relative position.
                    agent_i_velocity -= (relative_pos_real - relative_pos_desired) # Add the contribution of neighbor j to agent i's velocity
            
            u[i] = c * agent_i_velocity

        x += u * dt # Update positions using Euler's Method (x_i+1 = x_i + u * dt).
        history.append(x.copy())

    #* Visualization
    history = np.array(history)
    for i in range(3):
        plt.plot(history[:, i, 0], history[:, i, 1], label=f'Agent {i}')
        plt.scatter(history[-1, i, 0], history[-1, i, 1]) # Final position
    
    plt.legend()
    plt.title("Single Integrator Formation Control (Undirected)")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    simulate()