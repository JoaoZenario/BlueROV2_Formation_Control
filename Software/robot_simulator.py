#!/usr/bin/env python3

##########################################################
#                                                        #
# Robot Simulator                                        #
# Code Author: João Zenário                              #
#                                                        #     
##########################################################

"""
This module defines RobotSimulator, which wraps SimulationSystem for multi-robot
scenarios. Each robot has its own simulator instance that manages its dynamics
and state.
"""

import numpy as np
from systems import SimulationSystem, RK4Integrator, EulerIntegrator


class RobotSimulator:
    """
    Wraps a SimulationSystem for a single robot in a multi-robot formation.
    
    This class provides a clean interface for:
    - Integrating robot dynamics with a given input
    - Accessing the current state
    - Resetting the robot
    
    Attributes:
        robot_id (int): Unique identifier for this robot (0 to N-1)
        system (SystemBase): The dynamics system (e.g., SingleIntegrator, DoubleIntegrator)
        sim (SimulationSystem): The wrapped simulator with integrator and state management
    """
    
    def __init__(
        self,
        robot_id: int,
        system,
        x0: np.ndarray,
        dt: float,
        integrator=None,
        u_min: np.ndarray | None = None,
        u_max: np.ndarray | None = None,
        u_norm_max: float | None = None,
    ):
        """
        Initialize a RobotSimulator for a single robot.
        
        Args:
            robot_id: Unique identifier (0-indexed)
            system: System dynamics object (from systems.py)
            x0: Initial state vector
            dt: Integration time step
            integrator: ODE integrator (default: RK4Integrator)
            u_min: Minimum input constraints (optional)
            u_max: Maximum input constraints (optional)
            u_norm_max: Maximum input norm constraint (optional)
        """
        self.robot_id = robot_id
        self.system = system
        
        # Use RK4 by default if no integrator provided
        if integrator is None:
            integrator = RK4Integrator()
        
        # Wrap in SimulationSystem for state management and integration
        self.sim = SimulationSystem(
            system=system,
            x0=x0,
            dt=dt,
            integrator=integrator,
            u_min=u_min,
            u_max=u_max,
            u_norm_max=u_norm_max,
        )
    
    @property
    def state(self) -> np.ndarray:
        """Get the current state of the robot."""
        return self.sim.x.copy()
    
    @property
    def time(self) -> float:
        """Get the current simulation time."""
        return self.sim.t
    
    @property
    def state_dim(self) -> int:
        """Get the state dimension."""
        return self.system.state_dim
    
    @property
    def input_dim(self) -> int:
        """Get the input dimension."""
        return self.system.input_dim
    
    def integrate_dynamics(self, u: np.ndarray) -> np.ndarray:
        """
        Integrate the robot dynamics for one time step with input u.
        
        This is the main method for the simulation loop. It computes:
            x_{k+1} = integrate(x_k, u_k, dt)
        
        Args:
            u: Input vector (dimension must match system.input_dim)
        
        Returns:
            The new state x after integration
        """
        self.sim.step(u)
        return self.state
    
    def reset(self, x0: np.ndarray, t0: float = 0.0) -> None:
        """
        Reset the robot to an initial state at time t0.
        
        Args:
            x0: Initial state vector
            t0: Initial time (default: 0.0)
        """
        self.sim.reset(x0, t0)
    
    def __repr__(self) -> str:
        """String representation of the robot simulator."""
        return (
            f"RobotSimulator(id={self.robot_id}, state_dim={self.state_dim}, "
            f"input_dim={self.input_dim}, time={self.time:.3f}s)"
        )
