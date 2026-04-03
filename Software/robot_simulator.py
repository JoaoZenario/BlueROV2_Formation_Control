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
    Thin wrapper around SimulationSystem for a single robot in a multi-robot formation.

    This class provides a multi-robot-specific interface while delegating all dynamics,
    integration, and constraint handling to SimulationSystem. It closely mirrors the
    SimulationSystem API while adding robot identification.

    Key features aligned with SimulationSystem:
    - Input constraint application (u_min, u_max, u_norm_max)
    - Process noise handling
    - ODE integration with RK4 or custom integrator
    - State and time management

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

    # ===== State and Dimensions =====

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

    # ===== Input Constraints (aligned with SimulationSystem) =====

    @property
    def u_min(self) -> np.ndarray | None:
        """Get the minimum input constraints vector (if defined)."""
        return self.sim.u_min.copy() if self.sim.u_min is not None else None

    @property
    def u_max(self) -> np.ndarray | None:
        """Get the maximum input constraints vector (if defined)."""
        return self.sim.u_max.copy() if self.sim.u_max is not None else None

    @property
    def u_norm_max(self) -> float | None:
        """Get the maximum input norm constraint (if defined)."""
        return self.sim.u_norm_max

    def apply_input_constraints(self, u: np.ndarray) -> np.ndarray:
        """
        Apply input constraints to a command vector.

        Clamps input within [u_min, u_max] and enforces norm limit if specified.
        Aligned with SimulationSystem.apply_input_constraints().

        Args:
            u: Input vector

        Returns:
            Constrained input vector
        """
        return self.sim.apply_input_constraints(u)

    # ===== Process Noise (aligned with SimulationSystem) =====

    def add_process_noise(self, x: np.ndarray) -> np.ndarray:
        """
        Optionally add process noise to state.

        Aligned with SimulationSystem.add_process_noise().

        Args:
            x: State vector

        Returns:
            State with noise added (if noise function defined)
        """
        return self.sim.add_process_noise(x)

    # ===== Integration Methods =====

    def step(self, u: np.ndarray) -> np.ndarray:
        """
        Integrate the robot dynamics for one time step with input u.

        Primary integration method (aligned with SimulationSystem.step()).
        Computes: x_{k+1} = integrate(x_k, u_k, dt)
        Applies input constraints and process noise before integration.

        Args:
            u: Input vector (dimension must match system.input_dim)

        Returns:
            The new state x after integration
        """
        self.sim.step(u)
        return self.state

    def integrate_dynamics(self, u: np.ndarray) -> np.ndarray:
        """
        Integrate the robot dynamics for one time step with input u.

        Legacy method name for backward compatibility.
        Delegates to step() for consistency with SimulationSystem.

        Args:
            u: Input vector (dimension must match system.input_dim)

        Returns:
            The new state x after integration
        """
        return self.step(u)

    # ===== State Management =====

    def reset(self, x0: np.ndarray, t0: float = 0.0) -> None:
        """
        Reset the robot to an initial state at time t0.

        Aligned with SimulationSystem.reset().

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
