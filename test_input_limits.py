#!/usr/bin/env python3
"""
Test script to verify whether input limits are applied in different simulation modes.
"""

import sys

sys.path.insert(0, "/home/nicola/Code/BlueROV2_Formation_Control/Software")

import numpy as np
from blue_rov2_refactored import BlueROV2
from robot_simulator import RobotSimulator
from systems import RK4Integrator


def test_input_constraints():
    """Check if input constraints are applied in different modes."""

    print("=" * 80)
    print("Testing Input Constraint Application for BlueROV2")
    print("=" * 80)

    # Test 1: Check if BlueROV2 system has input constraint attributes
    print("\n1. Checking BlueROV2 System Object:")
    print("-" * 80)

    rov = BlueROV2()
    print(f"  BlueROV2 type: {type(rov).__name__}")
    print(f"  Has 'u_min' attribute: {hasattr(rov, 'u_min')}")
    print(f"  Has 'u_max' attribute: {hasattr(rov, 'u_max')}")
    print(f"  Has 'u_norm_max' attribute: {hasattr(rov, 'u_norm_max')}")
    print(f"  Has 'state_dim' attribute: {hasattr(rov, 'state_dim')}")
    print(f"  Has 'input_dim' attribute: {hasattr(rov, 'input_dim')}")

    # Try to get values
    if hasattr(rov, "u_min"):
        print(f"    u_min = {rov.u_min}")
    else:
        print(f"    u_min = NOT DEFINED")

    if hasattr(rov, "u_max"):
        print(f"    u_max = {rov.u_max}")
    else:
        print(f"    u_max = NOT DEFINED")

    # Test 2: Check RobotSimulator with BlueROV2 (without passing constraints)
    print("\n2. RobotSimulator with BlueROV2 (no constraints passed):")
    print("-" * 80)

    x0 = np.zeros(13)
    x0[0:3] = [0, 0, -2]  # Initial position
    x0[3:7] = [0, 0, 0, 1]  # Initial quaternion (identity)

    try:
        sim = RobotSimulator(robot_id=0, system=rov, x0=x0, dt=0.02)
        print(f"  ✓ RobotSimulator created successfully")
        print(f"    u_min passed to RobotSimulator: {sim.u_min}")
        print(f"    u_max passed to RobotSimulator: {sim.u_max}")
        print(f"    u_norm_max passed to RobotSimulator: {sim.u_norm_max}")
    except Exception as e:
        print(f"  ✗ Error creating RobotSimulator: {e}")
        import traceback

        traceback.print_exc()
        return

    # Test 3: Test integration with unconstrained torque
    print("\n3. Testing Integration with Unconstrained Torque:")
    print("-" * 80)

    # Create a very large torque command
    tau_very_large = np.array([1e6, 1e6, 1e6, 1e6, 1e6, 1e6])

    print(f"  Input torque (unreasonably large): {tau_very_large}")
    print(f"  Input torque norm: {np.linalg.norm(tau_very_large):.2e}")

    try:
        new_state = sim.step(tau_very_large)
        print(f"  ✓ Integration completed without error")
        print(f"    This means NO input constraints were applied!")
        print(f"    (If constraints existed, torque would have been clamped)")
    except Exception as e:
        print(f"  ✗ Error during integration: {e}")

    # Test 4: Check the underlying SimulationSystem
    print("\n4. Checking Underlying SimulationSystem:")
    print("-" * 80)
    print(f"  sim.sim.u_min: {sim.sim.u_min}")
    print(f"  sim.sim.u_max: {sim.sim.u_max}")
    print(f"  sim.sim.u_norm_max: {sim.sim.u_norm_max}")

    print("\n" + "=" * 80)
    print("CONCLUSION:")
    print("=" * 80)
    print("If sim.sim.u_min/u_max are None and we successfully integrated with a")
    print("massive torque command without error, then input limits are NOT applied")
    print("to the BlueROV2 model in feedback_linearization and clf_qp_tau modes.")
    print("=" * 80)


if __name__ == "__main__":
    test_input_constraints()
