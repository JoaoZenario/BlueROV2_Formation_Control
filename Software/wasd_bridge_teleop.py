#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import sys, select, termios, tty
import numpy as np

# Change message types to PX4 Native
from px4_msgs.msg import VehicleThrustSetpoint, VehicleTorqueSetpoint

class MultiAgentTeleop(Node):
    def __init__(self):
        super().__init__('multi_agent_teleop')
        
        self.agents = ['itrl_rov_1']  
        self.current_idx = 0
        self.thrust_pubs = {}
        self.torque_pubs = {}

        # Initialize PX4-specific publishers
        for ns in self.agents:
            thrust_topic = f"/{ns}/fmu/in/vehicle_thrust_setpoint"
            torque_topic = f"/{ns}/fmu/in/vehicle_torque_setpoint"
            self.get_logger().info(f"Setting up PX4 publishers for: {ns}")
            
            self.thrust_pubs[ns] = self.create_publisher(VehicleThrustSetpoint, thrust_topic, 10)
            self.torque_pubs[ns] = self.create_publisher(VehicleTorqueSetpoint, torque_topic, 10)

        # Settings (Normalized -1.0 to 1.0)
        self.speed = 0.3  # Max 1.0
        self.turn = 0.1   # Max 1.0

        self.settings = termios.tcgetattr(sys.stdin)
        self.create_timer(0.05, self.run_loop) # 20Hz is better for PX4
        self.print_status()

    def print_status(self):
        sys.stdout.write("\033[H\033[J") 
        print(f"--- ACTIVE AGENT (PX4 NATIVE): {self.agents[self.current_idx].upper()} ---")
        print("w/s : Forward/Back")
        print("a/d : Left/Right")
        print("r/f : Up/Down (Z is Down in NED!)")
        print("q/e : Yaw Left/Right")

    def get_key(self):
        tty.setraw(sys.stdin.fileno())
        rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
        key = sys.stdin.read(1) if rlist else ''
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
        return key

    def run_loop(self):
        key = self.get_key()
        
        # Initialize zero setpoints
        thrust = VehicleThrustSetpoint()
        torque = VehicleTorqueSetpoint()
        timestamp = int(self.get_clock().now().nanoseconds / 1000)
        thrust.timestamp = timestamp
        torque.timestamp = timestamp
        drift_comp = 0.0075  # No drift compensation in this simple teleop

        # Map keys to NED (North-East-Down)
        # Note: In PX4 NED, x=forward, y=right, z=down
        if key == 'w': thrust.xyz = [self.speed, -drift_comp, 0.0]
        elif key == 's': thrust.xyz = [-self.speed, drift_comp, 0.0]
        elif key == 'a': thrust.xyz = [0.0, -0.5*self.speed, 0.0] # Left
        elif key == 'd': thrust.xyz = [0.0, 0.5*self.speed, 0.0]  # Right
        elif key == 'r': thrust.xyz = [0.0, 0.0, -self.speed] # Up (Negative Z)
        elif key == 'f': thrust.xyz = [0.0, 0.0, self.speed]  # Down (Positive Z)
        elif key == 'q': torque.xyz = [0.0, 0.0, -self.turn] # Yaw Left
        elif key == 'e': torque.xyz = [0.0, 0.0, self.turn]  # Yaw Right
            
        elif key == '\x03': # CTRL+C
            rclpy.shutdown()
            return

        # Publish to the active agent
        active_ns = self.agents[self.current_idx]
        self.thrust_pubs[active_ns].publish(thrust)
        self.torque_pubs[active_ns].publish(torque)

def main():
    rclpy.init()
    node = MultiAgentTeleop()
    try:
        rclpy.spin(node)
    except Exception as e:
        print(e)
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, termios.tcgetattr(sys.stdin))

if __name__ == '__main__':
    main()