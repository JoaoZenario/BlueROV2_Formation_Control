from __future__ import annotations

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


FORMATION = [
    {
        "id": 1,
        "name": "itrl_rov_1",
        "neighbors": [0],
        "desired_rel": [(0.0, 1.0, 0.0)],
        "fov_target_id": 0,
    },
    {
        "id": 2,
        "name": "glub",
        "neighbors": [0],
        "desired_rel": [(0.0, -1.0, 0.0)],
        "fov_target_id": 0,
    }
]

def _csv(values) -> str:
    return ",".join(str(v) for v in values)

def _unzip_desired(desired_rel):
    if not desired_rel:
        return "", "", ""
    return (
        _csv([d[0] for d in desired_rel]),
        _csv([d[1] for d in desired_rel]),
        _csv([d[2] for d in desired_rel]),
    )


def _spawn_nodes(context, *args, **kwargs):
    dt = float(LaunchConfiguration("dt").perform(context))
    use_fov_str = LaunchConfiguration("use_fov").perform(context).lower()
    use_fov = use_fov_str == "true"
    use_pool = LaunchConfiguration("use_pool_barrier").perform(context).lower() == "true"

    nodes = []

    for agent in FORMATION:
        aid = agent["id"]
        robot_name = agent["name"]
        ns = robot_name
        rel_x_str, rel_y_str, rel_z_str = _unzip_desired(agent["desired_rel"])
        nbr_str = _csv(agent["neighbors"])
        fov_target_id = agent["fov_target_id"]
        agent_has_fov = use_fov and (fov_target_id >= 0)

        nodes.append(Node(
            package="bluerov2_control",
            executable="formation_controller_node_mocap",
            namespace=ns,
            name="controller",
            output="screen",
            parameters=[{
                # Identity & graph
                "agent_id":        aid,
                "neighbors":       nbr_str,
                "desired_rel_x":   rel_x_str,
                "desired_rel_y":   rel_y_str,
                "desired_rel_z":   rel_z_str,
                "robot_name":      robot_name,
                "robot_names": ["Splash", "itrl_rov_1", "glub"],

                # Controller
                "mode":            "clf_qp_tau",
                "dt":              dt,
                "d_min":           0.6,
                "d_max":           4.0,
                "use_connectivity": True,
                "use_collision":    True,
                "use_recentered":   True,
                "k_form":           20.0,
                "clf_alpha":        0.05,
                "k_att":            4.0,
                "k_att_d":          2.0,

                # Normalisation
                "Fx_max_N":   88.0,
                "Fy_max_N":   88.0,
                "Fz_max_N":  137.0,
                "Mx_max_Nm":  30.0,
                "My_max_Nm":  16.5,
                "Mz_max_Nm":  21.0,
                "thrust_sat": 0.1,
                "torque_sat": 0.1,

                # FoV
                # fov_target_id: ID of the agent whose current position is
                # used as the FoV target.  -1 = no FoV.
                "use_fov":         agent_has_fov,
                "use_fov_torque":  agent_has_fov,
                "fov_target_id":   fov_target_id,

                # Camera frustum half-angles
                "theta_h_max_deg": 45.0,
                "theta_v_max_deg": 30.0,

                # FoV barrier weights
                "mu_fov_pos":      0.01,
                "mu_fov_att":      0.1,
                "k_fov_att":       2.0,
                "k_fov_omega":     2.0,
                "k_fov_backstep":  3.0,
                "omega_fov_max":   1.0,
                "alpha_h_des":     0.0,
                "alpha_v_des":     0.0,
                "fov_eps":         1.0,
                "use_adaptive_fov_barrier": False,
                "lambda_sigma_fov_h": 2.0,
                "gamma_sigma_fov_h":  1.0,
                "sigma_fov_h_max":    0.0,
                "lambda_sigma_fov_v": 2.0,
                "gamma_sigma_fov_v":  1.0,
                "sigma_fov_v_max":    0.0,
                # Minimum rx in camera frame to activate FoV barrier.
                # Larger = more conservative (FoV activates later).
                # Too small → OSQP blow-up; too large → FoV barely activates.
                "fov_rx_min": 0.1,

                # Pool boundary barrier
                "use_pool_barrier": False,
                "pool_x_min": -1.5,   "pool_x_max": 7.7,
                "pool_y_min": -1.5,  "pool_y_max": 1.9,
                "pool_z_min": 2.4,  "pool_z_max": 0.18,
                "pool_margin": 0.8,
                "pool_h_max":  0.5,
                "pool_k":      2.0,
            }],
        ))

        nodes.append(Node(
            package="bluerov2_control",
            executable="offboard_heartbeat_wrench",
            namespace=ns,
            name="heartbeat",
            output="screen",
            parameters=[{"topic": f"/{ns}/fmu/in/offboard_control_mode"}],
        ))

    return nodes


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription([
        DeclareLaunchArgument("dt",               default_value="0.02"),
        DeclareLaunchArgument("use_fov",          default_value="True",
            description="Enable FoV barrier for followers"),
        DeclareLaunchArgument("use_pool_barrier", default_value="True",
            description="Enable pool boundary barrier"),
        OpaqueFunction(function=_spawn_nodes),
    ])
