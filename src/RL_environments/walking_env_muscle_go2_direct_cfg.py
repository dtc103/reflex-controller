from __future__ import annotations

from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import EventTermCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.sensors import ContactSensorCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.sim import SimulationCfg
from isaaclab.scene import InteractiveSceneCfg

from modules.robot_config.unitree_muscle_cfg import UNITREE_GO2_MUSCLE_CFG

import isaaclab.envs.mdp as mdp

@configclass
class EventCfg:
    reset_joints = EventTermCfg(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.5, 0.5),
            "velocity_range": (-0.5, 0.5)
        }
    )

    reset_base = EventTermCfg(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z":(-0.3, 0.3), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.1, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )

    body_perturbations = EventTermCfg(
        func=mdp.randomize_rigid_body_mass,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-5.0, 5.0),
            "operation": "add"
        },
        mode="startup"
    )

    physics_material = EventTermCfg(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.7, 0.8),
            "dynamic_friction_range": (0.5, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    base_com = EventTermCfg(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "com_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "z": (-0.02, 0.02)},
        },
    )

    base_external_force_torque = EventTermCfg(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    push_robot = EventTermCfg(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 15.0),
        params={"velocity_range": {"x": (-1.0, 1.0), "y": (-2.0, 2.0)}},
    )

@configclass
class WalkingMuscleGo2DirectCfg(DirectRLEnvCfg):
    decimation = 4
    episode_length_s = 20
    action_space = 24
    observation_space = 94 #58, when using joint_pos and joint_vel
    state_space = 0

    action_scale = 0.5
    action_offset = 0.5

    sim_dt = 1/500

    sim: SimulationCfg = SimulationCfg(dt=sim_dt, render_interval=decimation)
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=2.0, replicate_physics=True)
    events: EventCfg = EventCfg()

    robot = ArticulationCfg = UNITREE_GO2_MUSCLE_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*", history_length=3, track_air_time=True
    )

    camera_follow = False
    print_actions = False
    deactivate_perturbations = False

    # reward parameters
    target_vel = 1.5
    vel_std = 0.8
    vel_scale = 16.0

    action_reg_scale_1 = 0.8
    action_reg_scale_2 = 0.3

    angular_reg_scale = 10.0
    z_vel_scale = 2.5

    # curriculum parameters
    survival_scale = 15
    survival_ep_length = 150

    forward_push_scale = 10
    forward_push_ep_length = 150
    forward_push_threshold = 0.1

    def __post_init__(self):
        if self.deactivate_perturbations:
            self.events.base_com = None
            self.events.base_external_force_torque = None
            self.events.body_perturbations = None
            self.events.physics_material = None
            self.events.push_robot = None
            self.events.reset_joints = None

@configclass
class WalkingMuscleGo2DirectCfg_PLAY(WalkingMuscleGo2DirectCfg):
    deactivate_perturbations = False
    camera_follow = True
    print_actions=True

    episode_length_s = 5

    def __post_init__(self):
        self.events.push_robot.interval_range_s = (2.0, 5.0)

        if self.deactivate_perturbations:
            self.events.base_com = None
            self.events.base_external_force_torque = None
            self.events.body_perturbations = None
            self.events.physics_material = None
            self.events.push_robot = None
            self.events.reset_joints = None



