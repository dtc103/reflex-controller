import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ManagerBasedRLEnv
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import TerminationTermCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.sensors import ContactSensorCfg

from modules.robot_config.unitree_muscle_cfg import UNITREE_GO2_MUSCLE_CFG

from modules.mdp.actions import MuscleActionCfg, CameraActionCfg
from modules.mdp.commands import ReachPositionCommandCfg
from modules.mdp.rewards import *
import modules.mdp.observations as obs
import modules.mdp.terminations as terminations

import isaaclab.envs.mdp as mdp
import torch


@configclass
class UnitreeSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # articulation
    robot: ArticulationCfg = UNITREE_GO2_MUSCLE_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
    )

    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=False)

@configclass
class ActionsCfg:
    muscle_activation = MuscleActionCfg(asset_name="robot", joint_names=[".*"])
    #camera_follow = CameraActionCfg(asset_name="robot")

@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_position = ObsTerm(func=mdp.joint_pos, noise=Unoise(n_min=-0.2, n_max=0.2))
        joint_velocity = ObsTerm(func=mdp.joint_vel, noise=Unoise(n_min=-1.0, n_max=1.0))
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.5, n_max=0.5))
        actions = ObsTerm(func=mdp.last_action)
        base_pose = ObsTerm(func=obs.base_pose)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()

@configclass
class EventCfg:
    reset_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-1.0, 1.0),
            "velocity_range": (-5.0, 5.0)
        }
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z":(-0.1, 0.2), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.1, 0.1),
                "y": (-0.1, 0.1),
                "z": (-0.1, 0.1),
                "roll": (-0.1, 0.1),
                "pitch": (-0.1, 0.1),
                "yaw": (-0.1, 0.1),
            },
        },
    )

    body_perturbations = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-5.0, 5.0),
            "operation": "add"
        },
        mode="startup"
    )

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.4, 0.8),
            "dynamic_friction_range": (0.3, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.01, 0.01)},
        },
    )

    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "force_range": (0.0, 50.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )


@configclass
class RewardsCfg:
    linear_velocity_x = RewTerm(
        func=lin_vel_x,
        params={
            "target_vel": 2.0,
            "std":0.8
        },
        weight=10.0
    ) 

    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)

    alive = RewTerm(
        func=mdp.is_alive,
        weight=1.0
    )

    # termination = RewTerm(
    #     func=mdp.is_terminated_term,
    #     params={
    #         "term_keys": "root_height_termination"
    #     },
    #     weight=-1.5
    # )
 
    action_reg = RewTerm(
        func=action_regularization_reward,
        weight=0.05,
    )

    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)

    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base", ".*thigh", ".*calf"]), "threshold": 1.0},
    )


@configclass
class TerminationsCfg:
    time_out = TerminationTermCfg(
        func=mdp.time_out, 
        time_out=True
    )

    illegal_contacts = TerminationTermCfg(
        func = mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=['base', 'FL_calf', 'FR_calf', 'RL_calf', 'RR_calf']),
            "threshold": 1.0
        }
    )

    base_height = TerminationTermCfg(
        func=mdp.terminations.root_height_below_minimum,
        params={
            "minimum_height" : 0.2
        }
    )


@configclass
class WalkingMuscleGo2Cfg(ManagerBasedRLEnvCfg):
    scene: UnitreeSceneCfg = UnitreeSceneCfg(num_envs=4096, env_spacing=2)

    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    #events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        self.decimation = 4
        self.episode_length_s = 10

        
        
        self.sim.dt = self.scene.robot.actuators["base_legs"].muscle_params["dt"]
        #print(self.scene.robot.actuators["base_legs"].muscle_params["dt"])
        self.sim.render_interval = self.decimation

        self.scene.robot.spawn.articulation_props.fix_root_link = False

        self.scene.robot.init_state.pos = (0.0, 0.0, 0.5)
        self.scene.robot.init_state.joint_pos = {
            ".*L_hip_joint": 0.1,
            ".*R_hip_joint": -0.1,
            "F[L,R]_thigh_joint": 0.8,
            "R[L,R]_thigh_joint": 1.0,
            ".*_calf_joint": -1.5,
        }

"""
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.4)
        self.scene.robot.init_state.joint_pos = {
            ".*L_hip_joint": 0.3,
            ".*R_hip_joint": -0.3,
            "F[L,R]_thigh_joint": torch.pi/2.0 - 0.5,
            "R[L,R]_thigh_joint": torch.pi/2.0 - 0.5,
            ".*_calf_joint": -2.7,
        }
"""
"""
    pos=(0.0, 0.0, 0.6),
        joint_pos={
            ".*L_hip_joint": 0.1,
            ".*R_hip_joint": -0.1,
            "F[L,R]_thigh_joint": 0.8,
            "R[L,R]_thigh_joint": 1.0,
            ".*_calf_joint": -1.5,
        },
        joint_vel={".*": 0.0},
"""