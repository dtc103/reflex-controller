import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg
from isaaclab.managers import ObservationTermCfg
from isaaclab.managers import RewardTermCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import EventTermCfg
from isaaclab.managers import TerminationTermCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.sensors import ContactSensorCfg

from modules.robot_config.unitree_muscle_cfg import UNITREE_GO2_MUSCLE_CFG

from modules.mdp.actions import MuscleActionCfg, CameraActionCfg
from modules.mdp.rewards import *
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
    class PolicyCfg(ObservationGroupCfg):
        joint_position = ObservationTermCfg(func=mdp.joint_pos, noise=Unoise(n_min=-0.2, n_max=0.2))
        joint_velocity = ObservationTermCfg(func=mdp.joint_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        base_lin_vel = ObservationTermCfg(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.5, n_max=0.5))
        actions = ObservationTermCfg(func=mdp.last_action)
        orientation = ObservationTermCfg(func=mdp.root_quat_w, noise=Unoise(n_min=0.1, n_max=0.1))

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()

@configclass
class EventCfg:
    reset_joints = EventTermCfg(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-1.0, 1.0),
            "velocity_range": (-5.0, 5.0)
        }
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

    # base_external_force_torque = EventTermCfg(
    #     func=mdp.apply_external_force_torque,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names="base"),
    #         "force_range": (0.0, 30.0),
    #         "torque_range": (-0.0, 0.0),
    #     },
    # )

@configclass
class RewardsCfg:
    hopping = RewardTermCfg(
        func=hop_reward,
        weight=1e0
    )

    action_reg = RewardTermCfg(
        func=mdp.action_rate_l2,
        weight=-5e-1
    )

    # root_orientation_termination = RewardTermCfg(
    #     func=mdp.is_terminated_term,
    #     params={
    #         "term_keys": "root_orientation_deviation",
    #     },
    #     weight=-1.5
    # )

    # illegal_contact_termination = RewardTermCfg(
    #     func=mdp.is_terminated_term,
    #     params={
    #         "term_keys": "illegal_contacts",
    #     },
    #     weight=-1e-1
    # )


@configclass
class TerminationsCfg:
    time_out = TerminationTermCfg(
        func=mdp.time_out, 
        time_out=True
    )

    # root_height_termination = TerminationTermCfg(
    #     func = terminations.root_height_below_minimum_after_s,
    #     params={
    #         "seconds": 0.0,
    #         "minimum_height": 0.15
    #     }
    # )

    illegal_contacts = TerminationTermCfg(
        func = mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=['base', 'FL_calf', 'FR_calf', 'RL_calf', 'RR_calf']),
            "threshold": 1.0
        }
    )

    root_orientation_deviation = TerminationTermCfg(
        func = mdp.bad_orientation,
        params={
            "limit_angle": torch.pi / 4
        }
    )


@configclass
class HoppingMuscleGo2Cfg(ManagerBasedRLEnvCfg):
    scene: UnitreeSceneCfg = UnitreeSceneCfg(num_envs=4096, env_spacing=2)

    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        self.decimation = 5
        self.episode_length_s = 10

        self.sim.dt = self.scene.robot.actuators["base_legs"].muscle_params["dt"]
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