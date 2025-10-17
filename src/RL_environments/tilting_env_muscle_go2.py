import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ManagerBasedRLEnv
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from modules.robot_config.unitree_muscle_cfg import UNITREE_GO2_MUSCLE_CFG

from modules.mdp.actions import MuscleActionCfg
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

# #TODO tilting commands here
# @configclass
# class CommandsCfg:
#     position_command = ReachPositionCommandCfg(
#         asset_name="robot",
#         debug_vis=True,
#         resampling_time_range=(5.0, 5.0),
#         num_goals=1
#     )

@configclass
class ActionsCfg:
    muscle_activation = MuscleActionCfg(asset_name="robot", joint_names=[".*"])

@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_position = ObsTerm(func=mdp.joint_pos, noise=Unoise(n_min=-0.1, n_max=0.1))
        joint_velocity = ObsTerm(func=mdp.joint_vel, noise=Unoise(n_min=-0.5, n_max=0.5))
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

    body_perturbations = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-5.0, 5.0),
            "operation": "add"
        },
        mode="startup"
    )

    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "force_range": (0.0, 30.0),
            "torque_range": (-0.0, 0.0),
        },
    )

@configclass
class RewardsCfg:
    height_term = RewTerm(
        func=base_height,
        weight=100.0
    )

    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)

    termination = RewTerm(
        func=mdp.is_terminated_term,
        params={
            "term_keys": "root_height_termination"
        },
        weight=-1.0
    )
 
    action_reg = RewTerm(
        func=action_regularization_reward,
        weight=0.01,
    )

@configclass
class TerminationsCfg:
    time_out = DoneTerm(
        func=mdp.time_out, 
        time_out=True
    )

    root_height_termination = DoneTerm(
        func = terminations.root_height_below_minimum_after_s,
        params={
            "seconds": 5,
            "minimum_height": 0.15
        }
    )


@configclass
class StandingMuscleGo2Cfg(ManagerBasedRLEnvCfg):
    scene: UnitreeSceneCfg = UnitreeSceneCfg(num_envs=4096, env_spacing=2)

    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    #events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        self.decimation = 5 #with 4 we got huge results
        self.episode_length_s = 20
        
        self.sim.dt = self.scene.robot.actuators["base_legs"].muscle_params["dt"]
        #print(self.scene.robot.actuators["base_legs"].muscle_params["dt"])
        self.sim.render_interval = self.decimation

        self.scene.robot.spawn.articulation_props.fix_root_link = False

        self.scene.robot.init_state.pos = (0.0, 0.0, 0.4)
        self.scene.robot.init_state.joint_pos = {
            ".*L_hip_joint": 0.3,
            ".*R_hip_joint": -0.3,
            "F[L,R]_thigh_joint": torch.pi/2.0 - 0.5,
            "R[L,R]_thigh_joint": torch.pi/2.0 - 0.5,
            ".*_calf_joint": -2.7,
        }

class TiltingMuscleGo2Cfg(ManagerBasedRLEnv):
    scene: UnitreeSceneCfg = UnitreeSceneCfg(num_envs=4096, env_spacing=2)

    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    #commands: CommandsCfg = CommandsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        self.decimation = 5
        self.episode_length_s = 20
        
        self.sim.dt = self.scene.robot.actuators["base_legs"].muscle_params["dt"]
        self.sim.render_interval = self.decimation

        self.scene.robot.spawn.articulation_props.fix_root_link = False

        self.scene.robot.init_state.pos = (0.0, 0.0, 0.40)
        self.scene.robot.init_state.joint_pos = {
            ".*L_hip_joint": 0.3,
            ".*R_hip_joint": -0.3,
            "F[L,R]_thigh_joint": torch.pi/2.0 - 0.5,
            "R[L,R]_thigh_joint": torch.pi/2.0 - 0.5,
            ".*_calf_joint": -2.7,
        }