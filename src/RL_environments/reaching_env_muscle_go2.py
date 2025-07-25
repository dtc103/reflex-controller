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

@configclass
class CommandsCfg:
    position_command = ReachPositionCommandCfg(
        asset_name="robot",
        debug_vis=True,
        resampling_time_range=(10.0, 10.0)
    )

@configclass
class ActionsCfg:
    muscle_activation = MuscleActionCfg(asset_name="robot", joint_names=[".*"])

@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_position = ObsTerm(func=mdp.joint_pos, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_velocity = ObsTerm(func=mdp.joint_vel, noise=Unoise(n_min=-1.0, n_max=1.0))
        actions = ObsTerm(func=mdp.last_action)
        position_command = ObsTerm(
            func=mdp.generated_commands, 
            params={"command_name": "position_command"}
        )

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
            "position_range": (-0.5, 0.5),
            "velocity_range": (-0.5, 0.5)
        }
    )

@configclass
class RewardsCfg:
    goal_position_reach = RewTerm(
        func=reach_position_reward,
        params={
            "command_name": "position_command",
            "std": 0.1
        },
        weight=1.0
    )

    action_reg = RewTerm(
        func=action_regularization_reward,
        weight=0.5
    )

@configclass
class TerminationsCfg:
    time_out = DoneTerm(
        func=mdp.time_out, 
        time_out=True
    )

@configclass
class ReachingMuscleGo2Cfg(ManagerBasedRLEnvCfg):
    scene: UnitreeSceneCfg = UnitreeSceneCfg(num_envs=4096, env_spacing=4.0)

    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        self.decimation = 2
        self.episode_length_s = 20
        
        self.sim.dt = 1/500
        self.sim.render_interval = self.decimation

        self.scene.robot.spawn.articulation_props.fix_root_link = True
        self.scene.robot.init_state.pos = (0.0, 0.0, 1.0)