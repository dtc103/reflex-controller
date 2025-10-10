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
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from modules.robot_config.unitree_muscle_cfg import UNITREE_GO2_MUSCLE_CFG

from modules.mdp.actions import MuscleActionCfg
from modules.mdp.commands import ReachPositionCommandCfg
from modules.mdp.rewards import *
import modules.mdp.observations as obs

import isaaclab.envs.mdp as mdp

body_parts = ["FL_foot"]

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
        resampling_time_range=(5.0, 5.0),
        body_names=body_parts,
        goal_tolerance=0.05
    )

@configclass
class ActionsCfg:
    muscle_activation = MuscleActionCfg(
        asset_name="robot", 
        joint_names=[".*"],
        offset=0.5,
        scale=0.5
        )

@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_position = ObsTerm(func=mdp.joint_pos, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_velocity = ObsTerm(func=mdp.joint_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        actions = ObsTerm(func=mdp.last_action)
        #Idk, if this observation improves it or not
        feet_pos = ObsTerm(
            func=obs.body_pos,
            params={
                "body_names": body_parts
            }
        )
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

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "mass_distribution_params": (0.0, 2.0),
            "operation": "add"
        }
    )

    # external_force_torque = EventTerm(
    #     func=mdp.apply_external_force_torque,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot"),
    #         "force_range": (0.0, 0.0),
    #         "torque_range": (-1.0, 1.0)
    #     }
    # )

@configclass
class RewardsCfg:
    goal_position_reach_dense = RewTerm(
        func=reach_position_reward_l2,
        params={
            "command_name": "position_command",
            "std": 0.3,
            "body_parts": body_parts,
        },
        weight=5.0
    )

    goal_position_reach_sparse = RewTerm(
        func=reach_position_reward_goal_sparse,
        params={
            "command_name": "position_command",
            "body_parts": body_parts,
            "goal_tolerance": 0.05
        }, 
        weight=10.0
    )

    # action_reg = RewTerm(
    #     func=action_regularization_reward,
    #     weight=0.05,
    # )

@configclass
class TerminationsCfg:
    time_out = DoneTerm(
        func=mdp.time_out, 
        time_out=True
    )

@configclass
class ReachingMuscleGo2Cfg(ManagerBasedRLEnvCfg):
    scene: UnitreeSceneCfg = UnitreeSceneCfg(num_envs=4096, env_spacing=2)

    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        self.decimation = 5
        self.episode_length_s = 10
        
        self.sim.dt = self.scene.robot.actuators["base_legs"].muscle_params["dt"]
        self.sim.render_interval = self.decimation

        self.scene.robot.spawn.articulation_props.fix_root_link = True
        self.scene.robot.init_state.pos = (0.0, 0.0, 1.0)