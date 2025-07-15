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

from modules.robot_config.unitree_muscle_cfg import UNITREE_GO2_MUSCLE_CFG

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
    unitree: ArticulationCfg = UNITREE_GO2_MUSCLE_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
    )

TARGET_MARKER_CFG = FRAME_MARKER_CFG.copy()
TARGET_MARKER_CFG.markers["frame"].scale = (0.15, 0.15, 0.15)
TARGET_MARKER_CFG.prim_path = "/Visual/TargetMarker"

@configclass
class ReachRandomizationCfg:
    """Ranges for randomising the 4 target positions."""

    x_range = (-0.25, 0.25)  # forward/backward wrt hip
    y_range = (-0.25, 0.25)  # left/right
    z_range = (-0.10, 0.05)  # up/down relative to default foot height


def feet_positions(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return positions of the 4 feet in robot base frame."""
    feet_body_ids = env.feet_body_ids
    feet_pos_w = env.scene["unitree"].data.body_pos_w[:, feet_body_ids]  # (N,4,3)
    base_pos_w = env.scene["unitree"].data.root_pos_w.unsqueeze(1)
    return feet_pos_w - base_pos_w  # (N,4,3)


def feet_target_errors(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return vector from each foot to its current target marker."""
    return env.target_markers - feet_positions(env)  # (N,4,3)


def feet_l2_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Negative L2 distance to the corresponding marker."""
    return -torch.linalg.norm(feet_target_errors(env), dim=-1).sum(-1)  # (N,)


def randomise_target_markers(env: ManagerBasedRLEnv, env_ids: torch.Tensor | None):
    """Randomise the 4 marker positions at the start of each episode."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    cfg: ReachRandomizationCfg = env.cfg.rand_cfg

    # sample deltas
    dx = torch.empty(len(env_ids), 4, 1, device=env.device).uniform_(*cfg.x_range)
    dy = torch.empty(len(env_ids), 4, 1, device=env.device).uniform_(*cfg.y_range)
    dz = torch.empty(len(env_ids), 4, 1, device=env.device).uniform_(*cfg.z_range)

    # default foot positions in base frame (tuned for Go2)
    default_feet_pos = torch.tensor(
        [[0.19, 0.13, -0.30], [0.19, -0.13, -0.30], [-0.19, 0.13, -0.30], [-0.19, -0.13, -0.30]],
        device=env.device,
    ).expand(len(env_ids), -1, -1)

    env.target_markers[env_ids] = default_feet_pos + torch.cat([dx, dy, dz], dim=-1)
    env.marker_visual.visualize(env.target_markers.view(-1, 3), indices=env_ids)


@configclass
class UnitreeReachCfg(ManagerBasedRLEnvCfg):
    """Configuration for the Go2 foot-reach task."""

    # Scene
    scene: UnitreeSceneCfg = UnitreeSceneCfg(num_envs="${num_envs}", env_spacing=2.0)
    # Randomisation
    rand_cfg: ReachRandomizationCfg = ReachRandomizationCfg()

    # Observations
    observations: ObsGroup = ObsGroup(
        policy=ObsGroup(
            joint_pos=ObsTerm(func="joint_pos", scale=1.0),
            joint_vel=ObsTerm(func="joint_vel", scale=0.1),
            feet_pos=ObsTerm(func=feet_positions, scale=1.0),
            feet_err=ObsTerm(func=feet_target_errors, scale=1.0),
        )
    )

    # Rewards
    rewards: dict[str, RewTerm] = {
        "feet_l2": RewTerm(func=feet_l2_reward, weight=10.0),
    }

    # Terminations
    terminations: dict[str, DoneTerm] = {
        "time_out": DoneTerm(func="time_out", time_out=True),
    }

    # Events
    events: dict[str, EventTerm] = {
        "randomise_targets": EventTerm(
            func=randomise_target_markers,
            mode="reset",
        ),
    }

    # Misc
    episode_length_s: float = 5.0
    decimation: int = 4
    num_actions: int = 12  # 12 actuated joints



# -------------------------------------------------------------
# Concrete environment class
# -------------------------------------------------------------
class UnitreeReachEnv(ManagerBasedRLEnv):
    """Manager-based RL environment for Go2 foot reaching."""

    cfg: UnitreeReachCfg

    def __init__(self, cfg: UnitreeReachCfg, **kwargs):
        super().__init__(cfg, **kwargs)

        # cache foot body indices
        self.feet_body_ids, _ = self.scene["unitree"].find_bodies(
            [".*_foot"]  # Go2 feet are named FR_foot, FL_foot, RR_foot, RL_foot
        )
        self.feet_body_ids = torch.tensor(self.feet_body_ids, device=self.device)

        # create marker visualiser
        self.marker_visual = VisualizationMarkers(TARGET_MARKER_CFG)
        self.target_markers = torch.zeros(self.num_envs, 4, 3, device=self.device)

        # trigger initial randomisation
        randomise_target_markers(self, None)

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)
        # trigger marker randomisation (registered as reset event)

    def _apply_action(self) -> None:
        actions = self.action_manager.action
        self._pos_targets[:] = actions[:, :12]
        self._vel_target[:] = actions[:, 12:]

        robot = self.scene["unitree"]
        robot.set_joint_position_target