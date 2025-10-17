from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers.command_manager import CommandTerm

def root_height_below_minimum_after_s(
    env: ManagerBasedRLEnv, minimum_height: float, seconds: int, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), 
) -> torch.Tensor:
    """Terminate when the asset's root height is below the minimum height for a given time

    Note:
        This is currently only supported for flat terrains, i.e. the minimum height is in the world frame.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.logical_and(env.episode_length_buf * env.step_dt > torch.tensor(seconds).float(), asset.data.root_pos_w[:, 2] < minimum_height)