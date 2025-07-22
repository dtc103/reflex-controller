from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import RewardTermCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def reach_position_reward(
    env: ManagerBasedRLEnv, 
    command_name: str, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    std: float = 0.1,
    body_parts: list[str] = [".*_foot"]
) -> torch.Tensor:    
    asset: Articulation = env.scene[asset_cfg.name]

    foot_idxs, _ = asset.find_bodies(body_parts)
    foot_positions = asset.data.body_pos_w[:, foot_idxs, :]

    goal_positions = env.command_manager.get_command(command_name) # has shape (env, 12) -> need to be adapted
    
    # squared euclidian norm
    distances = torch.sum((goal_positions.view(env.num_envs, len(foot_idxs), 3) - foot_positions) ** 2, dim=-1)

    total_error = distances.sum(dim=1)

    reward = torch.exp(-total_error / (std ** 2))
    return reward

def action_regularization_reward(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    '''
    This action regularization is taken from 
    "Reinforcement learning-based motion imitation for physiologically plausible  musculoskeletal motor control" [Simos et al. 2025]
    '''
    l1 = torch.norm(env.action_manager.action, p=1, dim=1)
    l2 = torch.norm(env.action_manager.action, p=2, dim=1)

    reward = -l1 - l2

    return reward


