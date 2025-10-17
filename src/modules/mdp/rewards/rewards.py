from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import RewardTermCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def reach_position_reward_l2(
    env: ManagerBasedRLEnv, 
    command_name: str, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    std: float = 0.1,
    body_parts: list[str] = ["FL_foot"],
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]

    foot_idxs, _ = asset.find_bodies([body_part + "_foot" for body_part in body_parts], preserve_order=True)
    foot_positions = asset.data.body_pos_w[:, foot_idxs, :]

    goal_positions = env.command_manager.get_command(command_name).view(env.num_envs, len(foot_idxs), 3) # has shape (env, n_feet * 3) -> need to be adapted to (env, n_feet, 3)
    
    # squared euclidian norm as distance
    distances = torch.sum((goal_positions - foot_positions) ** 2, dim=-1)
    total_error = distances.sum(dim=1)

    reward = torch.exp(-total_error / (std ** 2))

    return reward

def reach_position_reward_goal_sparse(
    env: ManagerBasedRLEnv, 
    command_name: str, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    body_parts: list[str] = ["FL_foot"],
    goal_tolerance = 0.025
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]

    foot_idxs, _ = asset.find_bodies([body_part + "_foot" for body_part in body_parts], preserve_order=True)
    foot_positions = asset.data.body_pos_w[:, foot_idxs, :]

    goal_positions = env.command_manager.get_command(command_name).view(env.num_envs, len(foot_idxs), 3) # has shape (env, n_feet * 3) -> need to be adapted to (env, n_feet, 3)
    
    distances = torch.sqrt(torch.sum((goal_positions - foot_positions) ** 2, dim=-1))

    reached_goal = (distances < goal_tolerance).float()

    reward = torch.sum(reached_goal, dim=-1)
    
    return reward

def lin_vel_x(env: ManagerBasedRLEnv, target_vel: float, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward x-axis base linear velocity """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    return torch.exp(-torch.square(asset.data.root_lin_vel_b[:, 0] - target_vel) / std**2)

def hop_reward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]

    return torch.exp(10 * torch.clamp(asset.data.root_lin_vel_b[:, 2], min=0, max=10.0))

def action_regularization_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    joint_names: list[str] = [".*"]
) -> torch.Tensor:
    '''
    This action regularization is taken from 
    "Reinforcement learning-based motion imitation for physiologically plausible  musculoskeletal motor control" [Simos et al. 2025]
    '''
    l1 = torch.norm(env.action_manager.action, p=1, dim=1)
    l2 = torch.norm(env.action_manager.action, p=2, dim=1)

    reward = -l1 - l2

    return reward

def base_height(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward higher body position
    """
    asset: Articulation = env.scene[asset_cfg.name]

    return torch.square(asset.data.root_pos_w[:, 2])

def capped_base_height(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward higher body position
    """
    asset: Articulation = env.scene[asset_cfg.name]

    return torch.clamp(torch.exp(asset.data.root_pos_w[:, 2]), max=0.3)


def negative_action_penalty(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    return torch.exp((env.action_manager.action < 0.0).float() - 1.0, dim=-1)
