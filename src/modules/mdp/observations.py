from __future__ import annotations
import torch
from typing import TYPE_CHECKING
from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import Articulation

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

def foot_pos(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), body_names: list[str] = [".*"]) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    idx, _ = asset.find_bodies([body_part + "_foot" for body_part in body_names], preserve_order=True)

    return torch.flatten(asset.data.body_pos_w[:, idx], start_dim=1)

def base_pose(env:ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]

    return asset.data.root_pose_w

def muscle_length(env:ManagerBasedEnv, asset_cfg = SceneEntityCfg("robot")):
    robot: Articulation = env.scene[asset_cfg.name]

    if not hasattr(robot.actuators["base_legs"].muscle_model, "lce_tensor"):
        return torch.zeros(env.num_envs, 24, device=env.device)
    
    return robot.actuators["base_legs"].muscle_model.lce_tensor
    
def muscle_vel(env:ManagerBasedEnv, asset_cfg = SceneEntityCfg("robot")):
    robot: Articulation = env.scene[asset_cfg.name]

    if not hasattr(robot.actuators["base_legs"].muscle_model, "lce_dot_tensor"):
        return torch.zeros(env.num_envs, 24, device=env.device)

    return robot.actuators["base_legs"].muscle_model.lce_dot_tensor

def muscle_force(env:ManagerBasedEnv, asset_cfg = SceneEntityCfg("robot")):
    robot: Articulation = env.scene[asset_cfg.name]

    if not hasattr(robot.actuators["base_legs"].muscle_model, "force_tensor"):
        return torch.zeros(env.num_envs, 24, device=env.device)

    return robot.actuators["base_legs"].muscle_model.force_tensor
