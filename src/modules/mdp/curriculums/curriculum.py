from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def change_reward_weights(env: ManagerBasedRLEnv, env_ids: list[int], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    
    vel_rew_term_cfg = env.reward_manager.get_term_cfg("linear_velocity_x")
    standing_rew_term_cfg = env.reward_manager.get_term_cfg("capped_base_height")

    distance_travelled = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    avg_body_height = torch.mean(asset.data.root_pos_w[env_ids, 2])

    if avg_body_height > 0.3:
        env.reward_manager.set_term_cfg("linear_velocity_x", vel_rew_term_cfg)
        env.reward_manager.set_term_cfg("capped_base_height", standing_rew_term_cfg)

def change_hopping_stability_reward(env: ManagerBasedRLEnv, env_ids: list[int], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    #idea is to make a function, that increases the reward for stabilization in hopping task better over time
    env.common_step_counter


