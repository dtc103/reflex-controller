from __future__ import annotations

import torch 
from typing import TYPE_CHECKING

import isaaclab.envs.mdp as mdp
from isaaclab.managers.command_manager import CommandTerm
from isaaclab.utils import configclass
from isaaclab.assets import Articulation
from isaaclab.markers.config import POSITION_GOAL_MARKER_CFG
from isaaclab.markers import VisualizationMarkersCfg, VisualizationMarkers

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from .commands_cfg import ReachPositionCommandCfg

class ReachPositionCommand(CommandTerm):
    cfg: ReachPositionCommandCfg

    def __init__(self, cfg: ReachPositionCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self._robot: Articulation = env.scene[cfg.asset_name]

        self._hip_body_ids, _ = self._robot.find_bodies([".*_hip"])

        if len(self._hip_body_ids) != 4:
            raise ValueError("Expected exactly four hip joints (FL, FR, RL, RR).")
        
        self._desired_pos_w = torch.zeros(self.num_envs, len(self._hip_body_ids) * 3, device=self.device)
        self._reached = torch.zeros(self.num_envs, 4, dtype=torch.bool, device=self.device)

        #TODO implement still
        #self.metrics["goal_distance"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self._desired_pos_w
    
    def _resample_command(self, env_ids):
        if env_ids is None:
            env_ids = slice(None)

        hip_pos_w = self._robot.data.body_pos_w[env_ids][:, self._hip_body_ids, :]

        outward_directions = torch.tensor([
            [1, 1, 1],
            [1, -1, 1],
            [1, 1, 1],
            [1, -1, 1]
        ], device=self.device)

        num_hips = len(self._hip_body_ids)

        u = torch.rand(self.num_envs, num_hips, device=self.device)
        radii = (u * (self.cfg.radius_range[1]**3 - self.cfg.radius_range[0]**3) + self.cfg.radius_range[0]**3) ** (1/3)

        theta = 2 * torch.pi * torch.rand(self.num_envs, num_hips, device=self.device)
        phi = 0.5 * torch.pi * torch.rand(self.num_envs, num_hips, device=self.device)

        xs = radii * torch.sin(phi) * torch.cos(theta)
        ys = radii * torch.cos(phi)
        zs = radii * torch.sin(phi) * torch.sin(theta)

        offsets = torch.stack([xs, ys, zs], dim=2)
        offsets = offsets * outward_directions.unsqueeze(0)

        goal_pos = hip_pos_w + offsets

        self._desired_pos_w = goal_pos.view(-1, num_hips * 3)

        self._reached[env_ids] = False

    def _set_debug_vis_impl(self, debug_vis):
        if debug_vis:
            if not hasattr(self, "_spheres"):
                self._spheres = [] 
                for i in range(self.cfg.num_goals):
                    marker_cfg: VisualizationMarkersCfg = self.cfg.sphere_marker_cfg.copy()
                    marker_cfg.prim_path = self.cfg.sphere_prim_path.format(index=i)
                    self._spheres.append(VisualizationMarkers(marker_cfg))
            for sphere in self._spheres:
                sphere.set_visibility(True)            
        else:
            if hasattr(self, "_spheres"):
                for sphere in self._spheres:
                    sphere.set_visibility(False)  

    def _debug_vis_callback(self, event):
        if not self._robot.is_initialized:
            return
        
        goal_positions = self._desired_pos_w.view(self.num_envs, self.cfg.num_goals, 3)
        for i in range(self.cfg.num_goals):
            positions = goal_positions[:, i, :]
            self._spheres[i].visualize(
                positions,
                marker_indices=self._reached[:, i].int()
            )

    def _update_command(self):
        foot_idxs, _ = self._robot.find_bodies([".*_foot"])

        foot_positions = self._robot.data.body_pos_w[:, foot_idxs, :]

        #euclidian distance
        distances = torch.norm((self._desired_pos_w.view(self.num_envs, self.cfg.num_goals, 3) - foot_positions), dim=-1)

        # if the distance is smaller than the threshold, we did it
        self._reached[:, :] = (distances < self.cfg.goal_tolerance).int()

    def _update_metrics(self):
        #TODO implement this for logging
        # self.metrics["distance_to_goal"] = pass
        #self.metrics["goal_distance"] = torch.zeros(self.num_envs, device=self.device)
        pass
        
    def reset(self, env_ids = None):
        return super().reset(env_ids)
    
    def compute(self, dt):
        super().compute(dt)
    
