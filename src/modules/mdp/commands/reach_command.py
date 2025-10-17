from __future__ import annotations

import torch 
from typing import TYPE_CHECKING

import isaaclab.envs.mdp as mdp
from isaaclab.managers.command_manager import CommandTerm
from isaaclab.utils import configclass
from isaaclab.assets import Articulation
from isaaclab.markers.config import POSITION_GOAL_MARKER_CFG
from isaaclab.markers import VisualizationMarkersCfg, VisualizationMarkers
import re

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from .commands_cfg import ReachPositionCommandCfg

class ReachPositionCommand(CommandTerm):
    cfg: ReachPositionCommandCfg

    def __init__(self, cfg: ReachPositionCommandCfg, env: ManagerBasedRLEnv):
        self._robot: Articulation = env.scene[cfg.asset_name]

        hip_names = self._make_hip_string(cfg.body_names)
        self._hip_body_ids, self._hip_body_names = self._robot.find_bodies(hip_names, preserve_order=True) #order: (FL, FR, RL, RR)
        self._num_hips = len(self._hip_body_ids)

        super().__init__(cfg, env)

        self._desired_pos_w = torch.zeros(self.num_envs, len(self._hip_body_ids) * 3, device=self.device)
        self._reached = torch.zeros(self.num_envs, len(self._hip_body_ids), dtype=torch.bool, device=self.device)

        # defines, if the spheres are right or left of the body (-1 is right, 1 is left)
        is_R_part_re = re.compile(r"^.R")
        self._directions = torch.tensor([-1 if is_R_part_re.match(body_name) else 1 for body_name in cfg.body_names], device=self.device)
        
        #TODO implement still
        #self.metrics["goal_distance"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self._desired_pos_w

    def _make_hip_string(self, body_parts):
        return [body_part + "_hip" for body_part in body_parts]

    def _resample_command(self, env_ids):
        """Generate random goal positions for the end-effektors to reach. Here the positions to reach are sampled from a half-sphere, which center is defined as the hip joint position. 
        From there the circles are spanning from the mid of the body outwards
        """
        if env_ids is None:
            env_ids = slice(None)


        hip_pos_w = self._robot.data.body_pos_w[env_ids][:, self._hip_body_ids, :]
        env_count = len(env_ids)

        # sample a point in a cylinder
        r0, r1 = self.cfg.radius_range
        h0, h1 = self.cfg.height_range
        u = torch.rand(env_count, self._num_hips, device=self.device)
        radii = (u * (r1 - r0) + r0)
        #we do not take the whole circle here, but only the lower part, so half of a circle
        phi = torch.tensor(-torch.pi, device=self.device) # * torch.rand(env_count, self._num_hips, device=self.device)

        xs = radii * torch.cos(phi)
        ys = (u * (h1 - h0) + h0) * self._directions
        zs = radii * torch.sin(phi)

        offsets = torch.stack([xs, ys, zs], dim=2)

        goal_pos = hip_pos_w + offsets
        self._desired_pos_w[env_ids] = goal_pos.reshape(env_count, self._num_hips * 3)

        self._reached[env_ids] = False

    def _set_debug_vis_impl(self, debug_vis):
        if debug_vis:
            if not hasattr(self, "_spheres") :
                self._spheres = [] 
                for i in range(self._num_hips):
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
        
        goal_positions = self._desired_pos_w.view(self.num_envs, self._num_hips, 3)
        for i in range(self._num_hips):
            positions = goal_positions[:, i, :]
            self._spheres[i].visualize(
                positions,
                marker_indices=self._reached[:, i].int()
            )

    def _update_command(self):
        foot_idxs, _ = self._robot.find_bodies(["FL_foot"], preserve_order=True)

        foot_positions = self._robot.data.body_pos_w[:, foot_idxs, :]

        #euclidian distance
        distances = torch.norm((self._desired_pos_w.view(self.num_envs, self._num_hips, 3) - foot_positions), dim=-1)

        # if the distance is smaller than the threshold, we did it
        self._reached[:, :] = (distances < self.cfg.goal_tolerance)

    def _update_metrics(self):
        #TODO implement this for logging
        # self.metrics["distance_to_goal"] = pass
        #self.metrics["goal_distance"] = torch.zeros(self.num_envs, device=self.device)
        pass
        
    def reset(self, env_ids = None):
        return super().reset(env_ids)
    
    def compute(self, dt):
        super().compute(dt)
    
