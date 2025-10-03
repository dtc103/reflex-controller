from __future__ import annotations

import torch

from typing import TYPE_CHECKING
from collections.abc import Sequence
import isaaclab.utils.string as string_utils
from isaaclab.assets.articulation import Articulation
from isaaclab.managers.action_manager import ActionTerm

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from .actions_cfg import MuscleActionCfg


class MuscleAction(ActionTerm):
    cfg: MuscleActionCfg
    _asset: Articulation
    _clip: torch.Tensor

    def __init__(self, cfg: MuscleActionCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)
        self._joint_ids, self._joint_names = self._asset.find_joints(self.cfg.joint_names, preserve_order=self.cfg.preserve_order)
        self._num_joints = len(self._joint_ids)

        if self._num_joints == self._asset.num_joints and not self.cfg.preserve_order:
            self._joint_ids = slice(None)

        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self.raw_actions)

        if self.cfg.clip is not None:
            if isinstance(cfg.clip, dict):
                self._clip = torch.tensor([[-float("inf"), float("inf")]], device = self.device).repeat(
                    self.num_envs, self.action_dim, 1
                )
                index_list, _, value_list = string_utils.resolve_matching_names_values(self.cfg.clip, self._joint_names)
                # we need to repeat 2 here, since we have always 2 actions for each joint
                index_list = index_list.repeat(2)
                value_list = value_list.repeat(2)
                
                self._clip[:, index_list] = torch.tensor(value_list, device=self.device)
            else:
                raise ValueError(f"Unsupported clip type: {type(cfg.clip)}. Supported types are dict.")
            
    @property
    def action_dim(self) -> int:
        return 2 * self._num_joints
    
    @property
    def raw_actions(self) ->torch.Tensor:
        return self._raw_actions
    
    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions
    
    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        self._processed_actions = self.raw_actions * self.cfg.scale + self.cfg.offset # self.cfg.scale will be 0.5 and self.cfg.offset also 0.5

        self._processed_actions = self._processed_actions.clamp(min=0, max=1)
        # if self.cfg.clip is not None:
        #     self._processed_actions = torch.clamp(
        #         self._processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
        #     )

    def apply_actions(self):
        self._asset.set_joint_position_target(self.processed_actions[:, :self._num_joints])
        self._asset.set_joint_velocity_target(self.processed_actions[:, self._num_joints:])

    def reset(self, env_ids: Sequence | None = None) -> None:
        if env_ids is None:
            self._raw_actions.zero_()
        else:
            self._raw_actions[env_ids] = 0.0
            
