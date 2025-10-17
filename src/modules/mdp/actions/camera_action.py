from __future__ import annotations

from isaaclab.managers.action_manager import ActionTerm
from typing import TYPE_CHECKING
from isaaclab.assets.articulation import Articulation

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from .actions_cfg import CameraActionCfg


class CameraAction(ActionTerm):
    """The one and only reason for this class is to have the camera always pointed at one articulation
    since I haven't found out how to follow objects with cameras effciciently yet
    """
    cfg: CameraActionCfg
    _asset: Articulation
    _clip: torch.Tensor

    def __init__(self, cfg: CameraActionCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)

    @property
    def action_dim(self) -> int:
        return 0
    
    @property
    def raw_actions(self):
        pass
    
    @property
    def processed_actions(self):
        pass

    def process_actions(self, actions):
        pass

    def apply_actions(self):
        asset:Articulation = self._env.scene["robot"]
        root_pos = asset.data.root_pos_w[0].cpu().numpy().tolist()

        cam_x = root_pos[0] + 3.0 * torch.cos(torch.tensor(-torch.pi / 2)).item()
        cam_y = root_pos[1] + 3.0 * torch.sin(torch.tensor(-torch.pi / 2)).item()
        cam_z = root_pos[2] + 1.0
        self._env.sim.set_camera_view(eye=[cam_x, cam_y, cam_z], target=[root_pos[0], root_pos[1], 0.5])
