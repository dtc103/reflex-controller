from __future__ import annotations

import math
from dataclasses import MISSING

from isaaclab.managers import CommandTermCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import POSITION_GOAL_MARKER_CFG
from isaaclab.markers import VisualizationMarkersCfg, VisualizationMarkers
from isaaclab.utils import configclass

from .reach_command import ReachPositionCommand

@configclass
class ReachPositionCommandCfg(CommandTermCfg):
    class_type: type = ReachPositionCommand

    asset_name: str = MISSING

    radius_range: tuple[float, float] = (0.25, 0.5)
    goal_tolerance :float= 0.025 # maximal euclidian distance to center of positions, that is allowed

    sphere_prim_path: str = "/Visuals/Command/goal_pos_{index}"

    sphere_marker_cfg: VisualizationMarkersCfg = POSITION_GOAL_MARKER_CFG.replace(
        prim_path=sphere_prim_path
    )

    body_names: list[str] = None

    sphere_marker_cfg.markers["target_far"].radius = goal_tolerance
    sphere_marker_cfg.markers["target_near"].radius = goal_tolerance