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

    num_goals: int = 4

    radius_range: tuple[float, float] = (0.25, 0.5)
    goal_tolerance :float= 0.025 # maximal euclidian distance to center of positions, that is allowed

    sphere_prim_path: str = "/Visuals/Command/goal_pos_{index}"
    sphere_colour_red: tuple = (1.0, 0.0, 0.0, 1.0)
    sphere_colour_green: tuple = (0.0, 1.0, 0.0, 1.0)

    sphere_marker_cfg: VisualizationMarkersCfg = POSITION_GOAL_MARKER_CFG.replace(
        prim_path=sphere_prim_path
    )

    sphere_marker_cfg.markers["target_far"].radius = goal_tolerance
    sphere_marker_cfg.markers["target_near"].radius = goal_tolerance