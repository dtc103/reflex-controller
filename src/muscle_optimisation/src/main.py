import argparse
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on using the interactive scene interface.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
import math
from muscle_actuator import muscle_actuator, muscle_actuator_cfg

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, Articulation
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass

##
# Pre-defined configs
##
from robot_config.unitree_muscle_cfg import UNITREE_GO2_MUSCLE_CFG


def follow_robot_with_camera(sim: SimulationContext, robot, angle_rad=0.0, radius=3.0, height=1.0):
    # Get the root position of the first environment's robot
    root_pos = robot.data.root_pos_w[0].cpu().numpy()  # shape: (3,)
    target = root_pos.tolist()

    # Compute camera position in orbit
    cam_x = root_pos[0] + radius * math.cos(angle_rad)
    cam_y = root_pos[1] + radius * math.sin(angle_rad)
    cam_z = root_pos[2] + height

    sim.set_camera_view(eye=[cam_x, cam_y, cam_z], target=[target[0], target[1], 0.5])

@configclass
class UnitreeSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # articulation
    unitree: ArticulationCfg = UNITREE_GO2_MUSCLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")



def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):

    """Runs the simulation loop."""
    # Extract scene entities
    # note: we only do this here for readability.
    robot:Articulation = scene["unitree"]
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0
    angle = 0
    # Simulation loop

    while simulation_app.is_running():
        angle += 0.05
        follow_robot_with_camera(sim, robot, angle_rad=angle)

        if count % 500 == 0:
            print(robot.actuators["base_legs"])

        sim.step()
        count += 1
        scene.update(sim_dt)



def main():

    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.5, 0.0, 4.0], [0.0, 0.0, 2.0])
    # Design scene
    scene_cfg = UnitreeSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(cfg=scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)



if __name__ == "__main__":

    # run the main function

    main()

    # close sim app

    simulation_app.close()