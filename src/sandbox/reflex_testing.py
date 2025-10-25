import argparse
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on using the interactive scene interface.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--training", action="store_true")
parser.add_argument("--fine_tune", action="store_true")
parser.add_argument("--max_iterations", default=250, type=int)
parser.add_argument("--term_perc", default=0.95, type=float)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
from modules.muscle_actuator.muscle_actuator_parameters import muscle_params
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, Articulation, AssetBase
from isaaclab_assets import UNITREE_GO2_CFG
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass

import cma
import numpy as np
from tqdm import tqdm
import time
import pickle
import datetime

##
# Pre-defined configs
##
from modules.robot_config.unitree_muscle_cfg import UNITREE_GO2_REFLEX_CFG
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR


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
    unitree: ArticulationCfg = UNITREE_GO2_REFLEX_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
    )

T_SIM_MAX = 20 #seconds
HEIGH_TERMINATION_CRITERIA = 0.2 #meters
TARGET_VEL = 1.5

def cost_function(target_vel, sim_times, distances):
    """Const function calculation for each environment"""
    avg_vels = distances / torch.clamp(sim_times, min=1.0)
    
    rew_1 = torch.abs(1.0 - (avg_vels / target_vel)) + (1.0 - sim_times / T_SIM_MAX)

    return rew_1

def env_is_terminated(robot: Articulation):
    # index 0 of body_idx is base, index 2 is z-axis
    return robot.data.body_pos_w[:, 0, 2] < HEIGH_TERMINATION_CRITERIA

def reset_robot(scene: InteractiveScene, robot: Articulation):
    root_state = robot.data.default_root_state.clone()
    root_state[:, :3] += scene.env_origins
    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(root_state[:, 7:])
    joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    robot.reset()

    robot.write_data_to_sim()


def run_environments(sim: sim_utils.SimulationContext, scene: InteractiveScene, samples, term_perc = 0.95):
    robot: Articulation = scene["unitree"]

    #samples are given in a list of num_envs numpy arrays with each 1176 params
    #turn into shape offsets = (n_envs, 24), conn_mat_L = (n_envs, 24, 24), conn_mat_R = (n_envs, 24, 24)
    offsets = samples[:, :24]
    conn_L = samples[:, 24 : 24*24+24].reshape(-1, 24, 24)
    conn_F = samples[:, 24*24+24 : 2*24*24+24].reshape(-1, 24, 24)

    robot.actuators["base_legs"].reflex_controller.set_parameters(
        connection_matrix_L = conn_L, connection_matrix_F = conn_F, offsets = offsets
    )

    sim_dt = sim.get_physics_dt()
    terminated = torch.full((scene.num_envs,), False, device=scene.device)

    max_steps = int(T_SIM_MAX * (1 / sim_dt))
    distances = torch.zeros(scene.num_envs, device=scene.device)
    sim_times = torch.zeros(scene.num_envs, device=scene.device)

    reset_robot(scene, robot)

    # keep running, if
    #   - all simulations environments are below T_SIM_MAX seconds
    #   - if less than 95% of the environments stopped
    while simulation_app.is_running() and torch.all(sim_times < max_steps) and (terminated.float().mean() < term_perc):
        # if the environment ever gets terminated = True, it will stay terminated
        terminated = terminated | env_is_terminated(robot)
        #apply actuator model
        robot.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)
        
        curr_distances = torch.sum(torch.square(robot.data.root_pos_w[~terminated, :2] - scene.env_origins[~terminated, :2]), dim=-1)
        distances[~terminated] = curr_distances
        sim_times[~terminated] += 1

    return sim_times / (1/sim_dt), distances

def run_optimization(sim: sim_utils.SimulationContext, scene: InteractiveScene, args):
    reflex_controller = scene["unitree"].actuators["base_legs"].reflex_controller
    bounds = [[-2, 2] * reflex_controller._num_matrix_parameters * 2, [-1, 1] * reflex_controller.num_muscles]

    optimizer = cma.CMAEvolutionStrategy(
        x0=np.zeros(bounds.shape[0]),
        sigma0=1.5,
        options={
            "bounds": [bounds[:, 0].tolist(), bounds[:, 1].tolist()],
            "popsize": scene.num_envs
        }
    )

    str_time = datetime.datetime.strftime("%Y%m%d_%H%M")
    file_name = f"best_solution_{str_time}"

    best_solution = None
    best_value = float('inf')

    for generation in tqdm(range(args.max_iterations)):
        samples = optimizer.ask()
        samples_torch = torch.from_numpy(np.array(samples)).cuda().float()

        sim_times_s, dists = run_environments(sim, scene, samples_torch, args.term_perc)
        rewards = cost_function(TARGET_VEL, sim_times_s, dists)
        rewards_np = rewards.detach().cpu().numpy()

        min_idx = np.argmin(rewards_np)
        if rewards_np[min_idx] < best_value:
            best_value = rewards_np[min_idx]
            best_solution = samples[min_idx].copy()
            tqdm.write(f"New best at generation {generation}: {best_value}")

        tqdm.write(f"Avg steps at evolution {generation}: {torch.mean(sim_times_s)}")
        optimizer.tell(samples, rewards_np)

    optimizer.result_pretty()

    np.save(f'{file_name}.npy', best_solution)
    with open(f'{file_name}.pkl', 'wb') as f:
        pickle.dump(
            {
                'solution': best_solution,
                'cost': best_value,
                'generations': args.max_iterations
            }, 
            f
        )

    quit()

def run_evaluation(sim, scene):
    robot: Articulation = scene["unitree"]
    sim_dt = sim.get_physics_dt()

    best_solution: np.ndarray = np.load('best_solution_longest.npy')

    offsets = torch.from_numpy(best_solution[:24]).cuda().float()
    conn_L = torch.from_numpy(best_solution[24 : 24*24+24].reshape(24, 24)).cuda().float()
    conn_F = torch.from_numpy(best_solution[24*24+24 : 2*24*24+24].reshape(24, 24)).cuda().float()

    if len(offsets.shape) == 1:
        offsets = offsets.unsqueeze(0)
    if len(conn_L.shape) == 2:
        conn_L = conn_L.unsqueeze(0)
    if len(conn_F.shape) == 2:
        conn_F = conn_F.unsqueeze(0)

    robot.actuators["base_legs"].reflex_controller.set_parameters(
        connection_matrix_L = conn_L, connection_matrix_F = conn_F, offsets = offsets
    )

    count = 0
    secs_per_run = 5
    reset_robot(scene, robot)

    while simulation_app.is_running():
        if count >= (secs_per_run * (1 / sim_dt)):
            count = 0
            reset_robot(scene, robot)
            print("##### RESET #####")

        print(robot.actuators["base_legs"].reflex_controller._activations)

        robot.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)   

        count += 1     


def fine_tune_optimization(sim, scene, args):
    best_solution = np.load('best_solution_longest.npy')

    reflex_controller = scene["unitree"].actuators["base_legs"].reflex_controller
    bounds = [[-2, 2] * reflex_controller._num_matrix_parameters * 2, [-1, 1] * reflex_controller.num_muscles]

    optimizer = cma.CMAEvolutionStrategy(
        x0=best_solution,
        sigma0=1.0,
        options={
            "bounds": [bounds[:, 0].tolist(), bounds[:, 1].tolist()],
            "popsize": scene.num_envs
        }
    )

    best_solution = None
    best_value = float('inf')

    for generation in tqdm(range(args.max_iterations)):
        samples = optimizer.ask()
        samples_torch = torch.from_numpy(np.array(samples)).cuda().float()

        sim_times_s, dists = run_environments(sim, scene, samples_torch, args.term_perc)
        rewards = cost_function(TARGET_VEL, sim_times_s, dists)
        rewards_np = rewards.detach().cpu().numpy()

        min_idx = np.argmin(rewards_np)
        if rewards_np[min_idx] < best_value:
            best_value = rewards_np[min_idx]
            best_solution = samples[min_idx].copy()
            tqdm.write(f"New best at generation {generation}: {best_value}")

        tqdm.write(f"Avg steps at evolution {generation}: {torch.mean(sim_times_s)}")
        optimizer.tell(samples, rewards_np)

    optimizer.result_pretty()

    np.save('best_solution_2.npy', best_solution)

    with open('best_solution_2.pkl', 'wb') as f:
        pickle.dump(
            {
                'solution': best_solution,
                'cost': best_value,
                'generations': args.max_iterations
            },
            f
    )

    quit()

def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    
    # Find a suitable simulator dt for the simulation!!!!!!!!!!!!!
    sim_cfg.dt = muscle_params["dt"]
    sim_cfg.render_interval = 1
    
    sim = SimulationContext(sim_cfg)
    # Set main camera
    #sim.set_camera_view([0.0, 2.0, 2.0], [0.0, 0.0, 0.3])
    # Design scene
    scene_cfg = UnitreeSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene_cfg.unitree.spawn.articulation_props.fix_root_link = False

    scene_cfg.unitree.init_state.pos = (0.0, 0.0, 0.4)
    scene_cfg.unitree.init_state.lin_vel = (1.0, 0.0, 0.0)
    # scene_cfg.unitree.init_state.joint_pos = {
    #     ".*L_hip_joint": 0.3,
    #     ".*R_hip_joint": -0.3,
    #     "F[L,R]_thigh_joint": torch.pi/2.0 - 0.5,
    #     "R[L,R]_thigh_joint": torch.pi/2.0 - 0.5,
    #     ".*_calf_joint": -2.7,
    # }
    
    scene = InteractiveScene(cfg=scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")

    # Run the simulator
    if args_cli.training:
        run_optimization(sim, scene, args_cli)
    elif args_cli.fine_tune:
        fine_tune_optimization(sim, scene, args_cli)
    else:
        run_evaluation(sim, scene)

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
