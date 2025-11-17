import argparse
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on using the interactive scene interface.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--training", action="store_true")
parser.add_argument("--fine_tune", action="store_true")
parser.add_argument("--max_iterations", default=250, type=int)
parser.add_argument("--term_perc", default=0.95, type=float)
parser.add_argument("--num_cma", type=int, default=10, help="Number of CMA-ES islands.")
parser.add_argument("--cma_seed", type=int, default=0, help="Base RNG seed for CMA-ES islands.")

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
import pickle
import datetime

##
# Pre-defined configs
##
from modules.robot_config.unitree_muscle_cfg import UNITREE_GO2_REFLEX_8D_CFG
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
    unitree: ArticulationCfg = UNITREE_GO2_REFLEX_8D_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
    )

T_SIM_MAX = 10 #seconds
HEIGH_TERMINATION_CRITERIA = 0.25 #meters
TARGET_VEL = 2.0

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

    robot.actuators["base_legs"].reflex_controller.set_parameters(
        samples
    )

    sim_dt = sim.get_physics_dt()
    terminated = torch.full((scene.num_envs,), False, device=scene.device)

    max_steps = int(T_SIM_MAX * (1 / sim_dt))
    distances = torch.zeros(scene.num_envs, device=scene.device)
    sim_times = torch.zeros(scene.num_envs, device=scene.device)

    reset_robot(scene, robot)

    # keep running, if
    #   - all simulations environments are below T_SIM_MAX seconds
    #   - if less than term_perc of the environments stopped
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
    bounds = np.concatenate([np.array([[-1, 1]] * reflex_controller._num_matrix_parameters * 2), np.array([[-1, 1]] * reflex_controller.num_muscles)])
    num_params = bounds.shape[0]

    if scene.num_envs % args.num_cma != 0:
        raise ValueError(
            f"scene.num_envs ({scene.num_envs}) must be divisible by --num_cma ({args.num_cma})."
        )
    popsize_per_island = scene.num_envs //args.num_cma

    def spawn_strategy():
        rng = np.random.default_rng(args.cma_seed)
        seed = rng.integers(1, 2**30)
        return cma.CMAEvolutionStrategy(
            x0=np.zeros(num_params),
            sigma0=1.1,
            inopts={
                "bounds" : [bounds[:, 0].tolist(), bounds[:, 1].tolist()],
                "popsize" : popsize_per_island,
                "seed" : int(seed),
                "verb_disp" : 0
            }
        )

    strategies = [spawn_strategy() for _  in range(args.num_cma)]

    str_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"solutions/best_solution_{str_time}"
    print("Starting experiment at:", str_time)

    best_solution = None
    best_value = float('inf')

    for generation in tqdm(range(args.max_iterations), desc="CMA-ES generations"):
        batch = []
        slices= []

        for island_idx, strategy in enumerate(strategies):
            if strategy.stop():
                result = strategy.result
                if result.fbest < best_value:
                    best_value = float(result.fbest)
                    best_solution = np.array(result.xbest, copy=True)
                strategies[island_idx] = spawn_strategy()
                strategy = strategies[island_idx]

            samples = strategy.ask()
            start = len(batch)
            batch.extend(samples)
            slices.append((island_idx, start, len(samples)))

        batch_array = np.asarray(batch, dtype=np.float32)
        samples_torch = torch.from_numpy(batch_array).to(scene.device)
        sim_times, dists = run_environments(sim, scene, samples_torch, args.term_perc)
        rewards = cost_function(TARGET_VEL, sim_times, dists)
        rewards_np = rewards.detach().cpu().numpy()

        generation_best_idx = int(np.argmin(rewards_np))
        generation_best_value = float(rewards_np[generation_best_idx])
        if generation_best_value < best_value:
            best_solution = batch_array[generation_best_idx].astype(np.float64, copy=True)
            tqdm.write(f"New global best at generation {generation}: {best_value:.6f}")

        tqdm.write(
            f"[Gen {generation:04d}] mean seconds = {torch.mean(sim_times):.2f} | "
            f"std = {torch.std(sim_times):.2f} | "
            f"max = {torch.max(sim_times):.2f} | "
            f"min = {torch.min(sim_times):.2f} | "
            f"best reward = {generation_best_value:.6f}"
        )

        for island_idx, start, length in slices:
            end = start + length
            strategies[island_idx].tell(batch[start:end], rewards_np[start:end])
    
    for strategy in strategies:
        result = strategy.result
        if result.fbest < best_value:
            best_value = float(result.fbest)
            best_solution = np.array(result.xbest, copy=True)

    print("Best objective:", best_value)
    for idx, strategy in enumerate(strategies):
        print(f"Island {idx} summary:")
        strategy.result_pretty()

    np.save(f"{file_name}.npy", best_solution)
    with open(f"{file_name}.pkl", "wb") as f:
        pickle.dump(
            {"solution": best_solution, "cost": best_value, "generations": args.max_iterations},
            f,
        )

    quit()

def run_evaluation(sim, scene):
    robot: Articulation = scene["unitree"]
    sim_dt = sim.get_physics_dt()

    best_solution: np.ndarray = np.load('solutions/best_solution_20251107_020049.npy')
    print("best_solution:", best_solution.shape)

    robot.actuators["base_legs"].reflex_controller.set_parameters(
        torch.from_numpy(best_solution).float().cuda()
    )

    count = 0
    secs_per_run = 2
    reset_robot(scene, robot)

    while simulation_app.is_running():
        if count >= (secs_per_run * (1 / sim_dt)):
            count = 0
            reset_robot(scene, robot)
            print("##### RESET #####")
        print("Activations:", robot.actuators["base_legs"].reflex_controller._activations)

        robot.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)   

        count += 1

def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    
    sim_cfg.dt = muscle_params["dt"]
    sim_cfg.render_interval = 1
    
    sim = SimulationContext(sim_cfg)
    # Set main camera
    #sim.set_camera_view([0.0, 2.0, 2.0], [0.0, 0.0, 0.3])
    # Design scene
    scene_cfg = UnitreeSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene_cfg.unitree.spawn.articulation_props.fix_root_link = False

    scene_cfg.unitree.init_state.pos = (0.0, 0.0, 0.4)
    scene_cfg.unitree.init_state.lin_vel = (1.5, 0.0, 0.0)
    scene_cfg.unitree.init_state.joint_pos = {
        ".*L_hip_joint": 0.0,
        ".*R_hip_joint": 0.0,
        "(FL|RR)_thigh_joint": -0.2,
        "(FR|RL)_thigh_joint": torch.pi / 4,
        ".*_calf_joint": -1.0,
    }
    
    scene = InteractiveScene(cfg=scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")

    # Run the simulator
    if args_cli.training:
        run_optimization(sim, scene, args_cli)
    else:
        run_evaluation(sim, scene)

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
