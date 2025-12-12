# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import RL_environments

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
parser.add_argument(
    "--ray-proc-id", "-rid", type=int, default=None, help="Automatically configured by Ray integration, otherwise None."
)

parser.add_argument("--trial_id", type=int, default=0)
parser.add_argument("--n_steps", type=int, default=80) # num_env steps in PPO
parser.add_argument("--init_noise", type=float, default=1.0)
parser.add_argument("--value_loss_coeff", type=float, default=1.0)
parser.add_argument("--clip_param", type=float, default=0.2)
parser.add_argument("--ent_coef", type=float, default=0.01)
parser.add_argument("--num_learning_epochs", type=int, default=5)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--learning_rate", type=float, default=1e-4)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--gae_lambda", type=float, default=0.95)
parser.add_argument("--desired_kl", type=float, default=0.01)
parser.add_argument("--max_grad_norm", type=float, default=1.0)

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import logging
import os
import torch

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# import logger
logger = logging.getLogger(__name__)

# PLACEHOLDER: Extension template (do not remove this comment)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Train with RSL-RL agent."""
    
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    agent_cfg.num_steps_per_env = args_cli.n_steps
    agent_cfg.policy.init_noise_std = args_cli.init_noise
    agent_cfg.algorithm.value_loss_coef = args_cli.value_loss_coeff
    agent_cfg.algorithm.clip_param = args_cli.clip_param
    agent_cfg.algorithm.entropy_coef = args_cli.ent_coef
    agent_cfg.algorithm.num_learning_epochs = args_cli.num_learning_epochs
    agent_cfg.algorithm.num_mini_batches = args_cli.batch_size
    agent_cfg.algorithm.learning_rate = args_cli.learning_rate
    agent_cfg.algorithm.gamma = args_cli.gamma
    agent_cfg.algorithm.lam = args_cli.gae_lambda
    agent_cfg.algorithm.desired_kl =args_cli.desired_kl
    agent_cfg.algorithm.max_grad_norm = args_cli.max_grad_norm

    log_dir = os.path.join("logs", "optuna", f"trial_{args_cli.trial_id}")

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)

    print("---ISAAC_LAB_INIT_COMPLETE---")
    sys.stdout.flush()

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    avg_return = evaluate_policy(env, runner)

    result_path = os.path.join(log_dir, "optuna_result.txt")

    print(f"Writing result {avg_return} to {result_path}")
    with open(result_path, "w") as f:
        f.write(str(avg_return))

    # close the simulator
    env.close()

def evaluate_policy(env:RslRlVecEnvWrapper, runner: OnPolicyRunner):
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    policy_nn = runner.alg.policy

    rewards = torch.zeros(env.num_envs, device=env.unwrapped.device)

    with torch.inference_mode():
        policy_nn.reset(torch.ones(env.num_envs))
        env.reset()

    obs = env.get_observations()
    # simulate environment for 10 seconds to get avg rewards out
    for _ in range(int(10 * (1 / env.unwrapped.step_dt))):
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, rew, dones, _ = env.step(actions)
            # reset recurrent states for episodes that have terminated
            policy_nn.reset(dones)

            rewards = rewards + rew

    return torch.mean(rewards).item()


if __name__ == "__main__":
    os.makedirs(os.path.join("logs", "optuna"), exist_ok=True)
    main()
    # close sim app
    simulation_app.close()
