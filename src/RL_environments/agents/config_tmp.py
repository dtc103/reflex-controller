# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
from modules.muscle_actuator.muscle_actuator_parameters import muscle_params
import torch


@configclass
class UnitreeGo2MusclePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = int(0.4 / muscle_params["dt"]) # steps that are necessary to get 0.4 sec of rollout
    max_iterations = 1500
    save_interval = 50
    experiment_name = "unitree_go2_muscle_reach"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=6,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=torch.e ** (-muscle_params["dt"] / 3.0), # we want effect horizon of ~3.0 secs. This is how we calculate gamma then
        lam=0.95,
        desired_kl=0.005,
        max_grad_norm=1.0,
    )
