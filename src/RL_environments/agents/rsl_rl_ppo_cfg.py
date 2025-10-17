# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class UnitreeGo2MusclePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 80 #0.8 secs of rollout
    max_iterations = 1500
    save_interval = 50
    experiment_name = "unitree_go2_muscle_reach"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.6, #we change this, because we changed the PPO to be sqaushed between -1 and 1
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        #noise_std_type = "log",
        actor_hidden_dims=[1024, 512, 256],
        critic_hidden_dims=[1024, 512, 256],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.02, # a bit more exploration, because of changed actor critic
        num_learning_epochs=5,
        num_mini_batches=8,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.995,
        lam=0.95,
        desired_kl=0.01, # keep a bit lower, since we changed actor critic
        max_grad_norm=1.0,
    )

@configclass
class HoppingUnitreeGo2MusclePPORunnerCfg(UnitreeGo2MusclePPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "unitree_go2_muscle_hopping"

        self.policy.actor_obs_normalization = False
        self.policy.critic_obs_normalization = False
        
        self.algorithm.num_mini_batches = 8

@configclass
class WalkingUnitreeGo2MusclePPORunnerCfg(UnitreeGo2MusclePPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "unitree_go2_muscle_walking"

        self.policy.actor_obs_normalization = False
        self.policy.critic_obs_normalization = False
        
        self.algorithm.num_mini_batches = 10
        
        self.policy.init_noise_std = 1.0
        self.algorithm.num_learning_epochs = 7

        

"""
original values
@configclass
class UnitreeGo2MusclePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 50
    experiment_name = "unitree_go2_rough"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
"""