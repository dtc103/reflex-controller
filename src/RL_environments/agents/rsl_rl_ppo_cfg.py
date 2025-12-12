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

# ophysics freq = 500Hz
# agent frequency = 125Hz
# render step = 4
@configclass
class WalkingUnitreeGo2MuscleDirectPPORunnerCfg(UnitreeGo2MusclePPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.num_steps_per_env = 75

        self.clip_actions = 1.0
        
        self.policy.actor_hidden_dims = [512, 256, 128]
        self.policy.critic_hidden_dims = [512, 256, 128]


        self.experiment_name = "unitree_go2_muscle_walking_direct"

        self.policy.actor_obs_normalization = True
        self.policy.critic_obs_normalization = True

        #self.policy.noise_std_type = 'log'
        self.policy.init_noise_std = 0.7

        self.algorithm.num_mini_batches = 6
        self.algorithm.num_learning_epochs = 4

        self.algorithm.learning_rate = 1.0e-3
        self.algorithm.gamma = 0.991
        self.algorithm.entropy_coef = 0.008
        self.algorithm.clip_param = 0.2
        self.algorithm.lam = 0.95
        self.algorithm.desired_kl = 0.01
        self.algorithm.max_grad_norm = 1.0


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
        
        self.algorithm.num_mini_batches = 9
        
        self.policy.init_noise_std = 1.0
        self.algorithm.num_learning_epochs = 6
'''
Best Hyperparameters:
  learning_rate: 1.4519545357271965e-05
  n_steps: 80
  batch_size: 14
  gamma: 0.996
  gae_lambda: 0.9200000000000002
  clip_param: 0.1897956278506332
  ent_coef: 0.007
  desired_kl: 0.009000000000000001
  max_grad_norm: 0.9000000000000001
  init_noise: 0.4341612839088024
  num_learning_epochs: 4
'''

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