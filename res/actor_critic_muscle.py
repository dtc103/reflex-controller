# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.networks import MLP, EmpiricalNormalization


class ActorCritic(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        obs,
        obs_groups,
        num_actions,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        squash_actions: bool = True,
        tanh_squash_epsilon: float = 1e-7,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()

        # get the observation dimensions
        self.obs_groups = obs_groups
        num_actor_obs = 0
        for obs_group in obs_groups["policy"]:
            assert len(obs[obs_group].shape) == 2, "The ActorCritic module only supports 1D observations."
            num_actor_obs += obs[obs_group].shape[-1]
        num_critic_obs = 0
        for obs_group in obs_groups["critic"]:
            assert len(obs[obs_group].shape) == 2, "The ActorCritic module only supports 1D observations."
            num_critic_obs += obs[obs_group].shape[-1]

        # actor
        self.actor = MLP(num_actor_obs, num_actions, actor_hidden_dims, activation)
        # actor observation normalization
        self.actor_obs_normalization = actor_obs_normalization
        if actor_obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization(num_actor_obs)
        else:
            self.actor_obs_normalizer = torch.nn.Identity()
        print(f"Actor MLP: {self.actor}")

        # critic
        self.critic = MLP(num_critic_obs, 1, critic_hidden_dims, activation)
        # critic observation normalization
        self.critic_obs_normalization = critic_obs_normalization
        if critic_obs_normalization:
            self.critic_obs_normalizer = EmpiricalNormalization(num_critic_obs)
        else:
            self.critic_obs_normalizer = torch.nn.Identity()
        print(f"Critic MLP: {self.critic}")

        # Action noise
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        # Squashing configuration
        self.squash_actions = squash_actions
        self._squash_eps = tanh_squash_epsilon

        # Cached distributional parameters
        self._base_mean: torch.Tensor | None = None
        self._base_std: torch.Tensor | None = None
        self.base_distribution: Normal | None = None
        self._last_action: torch.Tensor | None = None  # Squashed action sample
        self._last_pre_tanh_action: torch.Tensor | None = None  # Unsquashed sample used to compute jacobians

        # disable args validation for speedup
        Normal.set_default_validate_args(False)

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        """Returns the *pre-squash* mean (used for KL computations and logging)."""
        if self._base_mean is None:
            raise RuntimeError("Distribution not initialised. Call act() or update_distribution() first.")
        return self._base_mean

    @property
    def action_std(self):
        """Returns the *pre-squash* std (used for KL computations and logging)."""
        if self._base_std is None:
            raise RuntimeError("Distribution not initialised. Call act() or update_distribution() first.")
        return self._base_std

    @property
    def entropy(self):
        if self.base_distribution is None:
            raise RuntimeError("Distribution not initialised. Call act() or update_distribution() first.")

        base_entropy = self.base_distribution.entropy()  # shape: [batch, num_actions]

        if not self.squash_actions:
            return base_entropy.sum(dim=-1)

        # Use the latest sample if available; otherwise draw a fresh rsample (ensures differentiability).
        if self._last_action is None or self._last_action.shape != base_entropy.shape:
            pre_tanh = self.base_distribution.rsample()
            action = torch.tanh(pre_tanh)
        else:
            action = self._last_action

        log_det_jacob = torch.log(torch.clamp(1.0 - action.pow(2), min=self._squash_eps))
        entropy = base_entropy + log_det_jacob
        return entropy.sum(dim=-1)

    def update_distribution(self, obs):
        # compute mean
        mean = self.actor(obs)
        # compute standard deviation
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        self._base_mean = mean
        self._base_std = std
        self.base_distribution = Normal(mean, std)

        # Keep compatibility with legacy code that references self.distribution directly
        self.distribution = self.base_distribution

    def act(self, obs, **kwargs):
        obs = self.get_actor_obs(obs)
        obs = self.actor_obs_normalizer(obs)
        self.update_distribution(obs)

        # Draw a reparameterised sample so gradients flow correctly during optimisation
        pre_tanh_action = self.base_distribution.rsample()

        if self.squash_actions:
            action = torch.tanh(pre_tanh_action)
            self._last_pre_tanh_action = pre_tanh_action
            self._last_action = action
            return action

        self._last_pre_tanh_action = None
        self._last_action = pre_tanh_action
        return pre_tanh_action

    def act_inference(self, obs):
        obs = self.get_actor_obs(obs)
        obs = self.actor_obs_normalizer(obs)
        mean = self.actor(obs)
        if self.squash_actions:
            return torch.tanh(mean)
        return mean

    def evaluate(self, obs, **kwargs):
        obs = self.get_critic_obs(obs)
        obs = self.critic_obs_normalizer(obs)
        return self.critic(obs)

    def get_actor_obs(self, obs):
        obs_list = []
        for obs_group in self.obs_groups["policy"]:
            obs_list.append(obs[obs_group])
        return torch.cat(obs_list, dim=-1)

    def get_critic_obs(self, obs):
        obs_list = []
        for obs_group in self.obs_groups["critic"]:
            obs_list.append(obs[obs_group])
        return torch.cat(obs_list, dim=-1)

    def get_actions_log_prob(self, actions):
        if self.base_distribution is None:
            raise RuntimeError("Distribution not initialised. Call act() or update_distribution() first.")

        if not self.squash_actions:
            return self.base_distribution.log_prob(actions).sum(dim=-1)

        eps = self._squash_eps
        clamped_actions = torch.clamp(actions, -1.0 + eps, 1.0 - eps)
        pre_tanh_actions = self._atanh(clamped_actions)

        base_log_prob = self.base_distribution.log_prob(pre_tanh_actions)
        log_det_jacob = torch.log(torch.clamp(1.0 - clamped_actions.pow(2), min=eps))

        return (base_log_prob - log_det_jacob).sum(dim=-1)

    def update_normalization(self, obs):
        if self.actor_obs_normalization:
            actor_obs = self.get_actor_obs(obs)
            self.actor_obs_normalizer.update(actor_obs)
        if self.critic_obs_normalization:
            critic_obs = self.get_critic_obs(obs)
            self.critic_obs_normalizer.update(critic_obs)

    def load_state_dict(self, state_dict, strict=True):
        """Load the parameters of the actor-critic model.

        Args:
            state_dict (dict): State dictionary of the model.
            strict (bool): Whether to strictly enforce that the keys in state_dict match the keys returned by this
                           module's state_dict() function.

        Returns:
            bool: Whether this training resumes a previous training. This flag is used by the `load()` function of
                  `OnPolicyRunner` to determine how to load further parameters (relevant for, e.g., distillation).
        """

        super().load_state_dict(state_dict, strict=strict)
        return True  # training resumes

    def _atanh(self, x: torch.Tensor) -> torch.Tensor:
        # x is assumed to be clamped to (-1, 1)
        return 0.5 * (torch.log1p(x) - torch.log1p(-x))