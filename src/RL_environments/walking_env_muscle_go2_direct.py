from __future__ import annotations

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.sim.spawners.from_files import spawn_ground_plane, GroundPlaneCfg

from .walking_env_muscle_go2_direct_cfg import WalkingMuscleGo2DirectCfg


class WalkingMuscleGo2Direct(DirectRLEnv):
    cfg: WalkingMuscleGo2DirectCfg

    def __init__(self, cfg: WalkingMuscleGo2DirectCfg, render_mode: str | None = None, **kwags):
        super().__init__(cfg, render_mode, **kwags)

        #for direct control
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._previous_actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device = self.device)

        # self._actions = torch.zeros(self.num_envs, 2 * gym.spaces.flatdim(self.single_action_space), device=self.device)
        # self._previous_actions = torch.zeros(self.num_envs, 2 * gym.spaces.flatdim(self.single_action_space), device = self.device)

        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device) for key in [
                "forward_gauss",
                "action_regularization_1",
                "action_regularization_2",
                "angular_penalty",
            ]
        }

        self.total_mean_timesteps = 0

        self._base_id = torch.tensor(self._contact_sensor.find_bodies("base")[0], device=self.device)
        self._feet_ids, _ = self._contact_sensor.find_bodies(".*foot")
        self._undesired_contact_body_ids, _ = self._contact_sensor.find_bodies(["base", ".*thigh"])
        self._undesired_contact_body_ids = torch.tensor(self._undesired_contact_body_ids, device=self.device)

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor

        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        self.scene.clone_environments(copy_from_source=False)

        light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions):
        if self.cfg.camera_follow:
            root_pos = self._robot.data.root_pos_w[0].cpu().numpy().tolist()

            cam_x = root_pos[0] + 3.0 * torch.cos(torch.tensor(-self.total_mean_timesteps / 100)).item()
            cam_y = root_pos[1] + 3.0 * torch.sin(torch.tensor(-self.total_mean_timesteps / 100)).item()
            cam_z = root_pos[2] + 1.0
            self.scene.sim.set_camera_view(eye=[cam_x, cam_y, cam_z], target=[root_pos[0], root_pos[1], 0.5])

        self.total_mean_timesteps += 1

        self._actions = actions.clone()
        self._processed_actions = (self.cfg.action_scale * self._actions + self.cfg.action_offset).clamp(min=0.0, max=1.0)

    def _apply_action(self):
        self._robot.set_joint_position_target(self._processed_actions[:, :self._robot.num_joints])
        self._robot.set_joint_velocity_target(self._processed_actions[:, self._robot.num_joints:])

    def _get_observations(self):
        self._previous_actions = self._actions.clone()
        
        obs = torch.cat(
            [
                self._robot.actuators["base_legs"].muscle_model.lce_tensor, #muscle length
                self._robot.actuators["base_legs"].muscle_model.lce_dot_tensor, #muscle velocity
                self._robot.data.root_lin_vel_b,
                self._robot.data.projected_gravity_b,
                self._robot.data.root_quat_w,
                self._actions,
                self._contact_sensor.data.net_forces_w[:, self._feet_ids, :].view(self.num_envs, -1)
            ],
            dim=-1
        )

        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        tot_rew, rew_dic = compute_rewards(
            self._robot.data.root_lin_vel_b,
            self._robot.data.root_ang_vel_b,
            self._robot.data.root_pos_w,
            self.scene.env_origins,
            self._actions,
            self._previous_actions,
            self.step_dt,
            self.num_envs,
            self.device
        )

        for key, value in rew_dic.items():
            self._episode_sums[key] += value

        return tot_rew
    
    def _get_dones(self):
        dones, time_out = compute_dones(
            self.episode_length_buf,
            self.max_episode_length,
            self._contact_sensor.data.net_forces_w_history,
            self._undesired_contact_body_ids,
            self._robot.data.root_pos_w
        )

        return dones, time_out

    def _reset_idx(self, env_ids):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        self._robot.reset(env_ids)

        super()._reset_idx(env_ids)

        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0

        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_com_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode_Termination/base_contact or root_height"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        extras["Total Timesteps"] = self.total_mean_timesteps
        self.extras["log"].update(extras)

@torch.jit.script
def compute_rewards(
    root_lin_vel_b: torch.Tensor,
    root_ang_vel_b: torch.Tensor,
    root_pos: torch.Tensor,
    env_origins: torch.Tensor,
    actions: torch.Tensor,
    prev_actions: torch.Tensor,
    step_dt: float,
    num_envs: int,
    device: str,  
):
    v_x = root_lin_vel_b[:, 0]  # forward speed
    ang_z = root_ang_vel_b[:, 2]

    #params
    target_v = 5.0
    temp_v = 10
    
    w_forward_gauss = 15.0

    reg_scale_1 = 1.0
    reg_scale_2 = 1.0

    angular_scale = 2.0

    forward_gauss = torch.exp(-((v_x - target_v) ** 2) / temp_v)

    action_regularization_1 = -torch.sum(torch.square(actions), dim=-1)
    action_regularization_2 = -torch.sum(torch.square(actions - prev_actions), dim=-1)

    angular_penalty = -(ang_z ** 2)

    reward = (
            w_forward_gauss * forward_gauss * step_dt
            + reg_scale_1 * action_regularization_1 * step_dt
            + reg_scale_2 * action_regularization_2 * step_dt
            + angular_scale * angular_penalty * step_dt
        )
    
    individual_rewards = {
        "forward_gauss": step_dt * w_forward_gauss * forward_gauss.to(device).reshape(num_envs),
        "action_regularization_1": step_dt * reg_scale_1 * action_regularization_1.to(device).reshape(num_envs),
        "action_regularization_2": step_dt * reg_scale_2 * action_regularization_2.to(device).reshape(num_envs),
        "angular_penalty": step_dt * angular_scale * angular_penalty.to(device).reshape(num_envs)
    }

    return reward, individual_rewards

@torch.jit.script
def compute_dones(
    episode_len_buf: torch.Tensor,
    max_episode_length: int,
    net_contact_forces: torch.Tensor,
    undesired_ids: torch.Tensor,
    root_pos: torch.Tensor
):
    time_out = episode_len_buf >= max_episode_length

    max_contact_forces = torch.max(torch.norm(net_contact_forces[:, :, undesired_ids], dim=-1), dim=1).values
    done_undesired_contacts = torch.any(max_contact_forces > 0.5, dim=1)

    done_root_height_below_minimum = root_pos[:, 2] < 0.2

    dones = done_undesired_contacts | done_root_height_below_minimum

    return dones, time_out
    