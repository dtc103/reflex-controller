from __future__ import annotations

import torch
from scipy.optimize import bisect
from ..utils.logger.logger import Logger

class MuscleModel:
    def __init__(
        self, 
        num_envs: int, 
        num_joints: int, 
        lmin: float, 
        lmax: float, 
        fvmax: float, 
        vmax: float, 
        fpmax: float, 
        fmin: float, 
        lce_max: float, 
        peak_force: float, 
        sim_dt: float, 
        joint_angle_extrema: torch.Tensor, 
        device: str
    ):
        #self.logger = Logger()

        self.num_envs = num_envs
        self.num_joints = num_joints
        self.lmin = lmin
        self.lmax = lmax
        self.fvmax = fvmax
        self.vmax = vmax
        self.fpmax = fpmax
        self.fmin = fmin
        self.lce_max = lce_max
        self.peak_force = peak_force
        self.dt = sim_dt
        self.device = device
        
        self.phi_min = joint_angle_extrema.detach().clone().requires_grad_(False)[:, 0]
        self.phi_max = joint_angle_extrema.detach().clone().requires_grad_(False)[:, 1]

        self.activation_tensor = torch.zeros(
            (self.num_envs, self.num_joints * 2),
            dtype=torch.float32,
            device=self.device,
        )

        self.lce_dot_tensor = torch.zeros(
            (self.num_envs, self.num_joints * 2),
            dtype=torch.float32,
            device=self.device,
        )

        self.force_tensor = torch.zeros(
            (self.num_envs, self.num_joints * 2),
            dtype=torch.float32,
            device=self.device,
        )

        self.lce_tensor = torch.zeros(
            (self.num_envs, self.num_joints * 2),
            dtype=torch.float32,
            device=self.device,
        )

        self.lce_min = self._calc_l_min(self.fmin, 10e-7, 128)
        self.moment, self.lce_ref = self._compute_parametrization()

    def _FL(self, lce: torch.Tensor) -> torch.Tensor:
        """
        Force length
        """
        length = lce
        b1 = self._bump(length, self.lmin, 1.0, self.lmax)
        b2 = self._bump(length, self.lmin, 0.5 * (self.lmin + 0.95), 0.95)
        bump_res = b1 + 0.15 * b2
        return bump_res

    def _calc_l_min(self, target_force: float, tol: float = 10e-7, max_iter: int = 128) -> float:
        lower = self.lmin + 1.0e-8
        upper = 1.0 - 1.0e-8
        for _ in range(max_iter):
            mid = 0.5 * (lower + upper)
            fl_mid = float(self._FL(torch.tensor([mid], device=self.device))[0].item())
            diff = fl_mid - target_force
            if diff > 0.0:
                upper = mid
            else:
                lower = mid
            if abs(diff) < tol:
                break
        return 0.5 * (lower + upper)

    def _bump(self, length: torch.Tensor, A: float, mid: float, B: float) -> torch.Tensor:
        """
        skewed bump function: quadratic spline
        Input:
            :length: tensor of muscle lengths [Nenv, Nactuator]
            :A: scalar
            :mid: scalar
            :B: scalar

        Returns:
            :torch.Tensor: contains FV result [Nenv, Nactuator]
        """
        left = 0.5 * (A + mid)
        right = 0.5 * (mid + B)
        # Order of assignment needs to be inverse to the if-else-clause case
        bump_result = torch.ones_like(
            length,
            dtype=torch.float32,
            device=self.device,
        )

        x = (B - length) / (B - right)
        bump_result = 0.5 * x**2

        x = (length - mid) / (right - mid)
        bump_result = torch.where(length < right, 1 - 0.5 * x**2, bump_result)

        x = (mid - length) / (mid - left)
        bump_result = torch.where(length < mid, 1 - 0.5 * x**2, bump_result)

        x = (length - A) / (left - A)
        bump_result = torch.where((length < left) & (length > A), 0.5 * x**2, bump_result)

        bump_result = torch.where(
            torch.logical_or((length <= A), (length >= B)),
            torch.tensor([0], dtype=torch.float32, device=self.device),
            bump_result,
        )
        return bump_result
    
    def _FV(self, lce_dot: torch.Tensor) -> torch.Tensor:
        """
        Force velocity
        Input:
            :lce_dot: tensor of muscle velocities [Nenv, Nactuator]
        """
        c = self.fvmax - 1
        velocity = lce_dot

        eff_vel = velocity / self.vmax

        c_result = torch.zeros_like(
            eff_vel,
            dtype=torch.float32,
            device=self.device,
        )

        c_result = torch.where(
            (eff_vel > c),
            torch.tensor([self.fvmax], dtype=torch.float32, device=self.device),
            c_result,
        )

        x = self.fvmax - ((c - eff_vel) ** 2 / c)
        c_result = torch.where(eff_vel <= c, x, c_result)

        x = (eff_vel + 1) ** 2
        c_result = torch.where(eff_vel <= 0, x, c_result)

        c_result = torch.where(
            (eff_vel < -1),
            torch.tensor([0], dtype=torch.float32, device=self.device),
            c_result,
        )

        return c_result
    
    def _FP(self, lce: torch.Tensor) -> torch.Tensor:
        """
        Force passive
        Inputs:
            :lce: muscle lengths [Nenv, Nactuator]
        return :fp_result: passive_force [Nenv, Nactuator]
        """
        b = 0.5 * (self.lmax + 1)

        # Order of assignment needs to be inverse to the if-else-clause case
        ## method to prevent
        cond_2_tmp = (lce - 1) / (b - 1)
        cond_2 = (cond_2_tmp ** 3) * (0.25 * self.fpmax)

        cond_3_tmp = (lce - b) / (b - 1)
        cond_3 = (cond_3_tmp * 3 + 1) * (0.25 * self.fpmax)
        
        ##### copy based on condition the correct output into new tensor
        c_result = torch.zeros_like(lce, dtype=torch.float32, device=self.device)
        c_result = torch.where(lce <= b, cond_2, c_result)
        c_result = torch.where(
            lce <= 1,
            torch.tensor([0], dtype=torch.float32, device=self.device),
            c_result,
        )
        c_result = torch.where(lce > b, cond_3, c_result)

        return c_result
    
    def _compute_parametrization(self):
        """
        Find parameters for muscle length computation.
        This should really only be done once...

        We compute them as one long vector now.
        """

        moment = torch.zeros((self.num_envs, self.num_joints * 2), device=self.device)
        moment[:, int(moment.shape[1] // 2):] = (self.lce_max - self.lce_min) / (self.phi_max - self.phi_min)
        moment[:, :int(moment.shape[1] // 2)] = (self.lce_max - self.lce_min) / (self.phi_min - self.phi_max)
        
        lce_ref = torch.zeros((self.num_envs, self.num_joints * 2), device=self.device)
        lce_ref[:, int(lce_ref.shape[1] // 2):] = self.lce_min - moment[:, int(moment.shape[1] // 2):].squeeze() * self.phi_min
        lce_ref[:, :int(lce_ref.shape[1] // 2)] = self.lce_min - moment[:, :int(moment.shape[1] // 2) :].squeeze() * self.phi_max
        return moment, lce_ref

    def _compute_virtual_lengths(self, actuator_pos: torch.Tensor) -> None:
        """
        Compute muscle fiber lengths l_ce depending on joint angle
        Attention: The mapping of actuator_trnid to qvel is only 1:1 because we have only
        slide or hinge joints and no free joint!!! Otherwise you have to build this mapping
        by looking at every single joint type.

        self.lce_x_tensor contains copy of given actions (they represent the actuator position)
        """
        # Repeat position tensor twice, because both muscles are computed from the same actuator position
        # the operation is NOT applied in-place to the original tensor, only the result is repeated.
        self.lce_tensor = actuator_pos.repeat(1, 2) * self.moment + self.lce_ref

    def _get_vel(self, moment, actuator_vel: torch.Tensor) -> torch.Tensor:
        """
        For muscle 1, the joint angle increases if it pulls. This means
        that the joint and the muscle velocity have opposite signs. But this is already
        included in the value of the moment arm. So we don't need if/else branches here.
        Attention: The mapping of actuator_trnid to qvel is only 1:1 because we have only
        slide or hinge joints and no free joint!!! Otherwise you have to build this mapping
        by looking at every single joint type.
        """
        return actuator_vel.repeat(1, 2) * moment

    def _activ_dyn(self, actions: torch.Tensor) -> None:
        """
        Activity and controls have to be written inside userdata. Assume
        two virtual muscles per real mujoco actuator and let's roll.
        """
        self.activation_tensor = 100 * (actions - self.activation_tensor) * self.dt + self.activation_tensor
        self.activation_tensor = torch.clip(self.activation_tensor, 0, 1)

    def _compute_moment(self, actuator_vel):
        """
        Joint moments are computed from muscle contractions and then returned
        """
        self.lce_dot_tensor = self._get_vel(self.moment, actuator_vel)
        lce_dot = self.lce_dot_tensor
        lce_tensor = self.lce_tensor

        FL = self._FL(lce_tensor)
        FV = self._FV(lce_dot)
        FP = self._FP(lce_tensor)

        self.force_tensor = self.activation_tensor * FL * FV + FP
        real_force = self.peak_force * self.force_tensor
        # due to the coordinate system in IsaacLab, we need to put the -1 here
        torque = -1 * real_force * self.moment

        moment = torch.sum(
            torch.reshape(torque, (self.num_envs, 2, self.num_joints)),
            dim=-2,
        )

        # if self.logger.is_logging:
        #     self.logger.log("FL", FL.detach().cpu().numpy())
        #     self.logger.log("FV", FP.detach().cpu().numpy())
        #     self.logger.log("lce_tensor", self.lce_tensor.detach().cpu().numpy())
        #     self.logger.log("lce_dot_tensor", self.lce_dot_tensor.detach().cpu().numpy())
        #     self.logger.log("normalized_muscle_force", self.force_tensor.detach().cpu().numpy())
        #     self.logger.log("individual_muscle_torque", torque.detach().cpu().numpy())
        #     self.logger.log("joint_torque", moment.detach().cpu().numpy())

        return moment
    
    def compute(self, muscle_activations, joint_pos, joint_vel):
        with torch.no_grad():
            clipped_muscle_activations = torch.clip(muscle_activations, min=0.0, max=1.0)

            self._activ_dyn(clipped_muscle_activations)
            self._compute_virtual_lengths(joint_pos)
            moment = self._compute_moment(joint_vel)

            # if self.logger.is_logging:
            #     self.logger.log("joint_position", joint_pos.detach().cpu().numpy())
            #     self.logger.log("joint_velocity", joint_vel.detach().cpu().numpy())
            #     self.logger.log("clipped_activations", muscle_activations.detach().cpu().numpy())
            #     self.logger.log("raw_activations", clipped_muscle_activations.detach().cpu().numpy())

            return moment
        
    def reset(self):
        self.activation_tensor = torch.zeros(
            (self.num_envs, self.num_joints * 2),
            dtype=torch.float32,
            device=self.device,
        )

        self.lce_dot_tensor = torch.zeros(
            (self.num_envs, self.num_joints * 2),
            dtype=torch.float32,
            device=self.device,
        )

        self.force_tensor = torch.zeros(
            (self.num_envs, self.num_joints * 2),
            dtype=torch.float32,
            device=self.device,
        )

        self.moment, self.lce_ref = self._compute_parametrization()
