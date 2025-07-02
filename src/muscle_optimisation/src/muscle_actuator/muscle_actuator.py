from __future__ import annotations
import torch
from typing import TYPE_CHECKING


from isaaclab.utils.types import ArticulationActions

from isaaclab.actuators import ActuatorBase
if TYPE_CHECKING:
    from .muscle_actuator_cfg import MuscleActuatorCfg

from scipy.optimize import bisect

class MuscleActuator(ActuatorBase):
    cfg: MuscleActuatorCfg

    def __init__(self, cfg: MuscleActuatorCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)

        MuscleActuator.is_implicit_model = False
        self.muscle_params = cfg.muscle_params

        for key, value in self.muscle_params.items():
            setattr(self, key, value)

        self.phi_min = torch.tensor(
            self.angles[:, 0].clone().detach(), 
            device=self.device, 
            requires_grad=False
        )
        
        self.phi_max = torch.tensor(
            self.angles[:, 1].clone().detach(), 
            device=self.device, 
            requires_grad=False
        )

        self.activation_tensor = torch.zeros(
            (self._num_envs, self.num_joints * 2),
            dtype=torch.float32,
            device=self.device,
            requires_grad=False,
        )

        self.lce_tensor = torch.zeros(
            (self._num_envs, self.num_joints * 2),
            dtype=torch.float32,
            device=self.device,
            requires_grad=False,
        )
        self.lce_dot_tensor = torch.zeros(
            (self._num_envs, self.num_joints * 2),
            dtype=torch.float32,
            device=self.device,
            requires_grad=False,
        )
        self.force_tensor = torch.zeros(
            (self._num_envs, self.num_joints * 2),
            dtype=torch.float32,
            device=self.device,
            requires_grad=False,
        )

        self.effort_limit = self.peak_force

        self.moment, self.lce_ref = self._compute_parametrization()
        

    def _FL(self, lce: torch.Tensor) -> torch.Tensor:
        """
        Force length
        """
        length = lce
        b1 = self._bump(length, self.lmin, 1, self.lmax)
        b2 = self._bump(length, self.lmin, 0.5 * (self.lmin + 0.95), 0.95)
        bump_res = b1 + 0.15 * b2
        return bump_res
    
    def _calc_l_min(self, Fmin, tol=10e-7):
        def f(l_ce):
            return self._FL(torch.tensor([l_ce], device=self.device)) - Fmin
        
        mid = 1.0
        return bisect(lambda x: f(x), self.lmin + 10e-9, mid - 10e-9, xtol=tol)

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
            requires_grad=False,
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
            requires_grad=False,
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
        c_result = torch.zeros_like(lce, dtype=torch.float32, device=self.device, requires_grad=False)
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

        self.lce_min = self._calc_l_min(self.fmin)

        moment = torch.zeros((self._num_envs, self.num_joints * 2), device=self.device)
        moment[:, : int(moment.shape[1] // 2)] = (self.lce_max - self.lce_min) / (self.phi_max - self.phi_min)
        moment[:, int(moment.shape[1] // 2) :] = (self.lce_max - self.lce_min) / (self.phi_min - self.phi_max)
        
        lce_ref = torch.zeros((self._num_envs, self.num_joints * 2), device=self.device)
        lce_ref[:, : int(lce_ref.shape[1] // 2)] = self.lce_min - moment[:, : int(moment.shape[1] // 2)].squeeze() * self.phi_min
        lce_ref[:, int(lce_ref.shape[1] // 2) :] = self.lce_min - moment[:, int(moment.shape[1] // 2) :].squeeze() * self.phi_max
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
        #print("FL", FL)
        FV = 1.0 #self._FV(lce_dot)
        FP = 0.0 #self._FP(lce_tensor)

        self.force_tensor = self.activation_tensor * FL * FV + FP
        real_force = self.peak_force * self.force_tensor
        torque = real_force * self.moment

        return torch.sum(
            torch.reshape(torque, (self._num_envs, 2, self.num_joints)),
            axis=-2,
        )

    def compute(self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel) -> ArticulationActions:
        """
        actuator_pos: Current position of actuator
        """

        # since isaac lab does not allow the definition of own input items, we just 
        # act like the inputted control_action.joint_positions contain the muscle activations for one muscle of each joint
        # and control_action.joint_velocities the activations of the other muscles of the joint
        muscle_activations_1 = control_action.joint_positions
        muscle_activations_2 = control_action.joint_velocities
        muscle_activations = torch.concatenate([muscle_activations_1, muscle_activations_2], dim=1)

        with torch.no_grad():
            actions = torch.clip(muscle_activations, 0, 1)

            # activity funciton for muscle activation
            self._activ_dyn(actions)

            # update virtual lengths
            self._compute_virtual_lengths(joint_pos)

            # compute moments
            moment = self._compute_moment(joint_vel)

            control_action.joint_efforts = moment
            control_action.joint_positions = None
            control_action.joint_velocities = None

            print(moment)

            # self.computed_effort = 
            # self.applied_effort = self._clip_effort()

        return control_action

    def reset(self, *args, **kwargs):
        pass
