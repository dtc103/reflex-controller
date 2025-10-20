from __future__ import annotations
import torch
import math

from ..muscle_model import MuscleModel
from ..utils.ring_buffer import FiFoRingBufferSimple

class ReflexController:
    def __init__(self, num_envs, num_joints, connections, muscle_delay_s, lmin, lmax, fvmax, vmax, fpmax, fmin, lce_max, peak_force, sim_dt, joint_angle_extrema, device="cuda:0"):
        """
        input of delays for each 
        Args:
            connections: 
                connection matrix. Has shape (n_muscles, n_muscles)
            muscle_delay: 
                muscle delay in s
        """
        self.device = device
        self.dt = sim_dt
        self.num_envs = num_envs
        self.num_joints = num_joints
        self.num_muscles = 2 * num_joints
        self.connections = connections

        self.muscle_model = MuscleModel(self.num_envs, self.num_joints, lmin, lmax, fvmax, vmax, fpmax, fmin, lce_max, peak_force, self.dt, joint_angle_extrema)

        capacity = muscle_delay_s // self.dt # the this will miss 1 step, but since we apply the reflex activations 1 step later, we will get the correct delay again
        self._length_buffer = FiFoRingBufferSimple(self.num_envs, self.num_muscles, capacity, device=self.device)
        self._force_buffer = FiFoRingBufferSimple(self.num_envs, self.num_muscles, capacity, device=self.device)
        
        self._connection_matrix_L = torch.diag(torch.full(self.num_muscles), device=self.device)
        self._connection_matrix_F = torch.diag(torch.zeros(self.num_muscles), device=self.device)
        self._offsets = torch.zeros(self.num_muscles, device=self.device)

        self._activations = torch.zeros(self.num_envs, self.num_muscles)

        

    def compute(self, joint_pos, joint_vel):
        torques = self.muscle_model.compute(self._activations, joint_pos, joint_vel)

        with torch.no_grad:

            muscle_lengths = self._length_buffer.pop(range(self.num_envs))
            muscle_forces = self._force_buffer.pop(range(self.num_envs))

            self._length_buffer.append(range(self.num_envs), self.muscle_model.lce_tensor)
            self._force_buffer.appen(range(self.num_envs), self.muscle_model.force_tensor)


            self._activations = self.offsets





    


