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
        
        self.muscle_model = MuscleModel(num_envs, num_joints, lmin, lmax, fvmax, vmax, fpmax, fmin, lce_max, peak_force, sim_dt, joint_angle_extrema)

        self.num_joints = num_joints
        self.num_muscles = 2 * num_joints

        capacity = muscle_delay_s // self.dt # the this will miss 1 step, but since we apply the reflex activations 1 step later, we will get the correct delay again
        self.length_buffer = FiFoRingBufferSimple(num_envs, self.num_muscles, capacity, device=self.device)
        self.force_buffer = FiFoRingBufferSimple(num_envs, self.num_muscles, capacity, device=self.device)
        
        self.connection_matrix_L = torch.diag(torch.full(self.num_muscles), device=self.device)
        self.connection_matrix_F = torch.diag(torch.zeros(self.num_muscles), device=self.device)
        self.offsets = torch.zeros(self.num_muscles, device=self.device)

        

    def update_activation(self, joint_pos, joint_vel):
        pass


    


