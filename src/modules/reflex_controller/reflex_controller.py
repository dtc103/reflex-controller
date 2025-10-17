from __future__ import annotations
import torch

from ..muscle_model import MuscleModel
from ..utils.ring_buffer import FiFoRingBuffer

class ReflexController:
    def __init__(self, num_envs, num_joints, connection, muscle_delays, lmin, lmax, fvmax, vmax, fpmax, fmin, lce_max, peak_force, sim_dt, joint_angle_extrema, device="cuda:0"):
        """
        input of delays for each 
        """
        self.device = device
        self.dt = sim_dt
        
        self.muscle_model = MuscleModel(num_envs, num_joints, lmin, lmax, fvmax, vmax, fpmax, fmin, lce_max, peak_force, sim_dt, joint_angle_extrema)

        self.num_joints = num_joints
        self.num_muscles = 2 * num_joints

        capacity = [] #<- need to find suitable capacity values for all connection  
        self.length_buffer = FiFoRingBuffer(num_envs, capacity, self.num_muscles, device=self.device)
        self.force_buffer = FiFoRingBuffer(num_envs, capacity, self.num_muscles, device=self.device)
        
        self.connection_matrix_L = torch.diag(torch.ones(self.num_muscles), device=self.device)
        self_connection_matrix_F = torch.diag(torch.ones(self.num_muscles), device=self.device)
        self.offsets = torch.zeros(self.num_muscles, device=self.device)


