from __future__ import annotations
import torch

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
        self._connections = torch.tensor(connections, dtype=torch.bool, device=self.device)
        self._num_matrix_parameters = torch.sum(self._connections.triu().int()).item()
        self._num_parameters = 2 * self._num_matrix_parameters + self.num_muscles

        self.muscle_model = MuscleModel(self.num_envs, self.num_joints, lmin, lmax, fvmax, vmax, fpmax, fmin, lce_max, peak_force, self.dt, joint_angle_extrema)

        self._capacity = muscle_delay_s // self.dt # the this will miss 1 step, but since we apply the reflex activations 1 step later, we will get the correct delay again
        self._length_buffer = FiFoRingBufferSimple(self.num_envs, self.num_muscles, self._capacity, device=self.device)
        self._force_buffer = FiFoRingBufferSimple(self.num_envs, self.num_muscles, self._capacity, device=self.device)
        
        self._connection_matrix_L = torch.zeros(self.num_envs, self.num_muscles, self.num_muscles, device=self.device)
        self._connection_matrix_F = torch.zeros(self.num_envs, self.num_muscles, self.num_muscles, device=self.device)
        self._offsets = torch.zeros(self.num_envs, self.num_muscles, device=self.device)

        self._activations = torch.zeros(self.num_envs, self.num_muscles, device=self.device)

        self._count = 0

    def set_parameters(self, parameters: torch.Tensor):
        """Function for setting the parameters of size (n_envs, num_parameters) intro the 3 parameter matrices"""
        if parameters.dim() == 1:
            parameters = parameters.unsqueeze(0)

        upper = self._connections.triu()

        params_L = torch.zeros(self.num_envs, self.num_muscles, self.num_muscles, device=self.device)
        params_R = torch.zeros(self.num_envs, self.num_muscles, self.num_muscles, device=self.device)

        params_L.view(self.num_envs, -1)[:, upper.view(-1)] = parameters[:, :self._num_matrix_parameters]
        params_R.view(self.num_envs, -1)[:, upper.view(-1)] = parameters[:, self._num_matrix_parameters : 2*self._num_matrix_parameters]
        self._offsets = parameters[:, 2*self._num_matrix_parameters :]

        self._connection_matrix_L = params_L + params_L.mT - torch.diag_embed(torch.diagonal(params_L, dim1=-2, dim2=-1))
        self._connection_matrix_R = params_R + params_R.mT - torch.diag_embed(torch.diagonal(params_R, dim1=-2, dim2=-1))



    def set_connection_matrices(self, connection_matrix_L = None, connection_matrix_F = None, offsets = None):
        if connection_matrix_L is not None:
            if connection_matrix_L.shape == self._connection_matrix_L.shape:
                self._connection_matrix_L = connection_matrix_L
            else: 
                raise Exception(f"WARNING: connection_matrix_L has wrong shape. Expected {self._connection_matrix_L.shape}, got {connection_matrix_L.shape}")

        if connection_matrix_F is not None:
            if connection_matrix_F.shape == self._connection_matrix_F.shape:
                self._connection_matrix_F = connection_matrix_F
            else:
                raise Exception(f"WARNING: connection_matrix_F has wrong shape. Expected {self._connection_matrix_F.shape}, got {connection_matrix_F.shape}")

        if offsets is not None:
            if offsets.shape == self._offsets.shape:
                self._offsets = offsets
            else:
                raise Exception(f"WARNING: connection_matrix_L has wrong shape. Expected {self._offsets.shape}, got {offsets.shape}")
    
    def get_parameters(self):
        """returns tuple: (offset_vectir, F matrix, L matrix)"""
        return self._offsets, self._connection_matrix_F, self._connection_matrix_L
    
    def check_waiting(self):
        """Function, that in the beginning creates the delay for the reflex controler"""
        if self._count < self._capacity:
            self._count += 1
            return True
        
        return False

    def compute(self, joint_pos, joint_vel):
        # call this function to update muscle lengths and muscle_forces and to return torques
        torques = self.muscle_model.compute(self._activations, joint_pos, joint_vel)

        with torch.no_grad():
            if self.check_waiting():
                self._length_buffer.append(range(self.num_envs), self.muscle_model.lce_tensor)
                self._force_buffer.append(range(self.num_envs), self.muscle_model.force_tensor)
                return torch.zeros(self.num_envs, self.num_joints, device=self.device)

            muscle_lengths = self._length_buffer.pop(range(self.num_envs))
            muscle_forces = self._force_buffer.pop(range(self.num_envs))

            self._length_buffer.append(range(self.num_envs), self.muscle_model.lce_tensor)
            self._force_buffer.append(range(self.num_envs), self.muscle_model.force_tensor)

            self._activations = self._offsets + (self._connection_matrix_L @ muscle_lengths.unsqueeze(-1) + self._connection_matrix_F @ muscle_forces.unsqueeze(-1)).squeeze(-1)
            self._activations.clip_(min=0.0, max=1.0)
        return torques
    
    def reset(self):
        self._length_buffer.clear()
        self._force_buffer.clear()
        self._connection_matrix_F.zero_()
        self._connection_matrix_L.zero_()
        self._offsets.zero_()
        self._count = 0
        self.muscle_model.reset()