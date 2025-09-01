from isaaclab.assets import Articulation
import torch
from ..base_experiment import BaseExperiment
from tqdm import tqdm
from datetime import datetime

class FvmaxExperiment(BaseExperiment):
    def __init__(self, simulation_app, sim, scene, muscle_parameters):
        super().__init__(simulation_app, sim, scene)

        self.muscle_parameters = muscle_parameters
        self.robot: Articulation = self.scene["unitree"]
        self.num_joints = len(self.robot.data.joint_names)
        self.sim_dt = sim.get_physics_dt()

        self.default_activations = torch.tensor([[0.3] * 2 * self.num_joints], device=self.muscle_parameters["device"])
        self.test_activations = torch.tensor([[0.25, 0.75], [0.75, 0.25]], device=self.muscle_parameters["device"])

        step_size = 0.5

        self.fvmax = torch.arange(1.0, 5.0 + step_size, step_size, device=self.muscle_parameters["device"])
        self.vmax = torch.arange(3.0, 9.0 + step_size, step_size, device=self.muscle_parameters["device"])
        
        self.joint_idxs, self.joint_names = self.robot.find_joints(["FL_hip_joint", "RL_hip_joint", "FL_thigh_joint", "RL_thigh_joint", "FL_calf_joint", "RL_calf_joint"])

        self.mode = 0

        self.reset_activations()

        print()

    def reset_activations(self):
        self.activations = self.default_activations.detach().clone()

    def reset_robot(self):
        root_state = self.robot.data.default_root_state.clone()
        self.robot.write_root_pose_to_sim(root_state[:, :7])
        self.robot.write_root_velocity_to_sim(root_state[:, 7:])

        joint_pos = self.robot.data.default_joint_pos.clone()
        joint_vel = self.robot.data.default_joint_vel.clone()

        self.robot.write_joint_state_to_sim(joint_pos, joint_vel)

        self.robot.reset()

        self.run_sim_for_steps(300)

    def run_sim_for_steps(self, steps):
        for _ in range(steps):
            self.robot.set_joint_position_target(self.activations[:, self.num_joints:])
            self.robot.set_joint_velocity_target(self.activations[:, :self.num_joints])
            self.robot.write_data_to_sim()

            self.sim.step()
            self.scene.update(self.sim_dt)

    def run_experiment(self):
        for i, joint_name in tqdm(zip(self.joint_idxs, self.joint_names), desc="Joint Loop", position=0):
            #print("Joint Experiments:", joint_name, i)
            self.reset_robot()

            for fv in tqdm(self.fvmax, desc="Fvmax loop", position=1, leave=False):
                for v in tqdm(self.vmax, desc="Vmax loop", position=2, leave=False):
                    self.robot.actuators["base_legs"].fvmax = fv.item()
                    self.robot.actuators["base_legs"].vmax = v.item()

                    self.robot.actuators["base_legs"].start_logging()

                    for _ in range(10):
                        self.activations[:, [i, i + self.num_joints]] = self.test_activations[self.mode]

                        self.run_sim_for_steps(500)

                        self.mode = 1 - self.mode


                    #print("Saving results to", f"data/vmax_fvmax_tuning/{joint_name}_fvmax_{round(fv.item(), 2)}_vmax_{round(v.item(), 2)}_<curr_time>.pkl")
                    self.robot.actuators['base_legs'].save_logs(f"/home/jan/dev/reflex-controller/data/vmax_fvmax_tuning/{joint_name}_fvmax_{round(fv.item(), 2)}_vmax_{round(v.item(), 2)}_{datetime.now().strftime('%Y-%m-%d-%H-%M')}.pkl")
                    self.robot.actuators['base_legs'].stop_logging()
                    self.robot.actuators['base_legs'].reset_logging()

                    self.reset_activations()
                    self.reset_robot()

        print("Experiment Finished")