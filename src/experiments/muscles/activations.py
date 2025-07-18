from isaaclab.assets import Articulation
import torch
from ..base_experiment import BaseExperiment
from tqdm import tqdm

class ActivationExperiments(BaseExperiment):
    def __init__(self, simulation_app, sim, scene, muscle_parameters):
        super().__init__(simulation_app, sim, scene)
        self.muscle_parameters = muscle_parameters
        self.robot: Articulation = self.scene["unitree"]
        self.num_joints = len(self.robot.data.joint_names)
        self.sim_dt = sim.get_physics_dt()

        self.action_steps = [0.1, 0.05]

        self.default_activations = torch.tensor([[0.3] * 2 * self.num_joints], device=self.muscle_parameters["device"])
        self.reset_activations()

    def reset_robot(self):
        root_state = self.robot.data.default_root_state.clone()
        self.robot.write_root_pose_to_sim(root_state[:, :7])
        self.robot.write_root_velocity_to_sim(root_state[:, 7:])

        joint_pos = self.robot.data.default_joint_pos.clone()
        joint_vel = self.robot.data.default_joint_vel.clone()

        self.robot.write_joint_state_to_sim(joint_pos, joint_vel)

        self.robot.reset()

        self.run_sim_for_steps(500)

    def reset_activations(self):
        self.activations = self.default_activations.detach().clone()

    def run_sim_for_steps(self, steps):
        for _ in range(steps):
            self.robot.set_joint_position_target(self.activations[:, self.num_joints:])
            self.robot.set_joint_velocity_target(self.activations[:, :self.num_joints])
            self.robot.write_data_to_sim()

            self.sim.step()
            self.scene.update(self.sim_dt)

    def run_experiment(self):
        # check each individual joint
        for i, joint_name in enumerate(self.robot.data.joint_names):
            print("Joint experiments:", joint_name)
            # try different changings of action steps to see dynamics

            for action_step in tqdm(self.action_steps, desc="Joint Loop", position=0):
                self.reset_activations()
                self.activations[:, [i, i + self.num_joints]] = 0.0

                self.reset_robot()

                self.robot.actuators['base_legs'].start_logging()
                # try all the co-contractions for all steps
                for a1 in tqdm(torch.arange(0.0, 1.0 + action_step, action_step), desc="Flexor actions", position=1, leave=False):
                    for a2 in tqdm(torch.arange(0.0, 1.0 + action_step, action_step), desc="Extensor actions", position=2, leave=False):
                        self.activations[:, i] = a2 # extensor
                        self.activations[:, i + self.num_joints] = a1 # flexor
                        
                        self.run_sim_for_steps(300)

                    self.robot.actuators['base_legs'].pause_logging()
                    self.reset_activations()
                    self.activations[:, [i, i + self.num_joints]] = 0.0

                    self.reset_robot()
                    self.robot.actuators['base_legs'].continue_logging()
                    
                self.robot.actuators['base_legs'].save_logs(f"/home/jan/dev/reflex-controller/data/co_contraction_experiment/{joint_name}_{action_step}.pkl")
                self.robot.actuators['base_legs'].pause_logging()
                self.robot.actuators['base_legs'].reset_logging()

        print("Experiment finished")