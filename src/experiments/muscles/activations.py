from isaaclab.assets import Articulation
import torch
from ..base_experiment import BaseExperiment
from tqdm import tqdm
from datetime import datetime

class ActivationExperiments(BaseExperiment):
    def __init__(self, simulation_app, sim, scene, muscle_parameters):
        super().__init__(simulation_app, sim, scene)
        self.muscle_parameters = muscle_parameters
        self.robot: Articulation = self.scene["unitree"]
        self.num_joints = len(self.robot.data.joint_names)
        self.sim_dt = sim.get_physics_dt()

        self.action_steps = [0.1] #[0.1, 0.05]

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

        self.run_sim_for_s(1)

    def reset_activations(self):
        self.activations = self.default_activations.detach().clone()

    def run_sim_for_s(self, s):
        for _ in range(int(s / self.sim_dt)):
            self.robot.set_joint_position_target(self.activations[:, :self.num_joints]) #flexor input
            self.robot.set_joint_velocity_target(self.activations[:, self.num_joints:]) # extensor input
            self.robot.write_data_to_sim()

            self.sim.step()
            self.scene.update(self.sim_dt)

    def run_experiment(self):
        start_time = datetime.now().strftime('%Y-%m-%d-%H-%M')
        logger = self.robot.actuators['base_legs'].muscle_model.logger

        # check each individual joint
        for i, joint_name in enumerate(self.robot.data.joint_names):
            # try different changings of action steps to see dynamics
            for action_step in tqdm(self.action_steps, desc="Activation Loop", position=0):
                self.reset_activations()
                self.activations[:, [i, i + self.num_joints]] = 0.0

                self.reset_robot()

                logger.start_logging()
                # try all the co-contractions for all steps
                for a1 in tqdm(torch.arange(0.0, 1.0 + action_step, action_step), desc="Flexor actions", position=1, leave=False):
                    for a2 in tqdm(torch.arange(0.0, 1.0 + action_step, action_step), desc="Extensor actions", position=2, leave=False):
                        self.activations[:, i] = a1
                        self.activations[:, i + self.num_joints] = a2
                        
                        self.run_sim_for_s(2.0)

                    logger.pause_logging()
                    self.reset_activations()
                    self.activations[:, [i, i + self.num_joints]] = 0.0

                    self.reset_robot()
                    logger.start_logging()
                    
                logger.save_logs(f"/home/jan/dev/reflex-controller/data/co_contraction_experiment/{start_time}/{joint_name}_{action_step}.pkl")
                logger.pause_logging()
                logger.reset_logging()

        print("Experiment finished")