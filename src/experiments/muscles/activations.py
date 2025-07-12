from isaaclab.assets import Articulation
import torch
from ..base_experiment import BaseExperiment

class ActivationExperiments(BaseExperiment):
    def __init__(self, simulation_app, sim, scene, muscle_parameters):
        super().__init__(simulation_app, sim, scene)
        self.muscle_parameters = muscle_parameters
        self.robot: Articulation = self.scene["unitree"]
        self.num_joints = len(self.robot.data.joint_names)
        self.sim_dt = sim.get_physics_dt()

    def reset_robot(self, activations):
        root_state = self.robot.data.default_root_state.clone()
        self.robot.write_root_pose_to_sim(root_state[:, :7])
        self.robot.write_root_velocity_to_sim(root_state[:, 7:])

        joint_pos = self.robot.data.default_joint_pos.clone()
        joint_vel = self.robot.data.default_joint_vel.clone()

        self.robot.write_joint_state_to_sim(joint_pos, joint_vel)

        self.robot.reset()

        for _ in range(300):
            self.robot.set_joint_position_target(activations[:, self.num_joints:])
            self.robot.set_joint_velocity_target(activations[:, :self.num_joints])
            self.robot.write_data_to_sim()

            self.sim.step()
            self.scene.update(self.sim_dt)

    def run_experiment(self):
        action_steps = [0.2, 0.1, 0.05, 0.01]

        default_activations = torch.tensor([[0.3] * 2 * self.num_joints], device=self.muscle_parameters["device"])

        # check each individual joint
        for i, joint_name in enumerate(self.robot.data.joint_names):
            print("Joint experiments:", joint_name)
            # try different changings of action steps to see dynamics
            for action_step in action_steps:
                print("Action Step:", action_step)

                print("Reset before new step...")
                activations = default_activations.clone()
                activations[:, [i, i + self.num_joints]] = 0.0

                self.reset_robot(activations)

                self.robot.actuators['base_legs'].start_logging()
                # try all the co-contractions for all steps
                for a1 in torch.arange(0.0, 1.0 + action_step, action_step):
                    for a2 in torch.arange(0.0, 1.0 + action_step, action_step):
                        count = 1
                        activations[:, i] = a2 #extensor
                        activations[:, i + self.num_joints] = a1 #flexor
                        
                        # give the simulation some time to reach a final state
                        while self.simulation_app.is_running:
                            if count % 300 == 0:
                                break
                                
                            self.robot.set_joint_position_target(activations[:, self.num_joints:])
                            self.robot.set_joint_velocity_target(activations[:, :self.num_joints])
                            self.robot.write_data_to_sim()

                            self.sim.step()
                            self.scene.update(self.sim_dt)

                            count += 1
                    
                    print("Reset for new activation loop...")
                    activations = default_activations.clone()
                    activations[:, [i, i + self.num_joints]] = 0.0

                    self.reset_robot(activations)
                    
                print("Saving results to", f"data/co_contraction_experiment/{joint_name}_{action_step}.pkl")
                self.robot.actuators['base_legs'].save_logs(f"data/co_contraction_experiment/{joint_name}_{action_step}.pkl")
                self.robot.actuators['base_legs'].stop_logging()
                self.robot.actuators['base_legs'].reset_logging()

        print("Experiment finished")