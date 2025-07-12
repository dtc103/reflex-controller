from abc import ABC, abstractmethod

class BaseExperiment(ABC):
    def __init__(self, simulation_app, sim, scene):
        self.simulation_app = simulation_app
        self.sim = sim
        self.scene = scene

    @abstractmethod
    def run_experiment(self):
        pass

    