from abc import ABC, abstractmethod
from numpy import asarray


class BaseModel(ABC):
    def __init__(self, machines, num_to_choose: int, num_trials: int, possible_rewards: set):
        self.machines = asarray(machines)
        self.N = machines.size
        self.K = num_to_choose
        self.T = num_trials
        self.rewards = asarray(possible_rewards)
        super().__init__()

    @abstractmethod
    def choose_machines(self):
        pass

    @abstractmethod
    def update(self, chosen_machines, outcomes):
        pass
