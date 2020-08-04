from abc import ABC, abstractmethod

import numpy as np
from numpy import asarray, sort


class BaseModel(ABC):
    def __init__(self, machines, num_to_choose: int, num_trials: int, possible_rewards: set):
        self.machines = asarray(machines)
        self.N = machines.size
        self.K = num_to_choose
        self.T = num_trials
        self.rewards = sort(asarray(possible_rewards))
        self.machine_reward_counter = np.ones((self.N, self.rewards.size))
        self.estimated_machine_reward_distribution = self.machine_reward_counter / self.rewards.size
        super().__init__()

    @abstractmethod
    def choose_machines(self, get_estimates):
        pass

    @abstractmethod
    def update(self, chosen_machines, outcomes):
        pass
