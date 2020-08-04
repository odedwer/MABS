from abc import ABC, abstractmethod

import numpy as np
from numpy import asarray, sort
from scipy.stats import entropy


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
    def choose_machines(self):
        pass

    @abstractmethod
    def update(self, chosen_machines, outcomes):
        pass

    def _get_estimated_entropy(self):
        R = self.rewards.size
        reward_counters_for_entropy = np.repeat(self.machine_reward_counter, R, 1).reshape((self.N, R, R), order='F')
        reward_counters_for_entropy[:, np.arange(R), np.arange(R)] += 1
        ent = entropy(reward_counters_for_entropy, axis=2)
        estimated_entropy = np.sum(self.estimated_machine_reward_distribution * ent, axis=1)
        entropy_gain = entropy(self.estimated_machine_reward_distribution, axis=1) - estimated_entropy
        return entropy_gain
