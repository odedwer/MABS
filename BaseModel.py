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
        self.machine_reward_counter = np.ones((self.N, *self.rewards.shape))
        self.estimated_machine_reward_distribution = self.machine_reward_counter / (
            self.rewards.size if len(self.rewards.shape) == 1 else self.rewards.shape[1])
        self.different_rewards = len(self.rewards.shape) > 1
        super().__init__()

    @property
    @abstractmethod
    def model_name(self):
        pass

    @abstractmethod
    def choose_machines(self, get_estimates):
        pass

    def update(self, chosen_machines, outcomes):
        if self.different_rewards:
            outcome_indices = np.diag(np.apply_along_axis(np.searchsorted, 1, self.rewards, outcomes))
        else:
            outcome_indices = np.searchsorted(self.rewards, outcomes)
        self.machine_reward_counter[chosen_machines, outcome_indices] += 1
        self.estimated_machine_reward_distribution = self.machine_reward_counter / np.sum(self.machine_reward_counter,
                                                                                          axis=1)[:, np.newaxis]

    def _get_estimated_entropy_gain(self):
        R = self.rewards.shape[1] if self.different_rewards else self.rewards.size
        reward_counters_for_entropy = np.repeat(self.machine_reward_counter, R, 1).reshape((self.N, R, R),
                                                                                           order='F')
        reward_counters_for_entropy[:, np.arange(R), np.arange(R)] += 1
        ent = entropy(reward_counters_for_entropy, axis=2)
        estimated_entropy = np.sum(self.estimated_machine_reward_distribution * ent, axis=1)
        entropy_gain = entropy(self.estimated_machine_reward_distribution, axis=1) - estimated_entropy
        return entropy_gain

    def _get_top_k(self, array):
        return np.flip(array.argsort()[-self.K:])
