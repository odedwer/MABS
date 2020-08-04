import numpy as np
from scipy.stats import entropy

from BaseModel import BaseModel


class UCBNormalModel(BaseModel):
    def __init__(self, machines, num_to_choose: int, num_trials: int, possible_rewards):
        super().__init__(machines, num_to_choose, num_trials, possible_rewards)
        self.estimated_machine_ucb = np.zeros((self.N,))
        self.num_of_plays = 0

    def choose_machines(self) -> np.array:
        # choose K machines with largest UCB
        return np.flip(self.estimated_machine_ucb.argsort()[-self.K:])

    def update(self, chosen_machines, outcomes):
        self.num_of_plays += self.K
        for machine_index, machine in enumerate(self.machines):
            # update UCB of all machines
            self.estimated_machine_ucb[machine_index] = self._get_ucb(machine)
        outcome_indices = np.searchsorted(self.rewards, outcomes)
        self.machine_reward_counter[chosen_machines, outcome_indices] += 1
        self.estimated_machine_reward_distribution = self.machine_reward_counter / np.sum(self.machine_reward_counter,
                                                                                          axis=1)[:, np.newaxis]

    def _get_ucb(self, machine):
        confidence = (2 * np.log(self.num_of_plays)) / machine.num_of_plays
        if confidence == -np.inf or confidence == np.NaN or confidence < 0:  # in case of division by 0
            confidence = 0
        return machine.get_mean_reward() + confidence


class ThompsonNormalModel(BaseModel):
    def __init__(self, machines, num_to_choose: int, num_trials: int, possible_rewards):
        super().__init__(machines, num_to_choose, num_trials, possible_rewards)

    def choose_machines(self):
        estimated_reward_probabilities = self._vectorized_dirichlet_sample()
        estimated_rewards = estimated_reward_probabilities @ self.rewards
        return np.flip(estimated_rewards.argsort()[-self.K:])

    def update(self, chosen_machines, outcomes):
        outcome_indices = np.searchsorted(self.rewards, outcomes)
        self.machine_reward_counter[chosen_machines, outcome_indices] += 1
        self.estimated_machine_reward_distribution = self.machine_reward_counter / np.sum(self.machine_reward_counter,
                                                                                          axis=1)[:, np.newaxis]

    def _vectorized_dirichlet_sample(self):
        """
        Generate samples from an array of alpha distributions.
        from https://stackoverflow.com/questions/15915446/why-does-numpy-random-dirichlet-not-accept-multidimensional-arrays
        """
        r = np.random.standard_gamma(self.machine_reward_counter)
        return r / r.sum(-1, keepdims=True)


class ThompsonEntropyModel(ThompsonNormalModel):
    def __init__(self, machines, num_to_choose: int, num_trials: int, possible_rewards):
        super().__init__(machines, num_to_choose, num_trials, possible_rewards)

    def choose_machines(self):
        estimated_reward_probabilities = self._vectorized_dirichlet_sample()
        estimated_rewards = estimated_reward_probabilities @ self.rewards
        return np.flip(estimated_rewards.argsort()[-self.K:])

    def _vectorized_dirichlet_sample(self):
        """
        Generate samples from an array of reward counters per machine
        from https://stackoverflow.com/questions/15915446/why-does-numpy-random-dirichlet-not-accept-multidimensional-arrays
        """
        r = np.random.standard_gamma(
            self.machine_reward_counter * (
                    1. / entropy(self.estimated_machine_reward_distribution, axis=1)[:, np.newaxis]))
        return r / r.sum(-1, keepdims=True)


class UCBEntropyModel(BaseModel):
    def __init__(self, machines, num_to_choose: int, num_trials: int, possible_rewards):
        super().__init__(machines, num_to_choose, num_trials, possible_rewards)
        self.machine_reward_counter = np.ones((self.N, self.rewards.size))
        self.estimated_machine_reward_distribution = self.machine_reward_counter / self.rewards.size
        self.estimated_machine_ucb = (self.estimated_machine_reward_distribution @ self.rewards) * (1. / entropy(
            self.estimated_machine_reward_distribution, axis=1))

    def choose_machines(self) -> np.array:
        # choose K machines with largest UCB
        return np.flip(self.estimated_machine_ucb.argsort()[-self.K:])

    def update(self, chosen_machines, outcomes):
        # update probabilities as simple frequency counters, where all counters are initialized at 1
        # basically - this is the mode of the Dirichlet conjugate prior
        outcome_indices = np.searchsorted(self.rewards,
                                          outcomes)  # find the indices of the outcomes in the reward array
        self.machine_reward_counter[chosen_machines, outcome_indices] += 1
        self.estimated_machine_reward_distribution = self.machine_reward_counter / np.sum(self.machine_reward_counter,
                                                                                          axis=1)[:, np.newaxis]
        # update mean reward of every chosen machine
        for machine_index in chosen_machines:
            self.estimated_machine_ucb[machine_index] = self.machines[machine_index].get_mean_reward()

        # add entropy for all chosen machines - the higher the entropy, the higher our uncertainty
        # so we are optimistic in the face of uncertainty
        self.estimated_machine_ucb[chosen_machines] = (self.estimated_machine_reward_distribution[chosen_machines, :] @
                                                       self.rewards) * (1. / entropy(
            self.estimated_machine_reward_distribution[chosen_machines, :], axis=1))


class UCBEntropyNormalizedModel(BaseModel):
    def __init__(self, machines, num_to_choose: int, num_trials: int, possible_rewards):
        super().__init__(machines, num_to_choose, num_trials, possible_rewards)
        self.estimated_machine_expectancy = self.estimated_machine_reward_distribution @ self.rewards
        self.estimated_entropy = entropy(self.estimated_machine_reward_distribution, axis=1)

    def choose_machines(self) -> np.array:
        # choose K machines with largest UCB
        return np.flip((self.estimated_machine_expectancy + self._get_entropy_estimation()).argsort()[-self.K:])

    def update(self, chosen_machines, outcomes):
        # update probabilities as simple frequency counters, where all counters are initialized at 1
        # basically - this is the mode of the Dirichlet conjugate prior

        # find the indices of the outcomes in the reward array
        outcome_indices = np.searchsorted(self.rewards, outcomes)
        self.machine_reward_counter[chosen_machines, outcome_indices] += 1
        self.estimated_machine_reward_distribution = self.machine_reward_counter / np.sum(self.machine_reward_counter,
                                                                                          axis=1)[:, np.newaxis]
        for machine_index in range(self.N):
            self.estimated_machine_expectancy[machine_index] = self.machines[machine_index].get_mean_reward()
        self.estimated_machine_expectancy /= np.max(self.estimated_machine_expectancy)

        self.estimated_entropy = entropy(self.estimated_machine_reward_distribution, axis=1)
        self.estimated_entropy /= np.max(self.estimated_entropy)

    def _get_entropy_estimation(self):
        confidence = (2 * np.log(np.sum(self.estimated_entropy))) / self.estimated_entropy
        confidence[(confidence == -np.inf) | (confidence == np.NaN) | (confidence < 0)] = 0
        return confidence


class RandomModel(BaseModel):
    def __init__(self, machines, num_to_choose: int, num_trials: int, possible_rewards):
        super().__init__(machines, num_to_choose, num_trials, possible_rewards)

    def choose_machines(self):
        return np.random.choice(np.arange(self.N), self.K, False)

    def update(self, chosen_machines, outcomes):
        self.machine_reward_counter[chosen_machines, np.searchsorted(self.rewards, outcomes)] += 1
        self.estimated_machine_reward_distribution = self.machine_reward_counter / np.sum(self.machine_reward_counter,
                                                                                          axis=1)[:, np.newaxis]


class UCBEntropyGainModel(BaseModel):
    def __init__(self, machines, num_to_choose: int, num_trials: int, possible_rewards):
        super().__init__(machines, num_to_choose, num_trials, possible_rewards)
        self.machine_reward_counter = np.ones((self.N, self.rewards.size))
        self.estimated_machine_reward_distribution = self.machine_reward_counter / self.rewards.size
        self.estimated_machine_ucb = (self.estimated_machine_reward_distribution @ self.rewards)
        self.num_of_plays = 0

    def choose_machines(self) -> np.array:
        # choose K machines with largest UCB
        entropy_gain = self._get_estimated_entropy()
        print(-np.log(entropy_gain))
        return np.flip((self.estimated_machine_ucb / (-np.log(entropy_gain))).argsort()[-self.K:])

    def update(self, chosen_machines, outcomes):
        # update probabilities as simple frequency counters, where all counters are initialized at 1
        # basically - this is the mode of the Dirichlet conjugate prior
        self.num_of_plays += self.K
        for machine_index, machine in enumerate(self.machines):
            # update UCB of all machines
            self.estimated_machine_ucb[machine_index] = self._get_ucb(machine)
        outcome_indices = np.searchsorted(self.rewards,
                                          outcomes)  # find the indices of the outcomes in the reward array
        self.machine_reward_counter[chosen_machines, outcome_indices] += 1
        self.estimated_machine_reward_distribution = self.machine_reward_counter / np.sum(self.machine_reward_counter,
                                                                                          axis=1)[:, np.newaxis]
        # update mean reward of every chosen machine
        for machine_index in chosen_machines:
            self.estimated_machine_ucb[machine_index] = self._get_ucb(self.machines[machine_index])

    def _get_ucb(self, machine):
        confidence = (2 * np.log(self.num_of_plays)) / machine.num_of_plays
        if confidence == -np.inf or confidence == np.NaN or confidence < 0:  # in case of division by 0
            confidence = 0
        return machine.get_mean_reward() + confidence
