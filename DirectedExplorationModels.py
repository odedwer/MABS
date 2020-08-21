from BaseModel import *


class UCBNormalModel(BaseModel):
    def __init__(self, machines, num_to_choose: int, num_trials: int, possible_rewards):
        super().__init__(machines, num_to_choose, num_trials, possible_rewards)
        self.estimated_machine_ucb = np.zeros((self.N,))
        self.num_of_plays = 0

    @property
    def model_name(self):
        return "UCB1"

    def choose_machines(self, get_estimates=False) -> np.array:
        # choose K machines with largest UCB
        return self.estimated_machine_ucb if get_estimates else super()._get_top_k(self.estimated_machine_ucb)

    def update(self, chosen_machines, outcomes):
        self.num_of_plays += self.K
        for machine_index, machine in enumerate(self.machines):
            # update UCB of all machines
            self.estimated_machine_ucb[machine_index] = self._get_ucb(machine)
        super().update(chosen_machines, outcomes)

    def _get_ucb(self, machine):
        confidence = (2 * np.log(self.num_of_plays)) / machine.num_of_plays
        if confidence == -np.inf or confidence == np.NaN or confidence < 0:  # in case of division by 0
            confidence = 0
        return machine.get_mean_reward() + confidence


class UCBEntropyGainModel(BaseModel):
    def __init__(self, machines, num_to_choose: int, num_trials: int, possible_rewards, beta_handle):
        super().__init__(machines, num_to_choose, num_trials, possible_rewards)
        self.machine_reward_counter = np.ones((self.N, self.rewards.size))
        self.estimated_machine_reward_distribution = self.machine_reward_counter / self.rewards.size
        self.beta_handle = beta_handle
        self.estimated_machine_ucb = (1 - beta_handle) * (
                self.estimated_machine_reward_distribution @ self.rewards) + self.beta_handle * (
                                         self._get_estimated_entropy_gain()[:, np.newaxis])

    @property
    def model_name(self):
        return "Entropy Driven UCB1"

    def choose_machines(self, get_estimates=False) -> np.array:
        # choose K machines with largest UCB
        return self.estimated_machine_ucb if get_estimates else super()._get_top_k(self.estimated_machine_ucb)

    def update(self, chosen_machines, outcomes):
        # update probabilities as simple frequency counters, where all counters are initialized at 1
        # basically - this is the mode of the Dirichlet conjugate prior
        super().update(chosen_machines, outcomes)
        # update mean reward of every chosen machine
        for machine_index in chosen_machines:
            self.estimated_machine_ucb[machine_index] = self.machines[machine_index].get_mean_reward()

        # add entropy for all chosen machines - the higher the entropy, the higher our uncertainty
        # so we are optimistic in the face of uncertainty
        self.estimated_machine_ucb[chosen_machines] = (1 - self.beta_handle) * (
                    self.estimated_machine_reward_distribution[chosen_machines, :] @
                    self.rewards) + self.beta_handle * self._get_estimated_entropy_gain()


class UCBEntropyNormalizedModel(BaseModel):
    def __init__(self, machines, num_to_choose: int, num_trials: int, possible_rewards):
        super().__init__(machines, num_to_choose, num_trials, possible_rewards)
        self.estimated_machine_expectancy = self.estimated_machine_reward_distribution @ self.rewards
        self.estimated_entropy = entropy(self.estimated_machine_reward_distribution, axis=1)
        self.estimated_entropy = entropy(self.estimated_machine_reward_distribution, axis=1)

    @property
    def model_name(self):
        return "Normalized Entropy Driven UCB1"

    def choose_machines(self, get_estimates=False) -> np.array:
        cur_estimatess = self.estimated_machine_expectancy + self._get_estimated_entropy_gain()
        return cur_estimatess if get_estimates else super()._get_top_k(cur_estimatess)

    def update(self, chosen_machines, outcomes):
        # update probabilities as simple frequency counters, where all counters are initialized at 1
        # basically - this is the mode of the Dirichlet conjugate prior

        super().update(chosen_machines, outcomes)
        for machine_index in range(self.N):
            self.estimated_machine_expectancy[machine_index] = self.machines[machine_index].get_mean_reward()
        self.estimated_machine_expectancy /= np.max(self.estimated_machine_expectancy)

        self.estimated_entropy = entropy(self.estimated_machine_reward_distribution, axis=1)
        self.estimated_entropy /= np.max(self.estimated_entropy)
