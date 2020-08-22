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


class UCBEntropyGainPlusModel(UCBNormalModel):
    def __init__(self, machines, num_to_choose: int, num_trials: int, possible_rewards, beta_handle):
        super().__init__(machines, num_to_choose, num_trials, possible_rewards)
        self.beta_handle = beta_handle

    @property
    def model_name(self):
        return r"AEG-UCB1, $\beta=%.2f$" % self.beta_handle

    def update(self, chosen_machines, outcomes):
        # update machine ucb
        super().update(chosen_machines, outcomes)
        # add entropy gain for all machines
        self.estimated_machine_ucb *= (1 - self.beta_handle)
        self.estimated_machine_ucb += self.beta_handle * (self._get_estimated_entropy_gain())


class UCBEntropyGainModel(UCBNormalModel):
    def __init__(self, machines, num_to_choose: int, num_trials: int, possible_rewards, beta_handle):
        super().__init__(machines, num_to_choose, num_trials, possible_rewards)
        self.beta_handle = beta_handle

    @property
    def model_name(self):
        return r"LEG-UCB1, $\beta=%.2f$" % self.beta_handle

    def update(self, chosen_machines, outcomes):
        # update machine ucb
        super().update(chosen_machines, outcomes)
        # add entropy gain for all machines
        self.estimated_machine_ucb /= -np.log(
            (10 ** (-self.beta_handle)) * self._get_estimated_entropy_gain())
