from BaseModel import *


class ThompsonNormalModel(BaseModel):
    def __init__(self, machines, num_to_choose: int, num_trials: int, possible_rewards):
        super().__init__(machines, num_to_choose, num_trials, possible_rewards)

    @property
    def model_name(self):
        return "Thompson Sampling"

    def choose_machines(self, get_estimates=False):
        estimated_reward_probabilities = self._vectorized_dirichlet_sample()
        estimated_rewards = np.sum(estimated_reward_probabilities * self.rewards,
                                   axis=1) if self.different_rewards else estimated_reward_probabilities @ self.rewards
        return estimated_rewards if get_estimates else super()._get_top_k(estimated_rewards)

    def _vectorized_dirichlet_sample(self):
        """
        Generate samples from an array of alpha distributions.
        from https://stackoverflow.com/questions/15915446/why-does-numpy-random-dirichlet-not-accept-multidimensional-arrays
        """
        r = np.random.standard_gamma(self.machine_reward_counter)
        return r / r.sum(-1, keepdims=True)


class ThompsonEntropyGainModel(ThompsonNormalModel):
    def __init__(self, machines, num_to_choose: int, num_trials: int, possible_rewards, beta_handle):
        super().__init__(machines, num_to_choose, num_trials, possible_rewards)
        self.beta_handle = beta_handle

    @property
    def model_name(self):
        return r"LEG-TS, $\beta=%.2f$" % self.beta_handle

    def choose_machines(self, get_estimates=False):
        estimated_rewards = super().choose_machines(True)
        estimated_rewards /= (-np.log((10 ** (-self.beta_handle)) * self._get_estimated_entropy_gain()))[:, np.newaxis]
        return estimated_rewards if get_estimates else super()._get_top_k(estimated_rewards)


class ThompsonEntropyGainPlusModel(ThompsonNormalModel):
    def __init__(self, machines, num_to_choose: int, num_trials: int, possible_rewards, beta_handle):
        super().__init__(machines, num_to_choose, num_trials, possible_rewards)
        self.beta_handle = beta_handle

    @property
    def model_name(self):
        return r"AEG-TS, $\beta=%.2f$" % self.beta_handle

    def choose_machines(self, get_estimates=False):
        estimated_rewards = super().choose_machines(True)
        estimated_rewards *= (1 - self.beta_handle)
        estimated_rewards += self.beta_handle * self._get_estimated_entropy_gain()
        return estimated_rewards if get_estimates else super()._get_top_k(estimated_rewards)
