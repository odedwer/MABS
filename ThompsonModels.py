from BaseModel import *


class ThompsonNormalModel(BaseModel):
    def __init__(self, machines, num_to_choose: int, num_trials: int, possible_rewards):
        super().__init__(machines, num_to_choose, num_trials, possible_rewards)

    @property
    def model_name(self):
        return "Thompson Sampling"

    def choose_machines(self, get_estimates=False):
        estimated_reward_probabilities = self._vectorized_dirichlet_sample()
        estimated_rewards = estimated_reward_probabilities @ self.rewards
        return estimated_rewards if get_estimates else super()._get_top_k(estimated_rewards)

    def _vectorized_dirichlet_sample(self):
        """
        Generate samples from an array of alpha distributions.
        from https://stackoverflow.com/questions/15915446/why-does-numpy-random-dirichlet-not-accept-multidimensional-arrays
        """
        r = np.random.standard_gamma(self.machine_reward_counter)
        return r / r.sum(-1, keepdims=True)


class ThompsonEntropyGainModel(ThompsonNormalModel):
    def __init__(self, machines, num_to_choose: int, num_trials: int, possible_rewards):
        super().__init__(machines, num_to_choose, num_trials, possible_rewards)

    @property
    def model_name(self):
        return "Entropy Driven Thompson Sampling"

    def choose_machines(self, get_estimates=False):
        estimated_reward_probabilities = self._vectorized_dirichlet_sample()
        estimated_rewards = estimated_reward_probabilities @ self.rewards
        return estimated_rewards if get_estimates else super()._get_top_k(estimated_rewards)

    def _vectorized_dirichlet_sample(self):
        """
        Generate samples from an array of reward counters per machine
        from https://stackoverflow.com/questions/15915446/why-does-numpy-random-dirichlet-not-accept-multidimensional-arrays
        """
        ent = self._get_estimated_entropy_gain()
        ent = ent / np.sum(ent)
        r = np.random.standard_gamma(
            self.machine_reward_counter / -np.log(ent)[:, np.newaxis])
        return r / r.sum(-1, keepdims=True)
