import numpy as np
from scipy.stats import entropy

from BaseModel import BaseModel


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


class ThompsonEntropyModel(ThompsonNormalModel):
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
        r = np.random.standard_gamma(
            self.machine_reward_counter / -np.log(self._get_estimated_entropy())[:, np.newaxis])
        return r / r.sum(-1, keepdims=True)


class UCBEntropyModel(BaseModel):
    def __init__(self, machines, num_to_choose: int, num_trials: int, possible_rewards):
        super().__init__(machines, num_to_choose, num_trials, possible_rewards)
        self.machine_reward_counter = np.ones((self.N, self.rewards.size))
        self.estimated_machine_reward_distribution = self.machine_reward_counter / self.rewards.size
        self.estimated_machine_ucb = (self.estimated_machine_reward_distribution @ self.rewards) * (1. / entropy(
            self.estimated_machine_reward_distribution, axis=1))

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
        self.estimated_machine_ucb[chosen_machines] = (self.estimated_machine_reward_distribution[chosen_machines, :] @
                                                       self.rewards) * (1. / entropy(
            self.estimated_machine_reward_distribution[chosen_machines, :], axis=1))


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
        cur_estimatess = self.estimated_machine_expectancy + self._get_entropy_estimation()
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

    def _get_entropy_estimation(self):
        confidence = (2 * np.log(np.sum(self.estimated_entropy))) / self.estimated_entropy
        confidence[(confidence == -np.inf) | (confidence == np.NaN) | (confidence < 0)] = 0
        return confidence


class LambdaModel(BaseModel):
    def __init__(self, machines, num_to_choose: int, num_trials: int, possible_rewards, lambda_handle: float):
        super().__init__(machines, num_to_choose, num_trials, possible_rewards)
        self.thompson = ThompsonNormalModel(machines, num_to_choose, num_trials, possible_rewards)
        self.ucb = UCBNormalModel(machines, num_to_choose, num_trials, possible_rewards)
        self.lambda_handle = lambda_handle
        self.joint_estimates = np.zeros((self.N,))

    @property
    def model_name(self):
        return r"TS & UCB1, $\lambda=%.2f$" % self.lambda_handle

    def choose_machines(self, get_estimates=False):
        thompson_estimates = self.thompson.choose_machines(True)
        ucb_estimates = self.ucb.choose_machines(True)
        self.joint_estimates = thompson_estimates * self.lambda_handle + ucb_estimates * (1 - self.lambda_handle)
        return self.joint_estimates if get_estimates else super()._get_top_k(self.joint_estimates)

    def update(self, chosen_machines, outcomes):
        super().update(chosen_machines, outcomes)
        self.thompson.update(chosen_machines, outcomes)
        self.ucb.update(chosen_machines, outcomes)


class LambdaBetaModel(LambdaModel):

    def __init__(self, machines, num_to_choose: int, num_trials: int, possible_rewards, lambda_handle: float,
                 beta_handle: float):
        super().__init__(machines, num_to_choose, num_trials, possible_rewards, lambda_handle)
        self.beta_handle = beta_handle

    @property
    def model_name(self):
        return r"TS, UCB1 & Entropy, $\lambda=%.2f, \beta=%.2f$" % (self.lambda_handle, self.beta_handle)

    def choose_machines(self, get_estimates=False):
        lambda_estimates = super().choose_machines(True)
        cur_estimates = lambda_estimates / (-np.log((10 ** (-self.beta_handle)) * self._get_estimated_entropy()))
        return cur_estimates if get_estimates else super()._get_top_k(cur_estimates)


class UCBBasedThompsonModel(ThompsonNormalModel):
    def __init__(self, machines, num_to_choose: int, num_trials: int, possible_rewards):
        super().__init__(machines, num_to_choose, num_trials, possible_rewards)
        self.estimated_machine_ucb = np.zeros_like(self.machines)
        self.num_of_plays = 0

    @property
    def model_name(self):
        return r"UCB Based Thompson"

    def choose_machines(self, get_estimates=False):
        thompson_estimates = super().choose_machines(True)
        cur_estimates = thompson_estimates + self.estimated_machine_ucb
        return cur_estimates if get_estimates else super()._get_top_k(cur_estimates)

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


class UCBBasedThompsonBetaModel(UCBBasedThompsonModel):
    def __init__(self, machines, num_to_choose: int, num_trials: int, possible_rewards, beta_handle: float):
        super().__init__(machines, num_to_choose, num_trials, possible_rewards)
        self.beta_handle = beta_handle

    def choose_machines(self, get_estimates=False):
        thompson_UCB_estimates = super().choose_machines(True)
        cur_estimates = thompson_UCB_estimates / (-np.log((10 ** (-self.beta_handle)) * self._get_estimated_entropy()))
        return cur_estimates if get_estimates else super()._get_top_k(cur_estimates)

    @property
    def model_name(self):
        return r"UCB based Thompson, $\beta=%.2f$" % self.beta_handle


class StochasticThompsonUCBModel(BaseModel):
    def __init__(self, machines, num_to_choose: int, num_trials: int, possible_rewards, theta: float):
        super().__init__(machines, num_to_choose, num_trials, possible_rewards)
        # ucb = 1, thompson = 0
        self.theta = theta
        self.ucb = UCBNormalModel(self.machines, num_to_choose, num_trials, possible_rewards)
        self.thompson = ThompsonNormalModel(self.machines, num_to_choose, num_trials, possible_rewards)

    @property
    def model_name(self):
        return r"UCB & Thompson, stochastic with $\theta=%.2f$" % self.theta

    def choose_machines(self, get_estimates=False):
        return self.ucb.choose_machines(get_estimates) if \
            np.random.binomial(1, self.theta) else self.thompson.choose_machines(get_estimates)

    def update(self, chosen_machines, outcomes):
        super().update(chosen_machines, outcomes)
        self.ucb.update(chosen_machines, outcomes)
        self.thompson.update(chosen_machines, outcomes)


class StochasticThompsonUCBBetaModel(StochasticThompsonUCBModel):
    def __init__(self, machines, num_to_choose: int, num_trials: int, possible_rewards, theta: float,
                 beta_handle: float):
        super().__init__(machines, num_to_choose, num_trials, possible_rewards, theta)
        self.beta_handle = beta_handle

    def choose_machines(self, get_estimates=False):
        thompson_ucb_estimates = super().choose_machines(True)
        cur_estimates = thompson_ucb_estimates / (-np.log((10 ** (-self.beta_handle)) * self._get_estimated_entropy()))
        return cur_estimates if get_estimates else super()._get_top_k(cur_estimates)

    @property
    def model_name(self):
        return r"UCB & Thompson stochastic, Entropy gain $\theta=%.2f, \beta=%.2f$" % (self.theta, self.beta_handle)


class EntropyGainModel(BaseModel):
    def __init__(self, machines, num_to_choose: int, num_trials: int, possible_rewards):
        super().__init__(machines, num_to_choose, num_trials, possible_rewards)

    def choose_machines(self, get_estimates=False):
        ent = super()._get_estimated_entropy()
        return ent if get_estimates else super()._get_top_k(ent)

    @property
    def model_name(self):
        return "Entropy Gain"


class RandomModel(BaseModel):
    def __init__(self, machines, num_to_choose: int, num_trials: int, possible_rewards):
        super().__init__(machines, num_to_choose, num_trials, possible_rewards)

    @property
    def model_name(self):
        return "Baseline (random)"

    def choose_machines(self, get_estimates=False):
        return np.random.choice(np.arange(self.N), self.N, False) if get_estimates else np.random.choice(
            np.arange(self.N), self.K, False)


class UCBEntropyGainModel(BaseModel):
    def __init__(self, machines, num_to_choose: int, num_trials: int, possible_rewards):
        super().__init__(machines, num_to_choose, num_trials, possible_rewards)
        self.machine_reward_counter = np.ones((self.N, self.rewards.size))
        self.estimated_machine_reward_distribution = self.machine_reward_counter / self.rewards.size
        self.estimated_machine_ucb = (self.estimated_machine_reward_distribution @ self.rewards)
        self.num_of_plays = 0

    @property
    def model_name(self):
        return "Entropy Gain Driven UCB1"

    def choose_machines(self, get_estimates=False) -> np.array:
        # choose K machines with largest UCB
        entropy_gain = self._get_estimated_entropy()
        cur_estimates = self.estimated_machine_ucb / (-np.log(entropy_gain))
        return cur_estimates if get_estimates else super()._get_top_k(cur_estimates)

    def update(self, chosen_machines, outcomes):
        # update probabilities as simple frequency counters, where all counters are initialized at 1
        # basically - this is the mode of the Dirichlet conjugate prior
        super().update(chosen_machines, outcomes)
        self.num_of_plays += self.K
        for machine_index, machine in enumerate(self.machines):
            # update UCB of all machines
            self.estimated_machine_ucb[machine_index] = self._get_ucb(machine)

    def _get_ucb(self, machine):
        confidence = (2 * np.log(self.num_of_plays)) / machine.num_of_plays
        if confidence == -np.inf or confidence == np.NaN or confidence < 0:  # in case of division by 0
            confidence = 0
        return machine.get_mean_reward() + confidence
