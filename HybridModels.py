from DirectedExplorationModels import *
from RandomExplorationModels import *


############################################################################################
############################################################################################
##############################         Lambda Models          ##############################
############################################################################################
############################################################################################

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
        cur_estimates = lambda_estimates / (-np.log((10 ** (-self.beta_handle)) * self._get_estimated_entropy_gain()))
        return cur_estimates if get_estimates else super()._get_top_k(cur_estimates)


class LambdaBetaModelPlus(LambdaModel):

    def __init__(self, machines, num_to_choose: int, num_trials: int, possible_rewards, lambda_handle: float,
                 beta_handle: float):
        super().__init__(machines, num_to_choose, num_trials, possible_rewards, lambda_handle)
        self.beta_handle = beta_handle

    @property
    def model_name(self):
        return r"TS, UCB1 + Entropy, $\lambda=%.2f, \beta=%.2f$" % (self.lambda_handle, self.beta_handle)

    def choose_machines(self, get_estimates=False):
        lambda_estimates = super().choose_machines(True)
        cur_estimates = (1 - self.beta_handle) * lambda_estimates + \
                        self.beta_handle * self._get_estimated_entropy_gain()
        return cur_estimates if get_estimates else super()._get_top_k(cur_estimates)


class LambdaBetaModelPlusNormalized(LambdaModel):

    def __init__(self, machines, num_to_choose: int, num_trials: int, possible_rewards, lambda_handle: float,
                 beta_handle: float):
        super().__init__(machines, num_to_choose, num_trials, possible_rewards, lambda_handle)
        self.beta_handle = beta_handle

    @property
    def model_name(self):
        return r"TS, UCB1 + Entropy normalized, $\lambda=%.2f, \beta=%.2f$" % (self.lambda_handle, self.beta_handle)

    def choose_machines(self, get_estimates=False):
        lambda_estimates = super().choose_machines(True)
        lambda_estimates /= np.sum(lambda_estimates)
        ent = self._get_estimated_entropy_gain()
        ent /= np.sum(ent)
        cur_estimates = (1 - self.beta_handle) * lambda_estimates + self.beta_handle * ent
        return cur_estimates if get_estimates else super()._get_top_k(cur_estimates)


############################################################################################
############################################################################################
########################         UCB Based Thompson Models          ########################
############################################################################################
############################################################################################

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
        cur_estimates = thompson_UCB_estimates / (
            -np.log((10 ** (-self.beta_handle)) * self._get_estimated_entropy_gain()))
        return cur_estimates if get_estimates else super()._get_top_k(cur_estimates)

    @property
    def model_name(self):
        return r"UCB based Thompson, $\beta=%.2f$" % self.beta_handle


class UCBBasedThompsonBetaPlusModel(UCBBasedThompsonModel):
    def __init__(self, machines, num_to_choose: int, num_trials: int, possible_rewards, beta_handle: float):
        super().__init__(machines, num_to_choose, num_trials, possible_rewards)
        self.beta_handle = beta_handle

    def choose_machines(self, get_estimates=False):
        thompson_UCB_estimates = super().choose_machines(True)
        cur_estimates = (1 - self.beta_handle) * thompson_UCB_estimates + (
                self.beta_handle * self._get_estimated_entropy_gain())
        return cur_estimates if get_estimates else super()._get_top_k(cur_estimates)

    @property
    def model_name(self):
        return r"UCB based Thompson, $\beta=%.2f$" % self.beta_handle


############################################################################################
############################################################################################
########################          Stochastic models Models          ########################
############################################################################################
############################################################################################


class StochasticThompsonUCBModel(BaseModel):
    def __init__(self, machines, num_to_choose: int, num_trials: int, possible_rewards, theta: float,
                 learning_rate=1e-4):
        super().__init__(machines, num_to_choose, num_trials, possible_rewards)
        # ucb = 1, thompson = 0
        self.theta = theta
        self.learning_rate = learning_rate
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
        self.theta += self.theta*self.learning_rate
        if self.theta > 1:
            self.theta = 1


class StochasticThompsonUCBBetaModel(StochasticThompsonUCBModel):
    def __init__(self, machines, num_to_choose: int, num_trials: int, possible_rewards, theta: float,
                 beta_handle: float, learning_rate=1e-4):
        super().__init__(machines, num_to_choose, num_trials, possible_rewards, theta, learning_rate)
        self.beta_handle = beta_handle

    def choose_machines(self, get_estimates=False):
        thompson_ucb_estimates = super().choose_machines(True)
        cur_estimates = thompson_ucb_estimates / (
            -np.log((10 ** (-self.beta_handle)) * self._get_estimated_entropy_gain()))
        return cur_estimates if get_estimates else super()._get_top_k(cur_estimates)

    @property
    def model_name(self):
        return r"UCB & Thompson stochastic, Entropy gain $\theta=%.2f, \beta=%.2f$" % (self.theta, self.beta_handle)


class StochasticThompsonUCBBetaPlusModel(StochasticThompsonUCBModel):
    def __init__(self, machines, num_to_choose: int, num_trials: int, possible_rewards, theta: float,
                 beta_handle: float, learning_rate=1e-4):
        super().__init__(machines, num_to_choose, num_trials, possible_rewards, theta, learning_rate=1e-4)
        self.beta_handle = beta_handle

    def choose_machines(self, get_estimates=False):
        thompson_ucb_estimates = super().choose_machines(True)
        cur_estimates = (1 - self.beta_handle) * thompson_ucb_estimates + (
                self.beta_handle * self._get_estimated_entropy_gain())
        return cur_estimates if get_estimates else super()._get_top_k(cur_estimates)

    @property
    def model_name(self):
        return r"UCB & Thompson stochastic, Entropy gain $\theta=%.2f, \beta=%.2f$" % (self.theta, self.beta_handle)


############################################################################################
############################################################################################
########################                noise Models                ########################
############################################################################################
############################################################################################
class LambdaBetaPlusNormalizedNoiseModel(LambdaBetaModelPlusNormalized):
    def __init__(self, machines, num_to_choose: int, num_trials: int, possible_rewards, lambda_handle: float,
                 beta_handle: float, noise_sigma):
        super().__init__(machines, num_to_choose, num_trials, possible_rewards, lambda_handle, beta_handle)
        self.noise = lambda: np.random.normal(0, noise_sigma, self.machines.size)

    def choose_machines(self, get_estimates=False):
        cur_estimates = super().choose_machines(True) + self.noise()
        return cur_estimates if get_estimates else super()._get_top_k(cur_estimates)


class LambdaBetaNoiseModel(LambdaBetaModel):
    def __init__(self, machines, num_to_choose: int, num_trials: int, possible_rewards, lambda_handle: float,
                 beta_handle: float, noise_sigma):
        super().__init__(machines, num_to_choose, num_trials, possible_rewards, lambda_handle, beta_handle)
        self.noise = lambda: np.random.normal(0, noise_sigma, self.machines.size)

    def choose_machines(self, get_estimates=False):
        cur_estimates = super().choose_machines(True) + self.noise()
        return cur_estimates if get_estimates else super()._get_top_k(cur_estimates)


class StochasticThompsonUCBBetaNoiseModel(StochasticThompsonUCBBetaModel):
    def __init__(self, machines, num_to_choose: int, num_trials: int, possible_rewards, theta: float,
                 beta_handle: float, learning_rate=1e-4, noise_sigma=None):
        super().__init__(machines, num_to_choose, num_trials, possible_rewards, theta, beta_handle)
        self.noise = lambda: np.random.normal(0, noise_sigma if noise_sigma else 1, self.machines.size)

    def choose_machines(self, get_estimates=False):
        cur_estimates = super().choose_machines(True) + self.noise()
        return cur_estimates if get_estimates else super()._get_top_k(cur_estimates)
