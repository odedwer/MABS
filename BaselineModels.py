from BaseModel import *


class EntropyGainModel(BaseModel):
    def __init__(self, machines, num_to_choose: int, num_trials: int, possible_rewards):
        super().__init__(machines, num_to_choose, num_trials, possible_rewards)

    def choose_machines(self, get_estimates=False):
        ent = super()._get_estimated_entropy_gain()
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
