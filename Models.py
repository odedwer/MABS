from BaseModel import BaseModel
import numpy as np


class UCBNormalModel(BaseModel):
    def __init__(self, machines, num_to_choose: int, num_trials: int, possible_rewards):
        super().__init__(machines, num_to_choose, num_trials, possible_rewards)
        self.estimated_machine_ucb = np.zeros((self.N,))
        self.number_of_visits_per_machine = np.zeros((self.N,))

    def choose_machines(self) -> np.array:
        # choose K machines with largest UCB
        return np.sort(self.estimated_machine_ucb.argsort()[-self.K:][::-1])

    def update(self, chosen_machines, outcomes):
        overall_visits = np.sum(self.number_of_visits_per_machine)
        for machine_index in chosen_machines:
            # update visits
            self.number_of_visits_per_machine[machine_index] += 1
        for machine_index in range(self.N):
            # update UCB of all machines
            self.estimated_machine_ucb[machine_index] = self._get_confidence(self.machines[machine_index],
                                                                             overall_visits)

    @staticmethod
    def _get_confidence(machine, overall_visits):
        confidence = (2 * np.log(overall_visits)) / machine.num_of_plays
        if confidence == -np.inf or confidence == np.NaN or confidence < 0:  # in case of division by 0
            confidence = 0
        return machine.get_mean_reward() + confidence
