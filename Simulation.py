import numpy as np
from Machine import Machine
from ModelFactory import getModel
from BaseModel import BaseModel


class Simulation:
    def __init__(self, num_machines: int, num_to_choose: int, num_trials: int, possible_rewards,
                 reward_probability_function, model_type):
        """
        Constructor for simulation class
        :param num_machines: number of overall machines in the simulation (N)
        :param num_to_choose: number of machines to choose in every trial (K)
        :param num_trials: number of trials the simulation will run (T)
        :param possible_rewards: set of possible rewards eah machine can give. The same for all machines
        :param reward_probability_function: function that returns a vector of probabilities, one for each possible
               reward
        :param model_type: The type of the model to run the simulation by
        """
        self.N = num_machines
        self.K = num_to_choose
        self.T = num_trials
        self.rewards = possible_rewards
        self.reward_probability_function = reward_probability_function
        self.machines_list = np.empty((self.N,), dtype=Machine)
        self.init_machines()
        self.model = getModel(model_type, self.machines_list, self.K, self.T, self.rewards)
        # numChosenMachines X Trials X (reward, machine number)
        self.results = np.zeros((self.K, self.T, 2))

    def init_machines(self):
        """
        initializes the machines for the simulation
        """
        for i in range(self.N):
            self.machines_list[i] = Machine(self.rewards, self.reward_probability_function())

    def run_simulation(self):
        for t in range(self.T):
            chosen_machines = self.model.choose_machines()
            for i, machine in enumerate(chosen_machines):
                self.results[i, t, :] = [self.machines_list[machine].play(), machine]
            self.model.update(chosen_machines, self.results[:, t, 0])
        return self.results.copy()
