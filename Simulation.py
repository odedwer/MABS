from sys import stderr

import matplotlib.pyplot as plt
import numpy as np

from Machine import Machine
from ModelFactory import getModel


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
        self.machine_list = np.empty((self.N,), dtype=Machine)
        self.init_machines()
        self.machine_indices_by_expectancy = np.asarray(
            [machine.get_expectancy() for machine in self.machine_list])  # ordered from highest to lowest
        self.machine_indices_by_expectancy = np.flip(np.argsort(self.machine_indices_by_expectancy))
        self.model = getModel(model_type, self.machine_list, self.K, self.T, self.rewards)
        self.type = model_type.name
        self.results = np.zeros((self.K, self.T, 2))  # numChosenMachines X Trials X (reward, machine number)

    def init_machines(self):
        """
        initializes the machines for the simulation
        """
        for i in range(self.N):
            self.machine_list[i] = Machine(self.rewards, self.reward_probability_function())

    def run_simulation(self):
        for t in range(self.T):
            chosen_machines = self.model.choose_machines()
            for i, machine in enumerate(chosen_machines):
                self.results[i, t, :] = [self.machine_list[machine].play(), machine]
            self.model.update(chosen_machines, self.results[:, t, 0])
        return self.results.copy()

    def plot_choice_distributions(self) -> plt.Figure:
        if np.sum(self.results) == 0:
            print("Simulation not run yet!\n "
                  "Please run the simulation (using run_simulation method) before plotting results", file=stderr)
            return None
        fig = plt.figure()
        ax = fig.subplots()
        l = ax.plot(self.results[:, :, 1].T, 'o', markersize=2)
        ax.legend(l, [f"Machine {i}" for i in range(1, self.K + 1)])
        ax.set_yticks(np.arange(self.N))
        ax.set_yticklabels(np.arange(self.N) + 1)
        # get best K machine indices and add to plot
        best_machines_text = "Highest expectation machines:\n"
        for i in self.machine_indices_by_expectancy[:self.K]:
            best_machines_text += f"{i + 1}, "
        ax.set_title(f"Machine choice by trial, {self.type}")
        ax.set_xlabel("Trial")
        ax.set_ylabel("Machine")
        plt.figtext(0.125, .9, best_machines_text, horizontalalignment='left')
        return fig
