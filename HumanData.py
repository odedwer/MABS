import numpy as np
from abc import ABC, abstractmethod


class humanTrial:
    def __init__(self, choice, reward):
        machine_choice = choice
        reward = reward

class data_machine(ABC):
    @abstractmethod
    def calc_reward_probability(self, reward):
        pass

    @property
    def mean(self):
        raise NotImplementedError


    # 0 is Gershman's, 1 is Stojic's
    @property
    def machine_type(self):
        raise NotImplementedError



class Gershman_Machine(data_machine):
    def __init__(self, mean):
        self.mean = mean
        self.machine_type = 0
    

    # Not sure how to implement
    def calc_reward_probability(self, reward):
        return 0


class Stojic_machine(data_machine):
    

class trialsBlock:
    def __init__(self, trials=[], machines=[]):
        self.trials = trials
        self.machines = machines

    def switchVec(self):
        vec = []
        for i in range(len(self.trials)-1):
            vec.append(self.trials[i].machine_choice == self.trials[i+1].machine_choice)
        return np.array(vec)

    def cumRewardVec(self):
        vec = []
        cur = 0
        for i in range(len(self.trials)):
            cur += self.trials[i].reward
            vec.append(cur)
        return np.array(vec)

    def rewardVec(self):
        vec=[]
        for i in self.trials:
            vec.append(i.reward)
        return np.array(vec)