import numpy as np
from abc import ABC, abstractmethod
from DataReader import *


class HumanTrial:
    def __init__(self, choice, reward):
        self.machine_choice = choice
        self.reward = reward

    def get_reward(self):
        return self.reward

    def get_choice(self):
        return self.machine_choice


class GershmanMachine:
    def __init__(self, mean):
        self.mean = mean
        self.machine_type = 0

    # Not sure how to implement
    def calc_reward_probability(self, reward):
        return 0


class TrialsBlock:
    def __init__(self, trials=None, machines=None):
        if machines is None:
            machines = []
        if trials is None:
            trials = []
        self.trials = trials
        self.machines = machines

    def get_trials(self):
        return self.trials

    def get_machines(self):
        return self.machines

    def switch_vec(self):
        vec = []
        for i in range(len(self.trials) - 1):
            vec.append(self.trials[i].machine_choice != self.trials[i + 1].machine_choice)
        return np.array(vec)

    def cum_reward_vec(self):
        vec = []
        cur = 0
        for i in range(len(self.trials)):
            cur += self.trials[i].reward
            vec.append(cur)
        return np.array(vec)

    def reward_vec(self):
        vec = []
        for i in self.trials:
            vec.append(i.reward)
        return np.array(vec)

    def add_trial(self, trial):
        self.trials.append(trial)


class GershmanExperimentData:
    def __init__(self, file):
        rows = file_to_rows(file)[1:]
        cur_par = 0
        cur_block = 0
        self.participants = []
        for r in rows:
            if int(r[0]) != cur_par:
                cur_par = int(r[0])
                cur_block = 1
                self.participants.append([])
                self.participants[-1].append(TrialsBlock(machines=[
                GershmanMachine(mean=int(r[3])),
                GershmanMachine(mean=int(r[4]))]))
            elif cur_block != int(r[1]):
                print(r[0],r[1])
                cur_block += 1
                self.participants[-1].append(TrialsBlock(machines=[
                GershmanMachine(mean=int(r[3])),
                GershmanMachine(mean=int(r[4]))]))
            self.participants[-1][-1].add_trial(HumanTrial(int(r[5]) + 1, int(r[6])))

    def get_object(self):
        return self.participants