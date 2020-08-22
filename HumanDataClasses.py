# %%
import numpy as np
# from abc import ABC, abstractmethod
from DataReader import *
import pandas as pd

GERSHMAN_REWARD_SIGMA = np.sqrt(10)


class HumanDataHandler:
    def __init__(self, ind):
        # 44/45 subjects, 20 blocks per participant, 10 trials per block
        self.df = pd.read_csv(gershman_data1_file if ind == 1 else gershman_data2_file,
                              usecols=lambda col: col != "RT")
        cum_reward = 0
        cur_subj = 1
        cur_block = 1
        for ind, row in self.df.iterrows():
            if row['subject'] != cur_subj or row['block'] != cur_block:
                cum_reward = 0
                cur_subj = row['subject']
                cur_block = row['block']
            cum_reward += row['reward']
            self.df.at[ind, 'cum_reward'] = cum_reward

    def get_block_data(self, subject, block):
        mu1 = np.array(self.df.loc[(self.df['subject'] == subject) & (self.df['block'] == block), 'mu1'])[0]
        mu2 = np.array(self.df.loc[(self.df['subject'] == subject) & (self.df['block'] == block), 'mu2'])[0]
        cum_reward_list = self.df.loc[(self.df['subject'] == subject) & (self.df['block'] == block), 'cum_reward']
        return mu1, mu2, np.array(cum_reward_list)

    def get_n_subjects(self):
        return np.max(self.df['subject'])

    def get_n_blocks(self, subject):
        return np.max(self.df.loc[self.df['subject'] == subject, 'block'])

    def get_n_trials(self, subject, block):
        return np.max(self.df.loc[(self.df['subject'] == subject) & (self.df['block'] == block), 'trial'])

# class HumanTrial:
#     def __init__(self, choice, reward):
#         self.machine_choice = choice
#         self.reward = reward
#
#     def get_reward(self):
#         return self.reward
#
#     def get_choice(self):
#         return self.machine_choice
#
#
# class GershmanMachine:
#     def __init__(self, mean):
#         self.mean = mean
#         self.machine_type = 0
#
#     # Not sure how to implement
#     def calc_reward_probability(self, reward):
#         return 0
#
#
# class TrialsBlock:
#     def __init__(self, trials=None, machines=None):
#         if machines is None:
#             machines = []
#         if trials is None:
#             trials = []
#         self.trials = trials
#         self.machines = machines
#
#     def get_trials(self):
#         return self.trials
#
#     def get_machines(self):
#         return self.machines
#
#     def switch_vec(self):
#         vec = []
#         for i in range(len(self.trials) - 1):
#             vec.append(self.trials[i].machine_choice != self.trials[i + 1].machine_choice)
#         return np.array(vec)
#
#     def cum_reward_vec(self):
#         vec = []
#         cur = 0
#         for i in range(len(self.trials)):
#             cur += self.trials[i].reward
#             vec.append(cur)
#         return np.array(vec)
#
#     def reward_vec(self):
#         vec = []
#         for i in self.trials:
#             vec.append(i.reward)
#         return np.array(vec)
#
#     def add_trial(self, trial):
#         self.trials.append(trial)
#
#
# class GershmanExperimentData:
#     def __init__(self, file):
#         rows = file_to_rows(file)[1:]
#         cur_par = 0
#         cur_block = 0
#         self.participants = []
#         for r in rows:
#             if int(r[0]) != cur_par:
#                 cur_par = int(r[0])
#                 cur_block = 1
#                 self.participants.append([])
#                 self.participants[-1].append(TrialsBlock(machines=[
#                     GershmanMachine(mean=int(r[3])),
#                     GershmanMachine(mean=int(r[4]))]))
#             elif cur_block != int(r[1]):
#                 print(r[0], r[1])
#                 cur_block += 1
#                 self.participants[-1].append(TrialsBlock(machines=[
#                     GershmanMachine(mean=int(r[3])),
#                     GershmanMachine(mean=int(r[4]))]))
#             self.participants[-1][-1].add_trial(HumanTrial(int(r[5]) + 1, int(r[6])))
#
#     def get_object(self):
#         return self.participants
#
