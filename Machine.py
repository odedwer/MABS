import numpy as np


class Machine:
    def __init__(self, rewards, real_probabilities):
        """
        initializes a machine
        :param rewards: set of possible rewards
        :param real_probability: array of reward probabilities
        """
        self.rewards = np.array(rewards)
        self.real_probabilities = real_probabilities.copy()
        self.num_of_plays = 0
        self.sum_reward = 0

    def get_mean_reward(self):
        return 0 if not self.num_of_plays else self.sum_reward / self.num_of_plays

    def play(self):
        """
        :return: reward sampled by the real reward probabilities
        """
        self.num_of_plays += 1
        reward = np.random.choice(self.rewards, 1, False, self.real_probabilities)
        self.sum_reward += reward
        return reward

    def get_expectancy(self):
        return np.sum(self.rewards * self.real_probabilities)
