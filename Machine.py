import numpy as np


class Machine:
    def __init__(self, rewards, reward_probabilities):
        """
        initializes a machine
        :param rewards: set of possible rewards
        :param real_probability: array of reward probabilities
        """
        self.rewards = np.array(rewards)
        self.reward_probabilities = reward_probabilities.copy()
        self.expectancy = self.reward_probabilities @ self.rewards
        self.num_of_plays = 0
        self.sum_reward = 0
        self.outcomes = []

    def get_mean_reward(self):
        return 0 if not self.num_of_plays else self.sum_reward / self.num_of_plays

    def play(self):
        """
        :return: reward sampled by the real reward probabilities
        """
        self.num_of_plays += 1
        reward = np.random.choice(self.rewards, 1, False, self.reward_probabilities)
        self.outcomes.append(reward)
        self.sum_reward += reward
        return reward

    def get_expectancy(self):
        return self.expectancy

    def get_outcomes(self):
        return np.zeros((1,)) if not self.outcomes else np.asarray(self.outcomes)
