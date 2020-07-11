from Simulation import Simulation
from ModelFactory import ModelType
import numpy as np

N = 20
K = 3
T = 100
POSSIBLE_REWARDS = np.array([1, 2, 3, 4, 5])


def get_reward_probabilities():
    probs = np.random.uniform(1, 5, len(POSSIBLE_REWARDS))
    probs /= np.sum(probs)
    return probs


sim = Simulation(N, K, T, POSSIBLE_REWARDS, get_reward_probabilities, ModelType.UCB_NORMAL)
# %%
simulation_results = sim.run_simulation()

