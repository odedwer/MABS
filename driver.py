import numpy as np

from ModelFactory import ModelType
from Simulation import Simulation

N = 30
K = 10
T = 10000
POSSIBLE_REWARDS = np.array([1, 2, 3, 4, 5])

get_reward_probabilities = lambda: np.random.dirichlet(np.flip(np.arange(1, len(POSSIBLE_REWARDS) + 1)))

# %%

SEED = 98  # same seeds
np.random.seed(SEED)
simEntropy = Simulation(N, K, T, POSSIBLE_REWARDS, get_reward_probabilities, ModelType.THOMPSON_ENTROPY)
resEntropy = simEntropy.run_simulation()
fig = simEntropy.plot_choice_distributions()
fig.savefig('entropy Thompson Simulation.png')

np.random.seed(SEED)
simNormal = Simulation(N, K, T, POSSIBLE_REWARDS, get_reward_probabilities, ModelType.THOMPSON_NORMAL)
resNormal = simNormal.run_simulation()
fig = simNormal.plot_choice_distributions()
fig.savefig('Normal Thompson Simulation.png')

# %%
simEntropy.machine_indices_by_expectancy + 1
# %%
print(np.sum(resEntropy[:, :, 0]) / np.sum(resNormal[:, :, 0]))
# %%
# TODO random model - for baseline comparison
# TODO qualitative evaluation over many seeds

"""
parameters for comparisons:
    1. Convergence rate (variance over time)
    2. Sum of reward over time
    3. ratio of end reward
    4. fisher rao - distance between estimated machine probabilities and actual probabilities
"""
