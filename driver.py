import numpy as np

import Visualization as vis
from ModelFactory import ModelType
from Simulation import Simulation

N = 30
K = 5
T = 10000
POSSIBLE_REWARDS = np.array([1, 3, 5, 10])

get_reward_probabilities = lambda: np.random.dirichlet(np.flip(1 + np.arange(len(POSSIBLE_REWARDS))))

# %%

SEED = 98  # same seeds
np.random.seed(SEED)
simEntropy = Simulation(N, K, T, POSSIBLE_REWARDS, get_reward_probabilities, ModelType.UCB_ENTROPY)
resEntropy = simEntropy.run_simulation()
simEntropy.plot_choice_distributions()
# fig.savefig('entropy Thompson Simulation.png')

np.random.seed(SEED)
simNormal = Simulation(N, K, T, POSSIBLE_REWARDS, get_reward_probabilities, ModelType.UCB_NORMAL)
resNormal = simNormal.run_simulation()
simNormal.plot_choice_distributions()
# fig.savefig('Normal Thompson Simulation.png')

np.random.seed(SEED)
simBaseline = Simulation(N, K, T, POSSIBLE_REWARDS, get_reward_probabilities, ModelType.BASELINE_MODEL)
resBaseline = simBaseline.run_simulation()
simBaseline.plot_choice_distributions()
# %%
# TODO qualitative evaluation over many seeds

"""
parameters for comparisons:
    1. Convergence rate (variance over time)
    2. Sum of reward over time
    3. ratio of end reward
    4. fisher rao - distance between estimated machine probabilities and actual probabilities
"""

# %%
window_size = 100
sim_list = [simNormal, simEntropy, simBaseline]
# %% plot convergence
vis.plot_convergences(sim_list)

# %% plot cumulative reward
vis.plot_rewards(sim_list)

# %% plot cumulative reward ratio
vis.plot_reward_ratios(sim_list)

# %% Fisher Rao
vis.plot_distance_of_distribution_estimations(sim_list)
