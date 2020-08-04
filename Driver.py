import numpy as np

import Visualization as vis
from ModelFactory import ModelType
from Simulation import Simulation
import matplotlib.pyplot as plt
#%%
N = 2
K = 1
T = 10000
POSSIBLE_REWARDS = np.array([0,1])

get_reward_probabilities = lambda: np.random.dirichlet(np.flip(1 + np.arange(len(POSSIBLE_REWARDS))))

fig = plt.figure(figsize=(24,8))
entropy_ax, normal_ax, baseline_ax = fig.subplots(1, 3)
SEED = 98  # same seeds
np.random.seed(SEED)
simEntropy = Simulation(N, K, T, POSSIBLE_REWARDS, get_reward_probabilities, ModelType.THOMPSON_ENTROPY)
resEntropy = simEntropy.run_simulation()
simEntropy.plot_choice_distributions(entropy_ax)

np.random.seed(SEED)
simNormal = Simulation(N, K, T, POSSIBLE_REWARDS, get_reward_probabilities, ModelType.THOMPSON_NORMAL)
resNormal = simNormal.run_simulation()
simNormal.plot_choice_distributions(normal_ax)

np.random.seed(SEED)
simBaseline = Simulation(N, K, T, POSSIBLE_REWARDS, get_reward_probabilities, ModelType.BASELINE_MODEL)
resBaseline = simBaseline.run_simulation()
simBaseline.plot_choice_distributions(baseline_ax)
#%%
model_name = "UCB"
fig.savefig("%s choices comparison.png" % model_name)
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
fig = vis.plot_convergences(sim_list)
# fig.savefig("%s convergence rate.png" % model_name)
# %% plot cumulative reward
fig = vis.plot_rewards(sim_list)
# fig.savefig("%s cumulative reward.png" % model_name)
# %% plot cumulative reward ratio
fig = vis.plot_reward_ratios(sim_list)
# fig.savefig("%s cumulative reward ratio.png" % model_name)

# %% Fisher Rao
fig = vis.plot_distance_of_distribution_estimations(sim_list)
# fig.savefig("%s FR distance between estimated and real reward probabilities.png" % model_name)
