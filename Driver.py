import numpy as np

import Visualization as vis
from ModelFactory import ModelType
from Simulation import Simulation
import matplotlib.pyplot as plt

# %%
N = 30
K = 5
T = 10000
POSSIBLE_REWARDS = np.array([1, 5, 10, 100])

get_reward_probabilities = lambda: np.random.dirichlet(np.ones_like(POSSIBLE_REWARDS))
# %%
SEED = 98  # same seeds

sim_list = []
#%%
np.random.seed(SEED)
sim = Simulation(N, K, T, POSSIBLE_REWARDS, get_reward_probabilities, ModelType.LAMBDA, lambda_handle=.5)
sim.run_simulation()
sim_list.append(sim)

np.random.seed(SEED)
sim = Simulation(N, K, T, POSSIBLE_REWARDS, get_reward_probabilities, ModelType.LAMBDA_BETA, lambda_handle=.5,
                 beta_handle=1)
sim.run_simulation()
sim_list.append(sim)

np.random.seed(SEED)
sim = Simulation(N, K, T, POSSIBLE_REWARDS, get_reward_probabilities, ModelType.LAMBDA_BETA, lambda_handle=.5,
                 beta_handle=30)
sim.run_simulation()
sim_list.append(sim)

np.random.seed(SEED)
sim = Simulation(N, K, T, POSSIBLE_REWARDS, get_reward_probabilities, ModelType.BETA_UBC_BASED_THOMPSON,
                 beta_handle=1)
sim.run_simulation()
sim_list.append(sim)

np.random.seed(SEED)
sim = Simulation(N, K, T, POSSIBLE_REWARDS, get_reward_probabilities, ModelType.BETA_UBC_BASED_THOMPSON,
                 beta_handle=30)
sim.run_simulation()
sim_list.append(sim)

np.random.seed(SEED)
sim = Simulation(N, K, T, POSSIBLE_REWARDS, get_reward_probabilities, ModelType.STOCHASTIC, theta=.5)
sim.run_simulation()
sim_list.append(sim)

np.random.seed(SEED)
sim = Simulation(N, K, T, POSSIBLE_REWARDS, get_reward_probabilities, ModelType.BETA_STOCHASTIC, theta=.5,
                 beta_handle=1)
sim.run_simulation()
sim_list.append(sim)

np.random.seed(SEED)
sim = Simulation(N, K, T, POSSIBLE_REWARDS, get_reward_probabilities, ModelType.BETA_STOCHASTIC, theta=.5,
                 beta_handle=30)
sim.run_simulation()
sim_list.append(sim)

np.random.seed(SEED)
sim = Simulation(N, K, T, POSSIBLE_REWARDS, get_reward_probabilities, ModelType.UCB_NORMAL)
sim.run_simulation()
sim_list.append(sim)

np.random.seed(SEED)
sim = Simulation(N, K, T, POSSIBLE_REWARDS, get_reward_probabilities, ModelType.UCB_ENTROPY)
sim.run_simulation()
sim_list.append(sim)

np.random.seed(SEED)
sim = Simulation(N, K, T, POSSIBLE_REWARDS, get_reward_probabilities, ModelType.UCB_ENTROPY_GAIN)
sim.run_simulation()
sim_list.append(sim)

np.random.seed(SEED)
sim = Simulation(N, K, T, POSSIBLE_REWARDS, get_reward_probabilities, ModelType.UCB_ENTROPY_NORMALIZED)
sim.run_simulation()
sim_list.append(sim)

np.random.seed(SEED)
sim = Simulation(N, K, T, POSSIBLE_REWARDS, get_reward_probabilities, ModelType.THOMPSON_NORMAL)
sim.run_simulation()
sim_list.append(sim)

np.random.seed(SEED)
sim = Simulation(N, K, T, POSSIBLE_REWARDS, get_reward_probabilities, ModelType.THOMPSON_ENTROPY)
sim.run_simulation()
sim_list.append(sim)

np.random.seed(SEED)
sim = Simulation(N, K, T, POSSIBLE_REWARDS, get_reward_probabilities, ModelType.BASELINE_MODEL)
sim.run_simulation()
sim_list.append(sim)

np.random.seed(SEED)
sim = Simulation(N, K, T, POSSIBLE_REWARDS, get_reward_probabilities, ModelType.ENTROPY_GAIN_MODEL)
sim.run_simulation()
sim_list.append(sim)

# %%
window_size = 300
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

