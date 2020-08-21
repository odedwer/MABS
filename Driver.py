import numpy as np
from tqdm import tqdm
import Visualization as vis
from ModelFactory import ModelType
from Simulation import Simulation

# %%
MIN_SEED = 1
MAX_SEED = 4
N = 30
K = 5
T = 1000
POSSIBLE_REWARDS = np.array([1, 5, 10, 40])
get_reward_probabilities = lambda: np.random.dirichlet(np.arange(len(POSSIBLE_REWARDS), 0, -1) / len(POSSIBLE_REWARDS))


# %%
def average_over_seeds(model_type_list, model_parameters_list, n=N, k=K, t=T, possible_rewards=POSSIBLE_REWARDS,
                       get_reward_probabilities=lambda: np.random.dirichlet(np.ones_like(POSSIBLE_REWARDS)),
                       min_seed=MIN_SEED, max_seed=MAX_SEED):
    """
    run simulation for many models, with different seeds and return the required arrays for Visualization.py
    plot functions with vis.DATA list_type argument
    :param model_type_list: list of ModelType enums, containing the models to run
    :param model_parameters_list: list of dictionaries containing parameters for the models in model_type_list
    :param n: number of machines
    :param k: number of machines to choose each round
    :param t: number of trials
    :param possible_rewards: list of possible machine rewards
    :param get_reward_probabilities: function that returns a list of reward probabilities
    :param min_seed: lowest seed to run
    :param max_seed: highest seed to run
    :return: convergence, reward_sum, distance_from_real_distributions - arrays of tuples, first argument in tuple
    is the sim.type and the second is the actual data
    """
    convergence_list = np.zeros((len(model_type_list), t - 1))
    reward_sums = np.zeros((len(model_type_list), t))
    regrets = np.zeros((len(model_type_list), t))
    distances = [[] for _ in range(len(model_type_list))]
    sim_titles = []
    for seed in tqdm(range(min_seed, max_seed + 1)):
        optimal_reward = None
        cur_rewards = np.zeros_like(reward_sums)
        for i, model_type, model_parameters in tqdm(zip(np.arange(len(model_type_list)), model_type_list,
                                                        model_parameters_list)):
            np.random.seed(seed)
            sim = Simulation(n, k, t, possible_rewards, get_reward_probabilities, model_type, **model_parameters)
            if seed == min_seed:
                sim_titles.append(sim.type)
            sim.run_simulation()
            convergence_list[i, :] += sim.get_convergence_rate()
            if sim.type == "Optimal Model":
                optimal_reward = sim.get_reward_sum()
            cur_rewards[i, :] += sim.get_reward_sum()
            reward_sums[i, :] += cur_rewards[i, :]
            for j, machine in enumerate(sim.machine_list):
                distances[i].append(vis.fr_metric(machine.reward_probabilities,
                                                  sim.model.estimated_machine_reward_distribution[j, :]))
        regrets[...] += optimal_reward - cur_rewards
    convergence_list /= (max_seed - min_seed + 1)
    reward_sums /= (max_seed - min_seed + 1)
    regrets /= (max_seed - min_seed + 1)
    return [(sim_titles[i], convergence_list[i, :]) for i in range(len(model_type_list))], \
           [(sim_titles[i], reward_sums[i, :]) for i in range(len(model_type_list))], \
           [(sim_titles[i], distances[i]) for i in range(len(model_type_list))], \
           [(sim_titles[i], regrets[i]) for i in range(len(model_type_list))]


# %% lambda comparison
model_type_list = [ModelType.UCB_NORMAL,
                   ModelType.THOMPSON_NORMAL,
                   ModelType.OPTIMAL_MODEL,
                   ModelType.LAMBDA,
                   ModelType.LAMBDA,
                   ModelType.LAMBDA]
model_parameters_list = [{},
                         {},
                         {},
                         {"lambda_handle": .1},
                         {"lambda_handle": .5},
                         {"lambda_handle": .9}]
lambda_convergences, lambda_rewards, lambda_fr_metrics, lambda_regret = average_over_seeds(model_type_list,
                                                                                           model_parameters_list)
vis.plot_average_over_seeds(lambda_convergences, lambda_rewards, lambda_fr_metrics, lambda_regret)

# %% beta plus model comparison
model_type_list = [ModelType.LAMBDA,
                   ModelType.OPTIMAL_MODEL,
                   ModelType.LAMBDA_BETA_PLUS_NORMALIZED,
                   ModelType.LAMBDA_BETA_PLUS_NORMALIZED,
                   ModelType.LAMBDA_BETA_PLUS_NORMALIZED,
                   ModelType.LAMBDA_BETA_PLUS_NORMALIZED,
                   ModelType.LAMBDA_BETA_PLUS_NORMALIZED]
model_parameters_list = [{"lambda_handle": .5},
                         {},
                         {"lambda_handle": .5, "beta_handle": .1},
                         {"lambda_handle": .5, "beta_handle": .3},
                         {"lambda_handle": .5, "beta_handle": .5},
                         {"lambda_handle": .5, "beta_handle": .7},
                         {"lambda_handle": .5, "beta_handle": .9}]
lambda_beta_convergences, lambda_beta_rewards, lambda_beta_fr_metrics, lambda_beta_regret = average_over_seeds(
    model_type_list,
    model_parameters_list)
vis.plot_average_over_seeds(lambda_beta_convergences, lambda_beta_rewards, lambda_beta_fr_metrics, lambda_beta_regret)

# %% beta model comparison
model_type_list = [ModelType.LAMBDA,
                   ModelType.OPTIMAL_MODEL,
                   ModelType.LAMBDA_BETA,
                   ModelType.LAMBDA_BETA,
                   ModelType.LAMBDA_BETA,
                   ModelType.LAMBDA_BETA,
                   ModelType.LAMBDA_BETA]
model_parameters_list = [{"lambda_handle": .5},
                         {},
                         {"lambda_handle": .5, "beta_handle": .1},
                         {"lambda_handle": .5, "beta_handle": 1},
                         {"lambda_handle": .5, "beta_handle": 10},
                         {"lambda_handle": .5, "beta_handle": 100},
                         {"lambda_handle": .5, "beta_handle": 1000}]
lambda_beta_convergences, lambda_beta_rewards, lambda_beta_fr_metrics, lambda_beta_regret = average_over_seeds(
    model_type_list,
    model_parameters_list)
vis.plot_average_over_seeds(lambda_beta_convergences, lambda_beta_rewards, lambda_beta_fr_metrics, lambda_beta_regret)

# %% beta-lambda heatmap comparison
model_type_list = [ModelType.OPTIMAL_MODEL] + [ModelType.LAMBDA_BETA for i in range(25)]
lambda_list = np.arange(0, 1, 2e-1)
beta_list = np.linspace(1e-1, 100, lambda_list.size)
# beta_list = lambda_list.copy()
model_parameters_list = [{}] + [{"lambda_handle": i, "beta_handle": j} for i in lambda_list for j
                                in beta_list]
lambda_beta_convergences, lambda_beta_rewards, lambda_beta_fr_metrics, lambda_beta_regret = average_over_seeds(
    model_type_list,
    model_parameters_list)
# %%

# %%
vis.plot_lambda_beta_surface(beta_list, lambda_list, lambda_beta_regret[1:])

# %%# %% lambda beta model comparisons
model_type_list = [ModelType.OPTIMAL_MODEL,
                   ModelType.LAMBDA_BETA,
                   ModelType.LAMBDA_BETA_PLUS]
model_parameters_list = [{},
                         {"lambda_handle": .4, "beta_handle": 60},
                         {"lambda_handle": .4, "beta_handle": 0.5}]
lambda_beta_convergences, lambda_beta_rewards, lambda_beta_fr_metrics, lambda_beta_regret = average_over_seeds(
    model_type_list,
    model_parameters_list)
vis.plot_average_over_seeds(lambda_beta_convergences, lambda_beta_rewards, lambda_beta_fr_metrics, lambda_beta_regret)
# %% final model comparison
model_type_list = [ModelType.OPTIMAL_MODEL,
                   ModelType.LAMBDA_BETA,
                   ModelType.BETA_STOCHASTIC]
model_parameters_list = [{},
                         {"lambda_handle": .4, "beta_handle": 60},
                         {"beta_handle": 60, "theta": 0.5}]
lambda_beta_convergences, lambda_beta_rewards, lambda_beta_fr_metrics, lambda_beta_regret = average_over_seeds(
    model_type_list,
    model_parameters_list)
vis.plot_average_over_seeds(lambda_beta_convergences, lambda_beta_rewards, lambda_beta_fr_metrics, lambda_beta_regret)
# %%

""" 
Models:
 * Lambda-beta model - add learning rate for beta, decreasing as trials progress
 * Stochastic beta modal - change theta according to learning rate as trials progress

regret plot - optimal-model, calculate and show slope

Which comparisons should we make - all comparisons include entropyGain + optimal
 * Normal VS entropy
 * Lambda with handle changes VS baseline & normal
 * Lambda-Beta with handle changes VS baseline & normal
 * mean regret over lambda over beta

 * 
"""
