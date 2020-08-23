import numpy as np
from tqdm import tqdm
import Visualization as vis
from ModelFactory import ModelType
import Simulation as s
from importlib import reload


def reload_imports():
    reload(vis)
    reload(s)


MIN_SEED = 90
MAX_SEED = 99
N = 20
K = 1
T = 10000
POSSIBLE_REWARDS = np.asarray([5, 10, 20, 30, 50])
get_reward_probabilities = lambda: np.random.dirichlet(np.ones_like(POSSIBLE_REWARDS))


def average_over_seeds(model_type_list, model_parameters_list, n=N, k=K, t=T, possible_rewards=POSSIBLE_REWARDS,
                       get_reward_probabilities=get_reward_probabilities,
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
            sim = s.Simulation(n, k, t, possible_rewards, get_reward_probabilities, model_type, **model_parameters)
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


# %% beta comparison L\AEG-UCB
model_num = 3
model_type_list = [ModelType.OPTIMAL_BASELINE, ModelType.UCB] + [ModelType.AEG_UCB for _ in range(model_num)] + [
    ModelType.LEG_UCB for _ in range(model_num)]
beta_list_aeg = np.linspace(0.1, 0.9, model_num)
beta_list_leg = np.linspace(5, 25, model_num)
model_parameters_list = [{}, {}] + [{"beta_handle": b} for b in beta_list_aeg] + [{"beta_handle": b} for b in
                                                                                  beta_list_leg]
beta_convergences, beta_rewards, beta_fr_metrics, beta_regret = average_over_seeds(model_type_list,
                                                                                   model_parameters_list)
vis.plot_average_over_seeds(beta_convergences, beta_rewards, beta_fr_metrics, beta_regret, "UCB VS EG-UCB")

# %% beta comparison L\AEG-TS
model_num = 3
model_type_list = [ModelType.OPTIMAL_BASELINE, ModelType.TS] + [ModelType.AEG_TS for _ in range(model_num)] + [
    ModelType.LEG_TS for _ in range(model_num)]
beta_list_aeg = np.linspace(0.1, 0.9, model_num)
beta_list_leg = np.linspace(5, 25, model_num)
model_parameters_list = [{}, {}] + [{"beta_handle": b} for b in beta_list_aeg] + [{"beta_handle": b} for b in
                                                                                  beta_list_leg]
beta_convergences, beta_rewards, beta_fr_metrics, beta_regret = average_over_seeds(model_type_list,
                                                                                   model_parameters_list)
vis.plot_average_over_seeds(beta_convergences, beta_rewards, beta_fr_metrics, beta_regret, "TS VS EG-TS")

# %% beta comparison AEG-TS
model_num = 10
model_type_list = [ModelType.OPTIMAL_BASELINE] + [ModelType.AEG_TS for _ in range(model_num)]
beta_list = np.linspace(0.1, 0.9, model_num)
model_parameters_list = [{}] + [{"beta_handle": b} for b in beta_list]
beta_convergences, beta_rewards, beta_fr_metrics, beta_regret = average_over_seeds(model_type_list,
                                                                                   model_parameters_list)
vis.plot_average_over_seeds(beta_convergences, beta_rewards, beta_fr_metrics, beta_regret, "AEG-TS beta comparison")

# %% beta-lambda heatmap comparison
model_type_list = [ModelType.OPTIMAL_BASELINE] + [ModelType.LEG_LH for i in range(225)]
lambda_list = np.linspace(0.1, 0.9, 15)
beta_list = np.linspace(0.1, 30, lambda_list.size)
# beta_list = lambda_list.copy()
model_parameters_list = [{}] + [{"lambda_handle": i, "beta_handle": j} for i in lambda_list for j
                                in beta_list]
lambda_beta_convergences, lambda_beta_rewards, lambda_beta_fr_metrics, lambda_beta_regret = average_over_seeds(
    model_type_list,
    model_parameters_list)
vis.plot_lambda_beta_surface(beta_list, lambda_list, lambda_beta_rewards[1:], title="Reward Slope")
vis.plot_lambda_beta_surface(beta_list, lambda_list, lambda_beta_regret[1:], title="Regret Slope")

# %%
model_type_list = [ModelType.OPTIMAL_BASELINE] + [ModelType.AEG_LH for i in range(225)]
lambda_list = np.linspace(0.1, 0.9, 15)
beta_list = np.linspace(0.1, 0.9, lambda_list.size)
model_parameters_list = [{}] + [{"lambda_handle": i, "beta_handle": j} for i in lambda_list for j
                                in beta_list]
convergences, rewards, fr_metrics, regret = average_over_seeds(
    model_type_list,
    model_parameters_list)
vis.plot_lambda_beta_surface(beta_list, lambda_list, lambda_beta_rewards[1:], title="Reward Slope")
vis.plot_lambda_beta_surface(beta_list, lambda_list, lambda_beta_regret[1:], title="Regret Slope")
# %%
model_type_list = [ModelType.OPTIMAL_BASELINE] + [ModelType.SMART_MODEL for i in range(225)]
aeg_beta_handle = np.linspace(0.1, 0.9, 15)
leg_beta_handle = np.linspace(10, 40, aeg_beta_handle.size)
# beta_list = lambda_list.copy()
model_parameters_list = [{}] + [{"leg_beta_handle": i, "aeg_beta_handle": j} for i in leg_beta_handle for j
                                in aeg_beta_handle]
lambda_beta_convergences, lambda_beta_rewards, lambda_beta_fr_metrics, lambda_beta_regret = average_over_seeds(
    model_type_list,
    model_parameters_list)
#%%
vis.plot_lambda_beta_surface(leg_beta_handle, aeg_beta_handle, lambda_beta_rewards[1:], title="Reward Slope")
#%%
vis.plot_lambda_beta_surface(leg_beta_handle, aeg_beta_handle, lambda_beta_regret[1:], title="Regret Slope")

# %% lambda hybrid comparisons
model_type_list = [ModelType.OPTIMAL_BASELINE,
                   ModelType.RANDOM_BASELINE,
                   ModelType.UCB,
                   ModelType.TS,
                   ModelType.LH,
                   ModelType.LH,
                   ModelType.LH,
                   ModelType.LH,
                   ModelType.LH]
model_parameters_list = [{},
                         {},
                         {},
                         {},
                         {"lambda_handle": .1},
                         {"lambda_handle": .3},
                         {"lambda_handle": .5},
                         {"lambda_handle": .7},
                         {"lambda_handle": .9}]
lambda_convergences, lambda_rewards, lambda_fr_metrics, lambda_regret = average_over_seeds(model_type_list,
                                                                                           model_parameters_list)
vis.plot_average_over_seeds(lambda_convergences, lambda_rewards, lambda_fr_metrics, lambda_regret,
                            "Hybrid vs Non-Hybrid")

# %% Hybrid vs no hybrid models
model_type_list = [ModelType.OPTIMAL_BASELINE,
                   ModelType.RANDOM_BASELINE,
                   ModelType.UCB,
                   ModelType.TS,
                   ModelType.SMART_MODEL,
                   ModelType.LEG_LH,
                   ModelType.AEG_LH]
model_parameters_list = [{},
                         {},
                         {},
                         {},
                         {},
                         {"lambda_handle": .5, "beta_handle": 15},
                         {"lambda_handle": .7, "beta_handle": .4}]

convergences, rewards, fr_metrics, regret = average_over_seeds(model_type_list, model_parameters_list)
vis.plot_average_over_seeds(convergences, rewards, fr_metrics, regret, "Hybrid VS EG hybrid")
