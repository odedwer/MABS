import matplotlib.pyplot as plt
import numpy as np


def plot_convergences(simulation_list, window_size=None) -> plt.Figure:
    """
    Plots the convergence rates of given simulations
    :param simulation_list: list of Simulation objects, whose run_simulation() method was called
    :param window_size: int or None, window size for convergence rate, all data points from start to t if None
    :return: the figure with the plots
    """
    fig = plt.figure()
    ax = fig.subplots()
    for sim in simulation_list:
        ax.plot(sim.get_convergence_rate(window_size), ':', label=sim.type, linewidth=1)
    ax.legend()
    ax.set_title("Machine switch rate")
    ax.set_xlabel(r"Trial")
    ax.set_ylabel("Variance " + (f"(window size = {window_size})" if window_size else "(all available trials)"))
    return fig


def plot_rewards(simulation_list) -> plt.Figure:
    """
    Plots the cumulative reward of given simulations
    :param simulation_list: list of Simulation objects, whose run_simulation() method was called
    :return: the figure with the plots
    """
    fig = plt.figure()
    ax = fig.subplots()
    for sim in simulation_list:
        ax.plot(sim.get_reward_sum(), linestyle=":", label=sim.type)
    ax.legend()
    ax.set_title("Cumulative Reward")
    ax.set_xlabel(r"Trial")
    ax.set_ylabel(r"Cumulative Reward")
    return fig


def plot_reward_ratios(simulation_list):
    """
    Plots the time by time ratio of cumulative reward of given simulations, for all pairs
    :param simulation_list: list of Simulation objects, whose run_simulation() method was called
    :return: the figure with the plots
    """
    from itertools import combinations
    fig = plt.figure()
    ax = fig.subplots()
    rewards = [sim.get_reward_sum() for sim in simulation_list]
    for indices in list(combinations(range(len(rewards)), 2)):
        ax.plot(rewards[indices[0]] / rewards[indices[1]], linestyle=":", linewidth=1,
                label=f"{simulation_list[indices[0]].type}/{simulation_list[indices[1]].type}")
    ax.legend()
    ax.set_title("Cumulative Reward Ratio")
    ax.set_xlabel(r"Trial")
    ax.set_ylabel(r"Cumulative Reward Ratio")
    return fig


def fix_sig(sig):
    """
    fixes a given signal in-place, changing all Nan and inf to 0
    :param sig: array to fix
    """
    sig[(sig == np.NaN) | (sig == np.inf) | (sig == -np.inf)] = 0


def fr_metric(pk, qk):
    """
    Computes Fisher-Rao metric on given distributions.
    :param pk: 1D numpy array
    :param qk: 1D numpy array
    :return:
    """
    fix_sig(pk)
    fix_sig(qk)
    # make sure the given arrays are probability distributions, summing to 1
    pk = pk / np.sum(pk)
    qk = qk / np.sum(qk)
    return np.arccos(np.sum(np.sqrt(pk) * np.sqrt(qk)))


def plot_distance_of_distribution_estimations(sim_list):
    fig = plt.figure()
    nrow = int(np.floor(np.sqrt(len(sim_list))))
    ncol = int(np.ceil(len(sim_list) / nrow))
    axs = fig.subplots(nrow, ncol)
    if not np.iterable(axs):
        axs = [axs]
    else:
        axs = axs.ravel()
    for j, sim in enumerate(sim_list):
        distances = np.zeros_like(sim.machine_list)
        for i, machine in enumerate(sim.machine_list):
            distances[i] = fr_metric(machine.reward_probabilities,
                                     sim.model.estimated_machine_reward_distribution[i, :])
        axs[j].hist(distances, label=sim.type, linestyle=":", alpha=.3, bins=np.arange(0, 1.025, 0.025))
        axs[j].set_xlim(0, 1)
        axs[j].set_title(sim.type)
        axs[j].set_xlabel(r"Fisher-Rao metric ($\in [0,1]$)")
        axs[j].set_ylabel(r"# Occurences")
        axs[j].legend()
    fig.suptitle("Distance between estimated and real machine reward distributions")
    return fig
