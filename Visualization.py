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
