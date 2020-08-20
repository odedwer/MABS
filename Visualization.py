import matplotlib.pyplot as plt
import numpy as np

SIMULATION = "s"

FIGSIZE = (50, 50)

colormap = plt.cm.jet
FRAME_ALPHA = 0.6
from matplotlib.font_manager import FontProperties

fontP = FontProperties()


def plot_convergences(simulation_list, list_type="s", window_size=None) -> plt.Figure:
    """
    Plots the convergence rates of given simulations
    :param simulation_list: list of Simulation objects, whose run_simulation() method was called
    :param window_size: int or None, window size for convergence rate, all data points from start to t if None
    :param list_type str, s for simulation, d for data. s means simulation list contains Simulation objects,
            d means simulation list contains tuples of label,data
    :return: the figure with the plots
    """
    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.subplots()
    global colormap, fontP
    fontP.set_size('small' if len(simulation_list) > 5 else 'medium')
    ax.set_prop_cycle(plt.cycler('color', colormap(np.linspace(0, 1, len(simulation_list)))))
    for sim in simulation_list:
        ax.plot(sim.get_convergence_rate(window_size) if list_type == SIMULATION else sim[1], ':',
                label=sim.type if list_type == SIMULATION else sim[0], linewidth=3)
    ax.legend(prop=fontP, framealpha=FRAME_ALPHA)
    ax.set_title("Machine switch rate")
    ax.set_xlabel(r"Trial")
    ax.set_ylabel("Variance " + (f"(window size = {window_size})" if window_size else "(all available trials)"))
    return fig


def plot_rewards(simulation_list, list_type="s") -> plt.Figure:
    """
    Plots the cumulative reward of given simulations
    :param simulation_list: list of Simulation objects, whose run_simulation() method was called
    :param list_type str, s for simulation, d for data. s means simulation list contains Simulation objects,
            d means simulation list contains tuples of label,data
    :return: the figure with the plots
    """
    global colormap, fontP
    fontP.set_size('small' if len(simulation_list) > 5 else 'medium')
    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.subplots()
    ax.set_prop_cycle(plt.cycler('color', colormap(np.linspace(0, 1, len(simulation_list)))))
    for sim in simulation_list:
        ax.plot(sim.get_reward_sum() if list_type == SIMULATION else sim[1], linestyle=":",
                label=sim.type if list_type == SIMULATION else sim[0])
    ax.legend(prop=fontP, framealpha=FRAME_ALPHA)
    ax.set_title("Cumulative Reward")
    ax.set_xlabel(r"Trial")
    ax.set_ylabel(r"Cumulative Reward")
    return fig


def plot_reward_ratios(simulation_list, list_type="s"):
    """
    Plots the time by time ratio of cumulative reward of given simulations, for all pairs
    :param simulation_list: list of Simulation objects, whose run_simulation() method was called
    :param list_type str, s for simulation, d for data. s means simulation list contains Simulation objects,
            d means simulation list contains tuples of label,data
    :return: the figure with the plots
    """
    from itertools import combinations
    fig = plt.figure(figsize=FIGSIZE)
    global colormap, fontP
    fontP.set_size('small' if len(simulation_list) > 5 else 'medium')
    ax = fig.subplots()
    ax.set_prop_cycle(plt.cycler('color', colormap(np.linspace(0, 1, len(simulation_list)))))
    rewards = [sim.get_reward_sum() for sim in simulation_list] if list_type == SIMULATION else [sim for sim in
                                                                                                 simulation_list]
    for indices in list(combinations(range(len(rewards)), 2)):
        ax.plot(rewards[indices[0]] / rewards[indices[1]] if list_type == SIMULATION else rewards[indices[0]][1] /
                                                                                          rewards[indices[1]][1],
                linestyle=":", linewidth=1,
                label=f"{simulation_list[indices[0]].type}/{simulation_list[indices[1]].type}" if list_type == SIMULATION
                else f"{simulation_list[indices[0]][0]}/{simulation_list[indices[1]][0]}")
    ax.legend(prop=fontP, framealpha=FRAME_ALPHA)
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


def plot_distance_of_distribution_estimations(sim_list, list_type="s"):
    fig = plt.figure(figsize=FIGSIZE)
    nrow = int(np.floor(np.sqrt(len(sim_list))))
    ncol = int(np.ceil(len(sim_list) / nrow))
    axs = fig.subplots(nrow, ncol, sharex='col')
    if not np.iterable(axs):
        axs = [axs]
    else:
        axs = axs.ravel()
    for j, sim in enumerate(sim_list):
        if list_type == SIMULATION:
            distances = np.zeros_like(sim.machine_list)
            for i, machine in enumerate(sim.machine_list):
                distances[i] = fr_metric(machine.reward_probabilities,
                                         sim.model.estimated_machine_reward_distribution[i, :])
        else:
            distances = sim[1]
        axs[j].hist(distances, label=sim.type if list_type == SIMULATION else sim[0], linestyle=":", alpha=.3,
                    bins=np.arange(0, 1.025, 0.025))
        axs[j].set_xlim(0, 1)
        axs[j].set_title(sim.type, size=7)
        if j > (((nrow - 1) * ncol) - 1):  # only last row
            axs[j].set_xlabel(r"Fisher-Rao metric ($\in [0,1]$)")
        if j % ncol == 0:
            axs[j].set_ylabel(r"# Occurences")
    fig.suptitle("Distance between estimated and real machine reward distributions")
    return fig
