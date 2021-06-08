'''

TOOLS FOR VEHICLE DETECTION AND TRACKING

AUTHOR: ABIJITH J. KAMATH
abijithj@iisc.ac.in, kamath-abhijith.github.io

'''

# %% LOAD LIBRARIES

import numpy as np

from matplotlib import pyplot as plt

from scipy.stats import multivariate_normal

# %% PLOTTING FUNCTIONS

def plot_signal(x, y, ax=None, plot_colour='blue', xaxis_label=None,
    yaxis_label=None, title_text=None, legend_label=None, legend_show=True,
    legend_loc='upper right', line_style='-', line_width=None,
    show=False, xlimits=[0,1], ylimits=[0,1], save=None):
    '''
    Plots signal with abscissa in x and ordinates in y 

    '''
    if ax is None:
        fig = plt.figure(figsize=(12,6))
        ax = plt.gca()

    plt.plot(x, y, linestyle=line_style, linewidth=line_width, color=plot_colour,
        label=legend_label)
    if legend_label and legend_show:
        plt.legend(loc=legend_loc, frameon=True, framealpha=0.8, facecolor='white')
    plt.xlabel(xaxis_label)
    plt.ylabel(yaxis_label)

    plt.xlim(xlimits)
    plt.ylim(ylimits)
    plt.title(title_text)

    if save:
        plt.savefig(save + '.pdf', format='pdf')

    if show:
        plt.show()

    return

# %% STATISTICS

def mean_stat_H0(num_stats, ambient_mean, dc=1, N=5, M=1, noise_var=1 ):
    '''
    Generates signal mean statistic for no vehicle hypothesis

    :param num_stats: number of realisations
    :param ambient_mean: mean due to ambient light
    :param dc: dc value due to source
    :param N: number of measurements
    :param M: number of LEDs
    :param noise_var: variance of AWGN

    :return: mean statistics

    '''
    
    stats = np.zeros(num_stats)
    for itr in range(num_stats):
        noise = np.sqrt(noise_var)*np.random.randn(N*M)

        stats[itr] = np.mean(ambient_mean + noise)

    return stats

def mean_stat_H1(num_stats, ambient_mean, dc=1, N=5, M=1, noise_var=1 ):
    '''
    Generates signal mean statistic for vehicle present hypothesis

    :param num_stats: number of realisations
    :param ambient_mean: mean due to ambient light
    :param dc: dc value due to source
    :param N: number of measurements
    :param M: number of LEDs
    :param noise_var: variance of AWGN

    :return: mean statistics

    '''
    
    stats = np.zeros(num_stats)
    for itr in range(num_stats):
        noise = np.sqrt(noise_var)*np.random.randn(N*M)

        stats[itr] = np.mean(dc + ambient_mean + noise)

    return stats