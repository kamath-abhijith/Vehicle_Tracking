'''

TOOLS FOR VEHICLE DETECTION AND TRACKING

AUTHOR: ABIJITH J. KAMATH
abijithj@iisc.ac.in, kamath-abhijith.github.io

'''

# %% LOAD LIBRARIES

import numpy as np

from matplotlib import pyplot as plt
from celluloid import Camera

from scipy.stats import multivariate_normal

from tqdm import tqdm

# %% PLOTTING FUNCTIONS

def plot_signal(x, y, ax=None, plot_colour='blue', xaxis_label=None,
    yaxis_label=None, title_text=None, legend_label=None, legend_show=True,
    legend_loc='upper right', legend_ncol=1, line_style='-', line_width=None,
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
        plt.legend(ncol=legend_ncol, loc=legend_loc, frameon=True, framealpha=0.8,
            facecolor='white')
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

def plot_trace(state_var, ax=None, plot_colour='blue', marker='o',
    fill_style='full', xaxis_label=r'$y$', yaxis_label=r'$x$',
    title_text=None, legend_label=None, legend_show=True,
    legend_loc='center right', line_style='-', line_width=4, show=False,
    xlimits=[-2,22], ylimits=[-2,10], save=None):
    '''
    Plots trace using the state variable

    '''
    if ax is None:
        fig = plt.figure(figsize=(12,6))
        ax = plt.gca()

    plt.plot(state_var[1,:], state_var[0,:], linestyle=line_style,
        marker=marker, fillstyle=fill_style, linewidth=line_width,
        color=plot_colour, label=legend_label)

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

def make_trace_video(true, measurements, filter, xlimits=[-2,22],
    ylimits=[-2,10],
    save=None):
    '''
    Makes video with static true trace and dynamic
    measurements and filter output

    '''
    
    fig = plt.figure(figsize=(12,6))
    ax = plt.gca()
    camera = Camera(fig)

    _, num_points = true.shape
    for i in tqdm(range(num_points)):
        ax.plot(true[1,:], true[0,:], c='green', linewidth=4, linestyle='-',
            marker='o')
        ax.plot(measurements[1,:i], measurements[0,:i], c='red', linewidth=4,
            linestyle='dotted', marker='o')
        ax.plot(filter[1,:i], filter[0,:i], c='blue', linewidth=1,
            linestyle=None, marker='o', fillstyle='none')
        camera.snap()

    plt.xlim(xlimits)
    plt.ylim(ylimits)

    animation = camera.animate()
    animation.save('trace.mp4')

    fig = plt.figure(figsize=(12,6))
    ax = plt.gca()
    camera = Camera(fig)

    _, num_points = true.shape
    for i in tqdm(range(num_points)):
        ax.plot(true[2,:], c='green', linewidth=4, linestyle='-',
            marker='o')
        ax.plot(measurements[2,:i], c='red', linewidth=4,
            linestyle='dotted')
        ax.plot(filter[2,:i], c='blue', linewidth=1,
            linestyle=None, fillstyle='none')
        camera.snap()

    plt.xlim([0,num_points])
    plt.ylim([-3,3])

    animation = camera.animate()
    animation.save('xvel.mp4')

    fig = plt.figure(figsize=(12,6))
    ax = plt.gca()
    camera = Camera(fig)

    _, num_points = true.shape
    for i in tqdm(range(num_points)):
        ax.plot(true[3,:], c='green', linewidth=4, linestyle='-',
            marker='o')
        ax.plot(measurements[3,:i], c='red', linewidth=4,
            linestyle='dotted')
        ax.plot(filter[3,:i], c='blue', linewidth=1,
            linestyle=None, fillstyle='none')
        camera.snap()

    plt.xlim([0,num_points])
    plt.ylim([-3,3])

    animation = camera.animate()
    animation.save('yvel.mp4')

    return

# %% STATISTICS

def mean_stat_H0(num_stats, ambient_mean, dc=1, N=5, M=1, noise_var=1):
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

        stats[itr] = np.mean(dc + ambient_mean + noise)

    return stats

def mean_stat_H1(num_stats, ambient_mean, dc=1, N=5, M=1, noise_var=1):
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

        stats[itr] = np.mean(ambient_mean + noise)

    return stats

# %% KALMAN FILTER

def kf_predict(state, state_mat, state_cov, noise_cov):
    '''
    Predict state, given past states

    :param state: past state variable
    :param state_mat: past state matrix
    :param state_cov: past covariance of state
    :param noise_var: noise variance

    :returns: predicted state variable, predicted error covariance

    '''

    prediction = np.dot(state_mat, state)
    covariance = np.dot(np.dot(state_mat, state_cov), state_mat.T) + noise_cov

    return prediction, covariance

def kf_update(meas, state, state_cov, meas_mat, meas_cov):
    '''
    Update the state variable given measurements

    :param meas: measurements
    :param state: state variable
    :param state_cov: error covariance in state
    :param meas_mat: measurement matrix
    :param meas_cov: error covariance in measurements

    :returns: updated state variable, error covariance

    '''

    S = np.dot(np.dot(meas_mat, state_cov), meas_mat.T) + meas_cov
    gain = np.dot(np.dot(state_cov, meas_mat.T), np.linalg.inv(S))
    updated = state + np.dot(gain, meas-np.dot(meas_mat, state))
    covariance = np.dot((np.eye(meas_mat.shape[1]) - np.dot(gain, meas_mat)), state_cov)

    return updated, covariance
