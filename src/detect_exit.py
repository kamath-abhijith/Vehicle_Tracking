'''

CONSTANT FALSE_ALARM RATE NEYMANN-PEARSON
DETECTOR FOR VEHICLE DETECTION BASED ON
SIGNAL MEAN

AUTHOR: ABIJITH J. KAMATH
abijithj@iisc.ac.in, kamath-abhijith.github.io

'''

# %% LOAD LIBRARIES

import os
import numpy as np

from tqdm import tqdm

from scipy.stats import norm

from matplotlib import style
from matplotlib import rcParams
from matplotlib import pyplot as plt

import utils

# %% PLOT SETTINGS

plt.style.use(['science','ieee'])

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["cm"],
    "mathtext.fontset": "cm",
    "font.size": 24})

# %% PARAMETERS

N = 5
M = 1
NUM_STATS = 1000

noise_var = 1
ambient_mean = 0.6
dc = 1

# %% MONTE CARLO SIMULATIONS

true_PFA = np.linspace(0.01, 0.99, 100)
true_PD = np.zeros(len(true_PFA))
est_PFA = np.zeros(len(true_PFA))
est_PD = np.zeros(len(true_PFA))

for itr, PFA in tqdm(enumerate(true_PFA)):
    threshold = norm.isf(PFA)*np.sqrt(noise_var/N) + ambient_mean

    stats_H0 = utils.mean_stat_H0(NUM_STATS, ambient_mean=ambient_mean)
    stats_H1 = utils.mean_stat_H1(NUM_STATS, ambient_mean=ambient_mean)

    false_alarms = sum(stats_H0 > threshold)
    detections = sum(stats_H1 > threshold)

    est_PFA[itr] = false_alarms / NUM_STATS
    est_PD[itr] = detections / NUM_STATS

    true_PD[itr] = norm.sf((threshold-dc-ambient_mean)/np.sqrt(noise_var/N))

# %% PLOTS

os.makedirs('./../results/', exist_ok=True)
path = './../results/'

plt.figure(figsize=(8,8))
ax = plt.gca()
utils.plot_signal(true_PFA, true_PFA, ax=ax, plot_colour='green', show=False)
utils.plot_signal(true_PFA, est_PFA, ax=ax, plot_colour='blue',
    title_text=r'EXIT  DETECTOR', yaxis_label=r'ESTIMATED $P_{D}$',
    xaxis_label=r'THEORETICAL $P_{D}$', show=True)

plt.figure(figsize=(8,8))
ax = plt.gca()
utils.plot_signal(true_PFA, true_PD, ax=ax, plot_colour='green', show=False)
utils.plot_signal(est_PFA, est_PD, ax=ax, plot_colour='blue',
    title_text=r'ROC OF EXIT DETECTOR', yaxis_label=r'$P_{D}$',
    xaxis_label=r'$P_{FA}$', show=True)

# %%
