'''

CONSTANT FALSE_ALARM RATE NEYMANN-PEARSON
DETECTOR FOR VEHICLE DETECTION BASED ON
SIGNAL MEAN

AUTHOR: ABIJITH J. KAMATH
abijithj@iisc.ac.in, kamath-abhijith.github.io

'''

# %% LOAD LIBRARIES

import os
import argparse
import numpy as np

from tqdm import tqdm

from scipy.stats import norm

from matplotlib import style
from matplotlib import rcParams
from matplotlib import pyplot as plt

import utils

# %% PARSE ARGUMENTS
parser = argparse.ArgumentParser(
    description = "CFAR DETECTOR AT THE EXIT"
)

parser.add_argument('--Bt', help="ambient light", type=float, default=0.1)

args = parser.parse_args()
ambient_mean = args.Bt

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
NUM_STATS = 10000

noise_var = 1
# ambient_mean = 0.6
est_ambient_mean = 0.1
dc = 1

# %% MONTE CARLO SIMULATIONS

Ms = np.arange(1,4)
true_PFA = np.linspace(0.01, 0.99, 100)
true_PD = np.zeros((len(true_PFA), len(Ms)))
est_PFA = np.zeros((len(true_PFA), len(Ms)))
est_PD = np.zeros((len(true_PFA), len(Ms)))

for itr, PFA in tqdm(enumerate(true_PFA)):
    for _, M in enumerate(Ms):
        threshold = np.sqrt(noise_var/(N*M))*norm.isf(1-PFA) + est_ambient_mean + dc

        stats_H0 = utils.mean_stat_H0(NUM_STATS, ambient_mean=ambient_mean, M=M)
        stats_H1 = utils.mean_stat_H1(NUM_STATS, ambient_mean=ambient_mean, M=M)

        false_alarms = sum(stats_H0 < threshold)
        detections = sum(stats_H1 < threshold)

        est_PFA[itr, M-1] = false_alarms / NUM_STATS
        est_PD[itr, M-1] = detections / NUM_STATS

        true_PD[itr, M-1] = 1 - norm.sf((threshold-est_ambient_mean)/np.sqrt(noise_var/(N*M)))

# %% PLOTS

os.makedirs('./../results/CFAR/', exist_ok=True)
path = './../results/CFAR/'

# plt.figure(figsize=(8,8))
# ax = plt.gca()
# utils.plot_signal(true_PFA, true_PFA, ax=ax, plot_colour='green', show=False)
# utils.plot_signal(true_PFA, est_PFA[:,0], ax=ax, plot_colour='red', line_width=4,
#     legend_label=r'$M=1$', show=False)
# utils.plot_signal(true_PFA, est_PFA[:,1], ax=ax, plot_colour='magenta', line_width=4,
#     legend_label=r'$M=2$', show=False)
# utils.plot_signal(true_PFA, est_PFA[:,2], ax=ax, plot_colour='blue', line_width=4,
#     title_text=r'$D_{\text{exit}}$, $B_t=%.1f$'%(ambient_mean), legend_loc='upper left',
#     legend_label=r'$M=3$', yaxis_label=r'$\hat{P}_{FA}$', xaxis_label=r'$\alpha$',
#     show=False, save=path+'CFAR_PFA_Bt_'+str(ambient_mean))

plt.figure(figsize=(8,8))
ax = plt.gca()
utils.plot_signal(true_PD[:,0], true_PFA, ax=ax, plot_colour='red',
    line_style='--', legend_label=r'$M=1$', show=False)
utils.plot_signal(true_PD[:,1], true_PFA, ax=ax, plot_colour='magenta',
    line_style='--', legend_label=r'$M=2$', show=False)
utils.plot_signal(true_PD[:,2], true_PFA, ax=ax, plot_colour='blue',
    line_style='--', legend_label=r'$M=3$', show=False)
utils.plot_signal(est_PD[:,0], est_PFA[:,0], ax=ax, plot_colour='red', line_width=4,
    legend_label=r'$M=1$', show=False)
utils.plot_signal(est_PD[:,1], est_PFA[:,1], ax=ax, plot_colour='magenta', line_width=4,
    legend_label=r'$M=2$', show=False)
utils.plot_signal(est_PD[:,2], est_PFA[:,2], ax=ax, plot_colour='blue', line_width=4,
    legend_label=r'$M=3$', legend_loc='upper left', legend_ncol=2,
    title_text=r'ROC OF $D_{\text{exit}}$, $B_t=%.1f$'%(ambient_mean),
    yaxis_label=r'$P_{FA}$', xaxis_label=r'$P_{D}$', show=False,
    save=path+'CFAR_ROC_Bt_'+str(ambient_mean))

# %%
