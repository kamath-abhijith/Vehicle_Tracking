'''

CONSTANT DETECTION RATE NEYMANN-PEARSON
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
    description = "CDAR DETECTOR AT THE ENTRY"
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
# ambient_mean = 0.1
est_ambient_mean = 0.6
dc = 1

# %% MONTE CARLO SIMULATIONS

Ms = np.arange(1,4)
true_PD = np.linspace(0.01, 0.99, 100)
true_PFA = np.zeros((len(true_PD), len(Ms)))
est_PFA = np.zeros((len(true_PD), len(Ms)))
est_PD = np.zeros((len(true_PD), len(Ms)))

for itr, PD in tqdm(enumerate(true_PD)):
    for _, M in enumerate(Ms):
        threshold = np.sqrt(noise_var/(N*M))*norm.isf(1-PD) + est_ambient_mean

        stats_H0 = utils.mean_stat_H0(NUM_STATS, ambient_mean=ambient_mean, M=M)
        stats_H1 = utils.mean_stat_H1(NUM_STATS, ambient_mean=ambient_mean, M=M)

        false_alarms = sum(stats_H0 < threshold)
        detections = sum(stats_H1 < threshold)

        est_PFA[itr, M-1] = false_alarms / NUM_STATS
        est_PD[itr, M-1] = detections / NUM_STATS

        true_PFA[itr, M-1] = 1-norm.sf((threshold-est_ambient_mean-dc)/np.sqrt(noise_var/(N*M)))

# %% PLOTS

os.makedirs('./../results/CDR/', exist_ok=True)
path = './../results/CDR/'

plt.figure(figsize=(8,8))
ax = plt.gca()
utils.plot_signal(true_PD, true_PD, ax=ax, plot_colour='green', show=False)
utils.plot_signal(true_PD, est_PD[:,0], ax=ax, plot_colour='red', line_width=4,
    legend_label=r'$M=1$', show=False)
utils.plot_signal(true_PD, est_PD[:,1], ax=ax, plot_colour='magenta', line_width=4,
    legend_label=r'$M=2$', show=False)
utils.plot_signal(true_PD, est_PD[:,2], ax=ax, plot_colour='blue', line_width=4,
    title_text=r'$D_{\text{entry}}$, $B_t=%.1f$'%(ambient_mean), legend_loc='lower right',
    legend_label=r'$M=3$', yaxis_label=r'$\hat{P}_{D}$', xaxis_label=r'$\beta$',
    show=False, save=path+'CDR_PD_Bt_'+str(ambient_mean))

plt.figure(figsize=(8,8))
ax = plt.gca()
utils.plot_signal(true_PFA[:,0], true_PD, ax=ax, plot_colour='red',
    line_style='--', legend_label=r'$M=1$', show=False)
utils.plot_signal(true_PFA[:,1], true_PD, ax=ax, plot_colour='magenta',
    line_style='--', legend_label=r'$M=2$', show=False)
utils.plot_signal(true_PFA[:,2], true_PD, ax=ax, plot_colour='blue',
    line_style='--', legend_label=r'$M=3$', show=False)
utils.plot_signal(est_PFA[:,0], est_PD[:,0], ax=ax, plot_colour='red', line_width=4,
    legend_label=r'$M=1$', show=False)
utils.plot_signal(est_PFA[:,1], est_PD[:,1], ax=ax, plot_colour='magenta', line_width=4,
    legend_label=r'$M=2$', show=False)
utils.plot_signal(est_PFA[:,2], est_PD[:,2], ax=ax, plot_colour='blue', line_width=4,
    legend_label=r'$M=3$', legend_loc='lower right', legend_ncol=2,
    title_text=r'ROC OF $D_{\text{entry}}$, $B_t=%.1f$'%(ambient_mean),
    yaxis_label=r'$P_{D}$', xaxis_label=r'$P_{FA}$', show=False,
    save=path+'CDR_ROC_Bt_'+str(ambient_mean))

# %%
