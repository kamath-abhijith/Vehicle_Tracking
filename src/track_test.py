'''

KALMAN FILTER FOR POSITION ESTIMATION
USING POSITION AND VELOCITY MEASUREMENTS
FOR AUTOMATED DRIVING TEST

AUTHOR: ABIJITH J. KAMATH
abijithj@iisc.ac.in

'''

# %% LOAD LIBRARIES

import os
import argparse
import numpy as np

from tqdm import tqdm

from scipy import io
from scipy.stats import norm

from matplotlib import style
from matplotlib import rcParams
from matplotlib import pyplot as plt

import utils

# %% PARSE ARGUMENTS
parser = argparse.ArgumentParser(
    description = "KALMAN FITLER WITH POSITION AND VELOCITY MEASUREMENTS"
)

parser.add_argument('--data', help="dataset", type=int, default=1)

args = parser.parse_args()
dataset = args.data

# %% PLOT SETTINGS

plt.style.use(['science','ieee'])

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["cm"],
    "mathtext.fontset": "cm",
    "font.size": 24})

# %% LOAD DATA

true_data = io.loadmat('./../dataset/trace_ideal.mat')
sample_data = io.loadmat('./../dataset/trace_1.mat')
radar_data = io.loadmat('./../dataset/Test'+str(dataset)+'.mat')
meas_noise = 0.1

true_trace = true_data['true_trace']
sample_trace = sample_data['x']
measurements = radar_data['radar_measurement']

# %% TRACKING PARAMETERS

# Initialise variables
state_dim, num_points = true_trace.shape
update_state = np.zeros((state_dim, num_points))
update_statecov = np.zeros((state_dim, state_dim, num_points))
predict_state = np.zeros((state_dim, num_points))
predict_statecov = np.zeros((state_dim, state_dim, num_points))

# Define transition matrices
time_step = 0.1
state_mat = np.zeros((state_dim, state_dim, num_points))
for idx in range(num_points):
    if idx<120 or (idx>152 and idx<312) or idx>344:
        state_mat[:,:,idx] = np.matrix([[1, 0, 0, 0],
                                        [0, 1, 0, time_step],
                                        [0, 0, 0, 0],
                                        [0, 0, 0, 1]])
    elif idx==120 or idx==312:
        state_mat[:,:,idx] = np.matrix([[1, 0, 0, 0],
                                        [0, 1, 0, time_step],
                                        [0, 0, 0, -1],
                                        [0, 0, 0, 0]])
    elif idx==152 or idx==344:
        state_mat[:,:,idx] = np.matrix([[1, 0, time_step, 0],
                                        [0, 1, 0, 0],
                                        [0, 0, 0, 0],
                                        [0, 0, -1, 0]])
    elif (idx>120 and idx<152) or (idx>312 and idx<344):
        state_mat[:,:,idx] = np.matrix([[1, 0, time_step, 0],
                                        [0, 1, 0, 0],
                                        [0, 0, 1, 0],
                                        [0, 0, 0, 0]])
meas_mat = np.eye(state_dim)

# Define noise covariances
noise_std = 0.1
state_noise_cov = noise_std*np.eye(state_dim)
meas_noise_cov = meas_noise*np.eye(state_dim)

# %% TRACKING

# Initialisation
update_state[:,0] = np.array([0.0, 0.0, 0.0, 0.0])
update_statecov[:,:,0] = 0.0*np.eye(state_dim)

# Kalman filtering
for idx in tqdm(range(num_points-1)):

    predict_state[:,idx], predict_statecov[:,:,idx] = utils.kf_predict(\
        update_state[:,idx], state_mat[:,:,idx], update_statecov[:,:,idx],
        state_noise_cov)

    update_state[:,idx+1], update_statecov[:,:,idx+1] = utils.kf_update(\
        measurements[:,idx], predict_state[:,idx], predict_statecov[:,:,idx],
        meas_mat, meas_noise_cov)

# %% ERROR

kalman_error = np.linalg.norm(predict_state[:2,:] - true_trace[:2,:], axis=0)**2
radar_error = np.linalg.norm(measurements[:2,:] - true_trace[:2,:], axis=0)**2

threshold = 3.125
if sum(kalman_error>threshold)==0:
    result = r'PASS'
elif sum(kalman_error>threshold)>0:
    result = r'FAIL'


# %% PLOTS

os.makedirs('./../results/KF/ex5/', exist_ok=True)
path = './../results/KF/ex5/'

plt.figure(figsize=(12,6))
ax = plt.gca()
utils.plot_trace(true_trace, ax=ax, plot_colour='green', line_style='-',
    legend_label=r'TRUE TRACE', show=False)
utils.plot_trace(measurements, ax=ax, plot_colour='red', line_style='dotted',
    legend_label=r'MEASUREMENTS', show=False)
utils.plot_trace(predict_state, ax=ax, plot_colour='blue', line_style=None,
    line_width=1, fill_style='none', legend_label=r'FILTERED TRACE', show=False,
    title_text=r'RESULT: '+result, save=path+'KF_Pos_Trace_Data_'+str(dataset))

plt.figure(figsize=(12,6))
ax = plt.gca()
utils.plot_signal(np.arange(num_points), threshold*np.ones(num_points), ax=ax,
    plot_colour='green', legend_label=r'THRESHOLD', show=False)
utils.plot_signal(np.arange(num_points), kalman_error, ax=ax,
    plot_colour='blue', legend_label=r'KALMAN ERROR', show=False)
utils.plot_signal(np.arange(num_points), radar_error, ax=ax,
    xlimits=[0,num_points], ylimits=[0,3.5], plot_colour='red',
    xaxis_label=r'$n$', yaxis_label=r'$\Vert\mathbf{p}-\hat{\mathbf{p}}\Vert_2^2$',
    legend_label=r'RADAR ERROR', title_text=r'RESULT: '+result, show=False,
    save=path+'KF_Pos_Errors_Data_'+str(dataset))

# %%
