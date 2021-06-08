Details of the dataset

The file "dataset.zip" contains 8 ".mat" files corresponding containing true positions and radar measurements
of the path traced by the vehicle under different circumstances. Each trace is a $4 \times 384$ matrix whose columns corresponding
to the instantaneous position and velocity of the car at different time instants. Details of each trace is as follows


trace_ideal – contains a variable "true_trace" which is the ideal test track in the shape of the letter ‘H’.
trace_1 – contains a variable "x", which is a sample trace of a vehicle under test.
Radar_med  - contains a variable "y", which is the radar measurements corresponding to “trace_1” with moderate amount of observation noise. We used Q_v = 0.1 I.
Radar_high  - contains a variable "y", which is the radar measurements corresponding to “trace_1” with large observation noise. We used Q_v =  I.

Test_{x}.mat (x = 1,2,3,4) – Each file contains a single variable "radar_measurement" which are radar measurements corresponding to the path traced by 4 different vehicles undergoing the test. We assume Q_v = 0.1I. 


