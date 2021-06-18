# VEHICLE DETECTION AND TRACKING

Vehicle detection and tracking for automated driving test. Vehicle detection at the entry and exit of the course is designed using constant detection rate (CDR) and constant false-alarm rate (CFAR) Neymann-Pearson detectors. Vehicle tracking estimation is done using Kalman filter. The repository contains Python3 scripts for:

- CDR Neymann-Pearson detector,
- CFAR Neymann-Pearson detector,
for detection, and,
- Kalman Filter,
for vehicle tracking.

## Documentation

This work is as part of E1 244 Detection and Estimation Theory, Spring 2021 course at the Indian Institute of Science. `docs/instructions.pdf` contains the necessary instructions for the assignment. `docs/solutions.pdf` contains the details of theory and algorithms. `docs/Vehicle\ Tracking.pdf` contains presentation slides. `docs/Vehicle\ Tracking.m4v` contains a video explaining the results and observations.

## Installation

Clone this repository and install the requirments using
```shell
git clone https://github.com/kamath-abhijith/Vehicle_Tracking
conda create --name <env> --file requirements.txt
```

## Run

- Run `run_CDR.sh` to generate Figure 2 and 3,
- Run `run_CFAR.sh` to generate Figure 4 and 5,
- Run `run_KFVel.sh` to generate Figure 6,
- Run `run_KFPos.sh` to generate Figure 7 and 8,
- Run `run_KFmis.sh` to generate Figure 9,
- Run `run_test.sh` to generate Figure 10.
- Use `utils.make_trace_video()` to generate the movies.