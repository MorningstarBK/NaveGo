# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

NaveGo is a MATLAB / GNU Octave toolbox (no build system, no package manager, no test suite) for INS, loosely-coupled INS/GNSS (+magnetometer, +visual) integration, IMU/GNSS simulation, and Allan-variance sensor characterization. Math follows Groves (2013) and Titterton & Weston (2004). Licensed LGPL v3.

## Running

There is no build or lint step. Functionality is exercised by running the example scripts **from inside their own folder**, because they use relative `addpath ../../<dir>/` calls and load `.mat` files from the current directory:

```bash
cd examples/synthetic-data && octave --no-gui --eval navego_example_synth
```

Examples (these serve as the de facto regression tests):
- `examples/synthetic-data/navego_example_synth.m` — simulate two IMUs + GNSS, fuse, compare.
- `examples/real-data/navego_example_real_*.m` — Ekinox-D / MPU-6000 real data (Turin), incl. GNSS outage and INS-only.
- `examples/allan-variance/navego_example_allan.m` — Allan variance on STIM300 data.
- `examples/visual-data/navego_example_{canada,katwijk}_data.m` — VISUAL/INS/GNSS.
- `examples/sensor-read-example/` — reading raw sensor logs.

Example scripts are controlled by `'ON'`/`'OFF'` string flags at the top (e.g. `GNSS_DATA`, `IMU1_DATA`, `IMU1_INS`, `PLOT`). When `'OFF'`, the script loads the previously saved `.mat` result instead of regenerating it — set only the stage you want to rerun to `'ON'`.

MATLAB vs Octave differences are handled via `misc/is_octave.m`.

## Architecture

Each top-level folder is a flat set of functions put on the path by the examples:

- `ins/` — strapdown mechanization primitives: attitude (`att_update` → `qua_update`/`dcm_update`), `vel_update`, `pos_update`, Earth models (`gravity`, `radius`, `earth_rate`, `transport_rate`), magnetometer compass.
- `ins-gnss/` — the filters. `ins_gnss.m` is the core loosely-coupled EKF loop: propagates INS at IMU rate and, whenever a GNSS epoch matches the IMU time (within `gnss.eps`), builds `F_update` and runs `kalman` (`kf_prediction`/`kf_update`). 15-state error-state filter (attitude, velocity, position, gyro & accel dynamic biases); includes ZUPT and lever-arm handling. `ins_gnss_mag.m` adds magnetometer aiding; `ins.m` is pure INS.
- `ins-visual/` — `ins_visual*.m` variants using visual odometry as an additional aiding source.
- `simulation/` — generate synthetic IMU (`acc_gen`, `gyro_gen`, noise/bias models) and GNSS (`gnss_gen`, `gnss_outage`) from a reference trajectory (`ref.mat`); NaveGo has no trajectory generator.
- `conversions/` — frame/attitude conversions (Euler/DCM/quaternion, LLH/ECEF/NED); `imu_si_errors` converts datasheet error specs to SI units.
- `allan-variance/`, `performance-analysis/` (RMSE, statistics, profile comparison), `plot/`, `data-acquisition/` (parsers for Xsens, Microstrain, RTKLIB, etc.).

### Data conventions
Everything is passed as structs, documented in each function's header comment:
- `imu`: `t`, `fb` (m/s², body), `wb` (rad/s, body), error params (`arw`, `vrw`, `gb_sta`, `gb_dyn`, `gb_corr`, `ab_*`, ...), `freq`, `ini_align`, `ini_align_err`.
- `gnss`: `t`, `lat`/`lon` (radians), `h` (m), `vel` (NED), `std`/`stdm`/`stdv`, `larm` (IMU→antenna, body frame x-fwd/y-right/z-down), `freq`, `zupt_th`, `zupt_win`, `eps`.
- Navigation output `nav_e`/`ref`: `t`, `roll`/`pitch`/`yaw`, `vel`, `lat`/`lon`/`h`, plus KF histories (`xi`, `xp`, `z`, `v`, ...).
- Units are SI with angles in radians; navigation frame is NED. Attitude mode is `'quaternion'` (default) or `'dcm'`.

## Contributing

Branches: `master` = stable releases, `develop` = integration branch for new features. Contribution workflow is described in `doc/navego-contribution-workflow.md`. New source files carry the LGPL header comment used by existing files.

## Python port (`python/`)

`python/pynavego` is a standalone Python package (numpy only; matplotlib for plots) that re-implements INS mechanization, IMU/GNSS simulation and a 16-state error-state EKF, adding barometric-altitude aiding with a baro-bias state. Run `cd python && pytest -q` and `python examples/example_ins_baro.py`. The IMU generator (`sim/imu.py: true_imu_from_trajectory`) exactly inverts `ins.py: ins_step`, so if you change the mechanization discretization, change both (the perfect-IMU test enforces this).
