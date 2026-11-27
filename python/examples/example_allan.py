"""Allan deviation of a simulated static IMU: recovers ARW and bias instability."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pynavego.allan import allan_deviation  # noqa: E402
from pynavego.sim import generate_trajectory, simulate_imu  # noqa: E402

ref = generate_trajectory(duration=3600.0, still=1e9)      # stationary for 1 h
imu, true = simulate_imu(ref, "industrial", seed=3)
gyro = np.rad2deg(imu.wb[:, 0] - true.wb[:, 0]) * 3600.0  # deg/h
taus, adev = allan_deviation(gyro, imu.freq)
arw = adev[np.argmin(abs(taus - 1.0))] / 60.0             # sigma(1 s) -> deg/sqrt(h)
print(f"estimated ARW ~ {arw:.3f} deg/sqrt(h)  (spec {0.3})")
print(f"min Allan deviation ~ {adev.min():.2f} deg/h at tau = {taus[np.argmin(adev)]:.0f} s")
