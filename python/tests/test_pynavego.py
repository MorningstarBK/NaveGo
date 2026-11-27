import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pynavego.attitude import dcm2euler, dcm2rotvec, euler2dcm, rotvec2dcm  # noqa: E402
from pynavego.earth import gravity_magnitude  # noqa: E402
from pynavego.fusion import initial_state, run_fusion  # noqa: E402
from pynavego.ins import ins_step  # noqa: E402
from pynavego.metrics import nav_errors, rmse  # noqa: E402
from pynavego.sim import (generate_trajectory, simulate_baro, simulate_gnss,  # noqa: E402
                          simulate_imu, true_imu_from_trajectory)


@pytest.fixture(scope="module")
def ref():
    return generate_trajectory(duration=120.0)


def test_gravity():
    assert gravity_magnitude(0.0, 0.0) == pytest.approx(9.7803, abs=1e-3)
    assert gravity_magnitude(np.pi / 2, 0.0) == pytest.approx(9.8322, abs=1e-3)


def test_attitude_roundtrip():
    e = np.array([0.3, -0.2, 2.5])
    assert np.allclose(dcm2euler(euler2dcm(*e)), e)
    r = np.array([0.1, -0.4, 0.25])
    assert np.allclose(dcm2rotvec(rotvec2dcm(r)), r)


def test_perfect_imu_reproduces_trajectory(ref):
    imu = true_imu_from_trajectory(ref)
    s = initial_state(ref, att_err_deg=(0, 0, 0))
    for k in range(len(ref.t) - 1):
        s, _ = ins_step(s, imu.wb[k], imu.fb[k], ref.t[k + 1] - ref.t[k])
    assert abs(s.h - ref.h[-1]) < 1e-6
    assert abs(s.lat - ref.lat[-1]) * 6.4e6 < 1e-6
    assert np.allclose(s.C_nb, ref.C_nb[-1], atol=1e-9)


def test_baro_bounds_altitude(ref):
    imu, _ = simulate_imu(ref, "industrial", seed=0)
    baro = simulate_baro(ref, seed=1)
    init = initial_state(ref)
    e_ins = nav_errors(run_fusion(imu, init), ref)
    e_baro = nav_errors(run_fusion(imu, init, baro=baro), ref)
    # Baro-aided altitude stays within the (unobservable) baro bias + a few metres.
    assert rmse(e_baro["h"]) < 8.0
    assert rmse(e_baro["h"]) < rmse(e_ins["h"]) / 5


def test_gnss_estimates_baro_bias(ref):
    imu, _ = simulate_imu(ref, "industrial", seed=0)
    baro = simulate_baro(ref, seed=1)
    gnss = simulate_gnss(ref, seed=2)
    res = run_fusion(imu, initial_state(ref), baro=baro, gnss=gnss)
    assert abs(res.baro_bias[-1] - baro.data["bias_true"][-1]) < 1.5
    assert rmse(nav_errors(res, ref)["h"]) < 2.0


@pytest.mark.parametrize("axis", ["x", "y", "z"])
def test_encoder_recovers_vehicle_attitude(ref, axis):
    from pynavego.attitude import wrap_angle
    from pynavego.sim import rotate_trajectory, rotation_angle, simulate_encoder, vehicle_attitude
    theta = rotation_angle(ref.t, 36.0)
    sref = rotate_trajectory(ref, theta, axis)
    imu = true_imu_from_trajectory(sref)
    i = "xyz".index(axis)   # that gyro sees the turntable rate on top of vehicle motion
    assert np.mean(imu.wb[:-1, i]) == pytest.approx(np.deg2rad(36.0), abs=0.01)
    s = initial_state(sref, att_err_deg=(0, 0, 0))
    for k in range(len(ref.t) - 1):
        s, _ = ins_step(s, imu.wb[k], imu.fb[k], 0.01)
    enc = simulate_encoder(ref.t, theta)
    att = vehicle_attitude(np.array([dcm2euler(s.C_nb)]), enc.theta[-1:], axis)
    assert np.all(np.abs(wrap_angle(att[0] - ref.euler[-1])) < np.deg2rad(0.01))


def test_mag_bounds_heading(ref):
    from pynavego.attitude import wrap_angle
    from pynavego.sim import simulate_mag
    imu, _ = simulate_imu(ref, "consumer", seed=0)
    gnss = simulate_gnss(ref, outages=[(30.0, 1e9)], seed=2)
    init = initial_state(ref)
    yaw = lambda r: rmse(nav_errors(r, ref)["att"][3000:, 2])
    y_ins = yaw(run_fusion(imu, init, gnss=gnss))
    y_mag = yaw(run_fusion(imu, init, gnss=gnss, mag=simulate_mag(ref)))
    assert y_mag < np.deg2rad(1.0) and y_mag < y_ins


def test_los_range_reduces_drift(ref):
    from pynavego.sim import simulate_los
    imu, _ = simulate_imu(ref, "industrial", seed=0)
    gnss = simulate_gnss(ref, outages=[(40.0, 1e9)], seed=2)
    init = initial_state(ref)
    los = simulate_los(ref)
    e0 = nav_errors(run_fusion(imu, init, gnss=gnss), ref)
    e1 = nav_errors(run_fusion(imu, init, gnss=gnss, los=los), ref)
    assert rmse(e1["horizontal"]) < rmse(e0["horizontal"])
