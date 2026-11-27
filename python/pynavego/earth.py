"""WGS-84 Earth model (cf. NaveGo ins/radius.m, gravity.m, earth_rate.m, transport_rate.m)."""
import numpy as np

A = 6378137.0                # semi-major axis (m)
E2 = 0.00669437999014        # first eccentricity squared
OMEGA_IE = 7.292115e-5       # Earth rotation rate (rad/s)


def radius(lat):
    """Meridian (RM) and normal (RN) radii of curvature (m)."""
    den = 1.0 - E2 * np.sin(lat) ** 2
    RN = A / np.sqrt(den)
    RM = A * (1.0 - E2) / den ** 1.5
    return RM, RN


def gravity_magnitude(lat, h):
    """Somigliana normal gravity with a free-air (2h/R) altitude correction (m/s^2)."""
    s2 = np.sin(lat) ** 2
    g0 = 9.7803253359 * (1.0 + 0.001931853 * s2) / np.sqrt(1.0 - E2 * s2)
    return g0 * (1.0 - 2.0 * h / A)


def gravity(lat, h):
    """Gravity vector in NED (down positive)."""
    return np.array([0.0, 0.0, gravity_magnitude(lat, h)])


def earth_rate(lat):
    """Earth rotation rate expressed in NED (rad/s)."""
    return OMEGA_IE * np.array([np.cos(lat), 0.0, -np.sin(lat)])


def transport_rate(lat, vel, h, RM=None, RN=None):
    """Rotation of NED w.r.t. Earth due to motion (rad/s)."""
    if RM is None:
        RM, RN = radius(lat)
    return np.array([
        vel[1] / (RN + h),
        -vel[0] / (RM + h),
        -vel[1] * np.tan(lat) / (RN + h),
    ])
