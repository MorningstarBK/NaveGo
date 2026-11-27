"""Navigation error metrics (cf. NaveGo performance-analysis/)."""
import numpy as np

from .attitude import wrap_angle
from .earth import radius


def nav_errors(res, ref):
    """Errors of a FusionResult against a Trajectory sampled at the same times."""
    RM, RN = radius(ref.lat)
    dN = (res.lat - ref.lat) * (RM + ref.h)
    dE = (res.lon - ref.lon) * (RN + ref.h) * np.cos(ref.lat)
    return {
        "north": dN, "east": dE, "horizontal": np.hypot(dN, dE),
        "h": res.h - ref.h,
        "vel": res.vel - ref.vel,
        "att": wrap_angle(res.euler - ref.euler),
    }


def rmse(x, mask=None):
    x = np.asarray(x)
    if mask is not None:
        x = x[mask]
    return float(np.sqrt(np.mean(np.square(x))))
