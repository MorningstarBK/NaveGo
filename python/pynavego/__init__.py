"""pynavego: inertial sensor simulation and INS / GNSS / barometric-altitude integration.

Python port of ideas from the NaveGo MATLAB toolbox (LGPL v3).
Conventions: SI units, angles in radians, NED navigation frame,
C_nb = body-to-navigation DCM.
"""
from .ins import NavState, ins_step
from .fusion import FilterConfig, run_fusion

__all__ = ["NavState", "ins_step", "FilterConfig", "run_fusion"]
__version__ = "0.1.0"
