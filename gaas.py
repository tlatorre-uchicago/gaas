import numpy as np
"""
This is a script to calculate the differential event rate (in terms of the
scattered electron's energy) for dark matter interacting with the electrons in
a GaAs semiconductor.

The calculation comes from the paper "Direct Detection of sub-GeV Dark Matter
With Semiconductor Targets" by Essig et. al. (see
https://arxiv.org/pdf/1509.01598.pdf).

Date: July 22, 2022
Author: Anthony LaTorre
"""
import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
from scipy.special import erf

# From https://arxiv.org/pdf/1509.01598.pdf Equation 3.13

# Dark matter energy density
RHO_X = 0.4 # GeV/cm^3
# N_CELL = M_Target/M_cell is the number of unit cells in the target crystal
# FIXME: This is for Germanium, need to determine it for GaAs
M_CELL = 135.33 # GeV
M_TARGET = 1 # 1 kg
N_CELL = M_TARGET*5.6e26/M_CELL
# Mass of dark matter particle
M_X = 0.100 # GeV
# Reference cross section (the current limit from Xenon10)
SIGMA_E = 3.6e-37 # cm^2
# Fine structure constant
ALPHA = 1/137.035999084
# Mass of electron
M_E = 0.511e-3 # GeV
# Reduced mass of electron dark matter system
MU = M_E*M_X/(M_E + M_X)
# v_min = q/(2*M_X) + delta_E/q

# Escape velocity
V_ESC = 600 # km/s
# Earth velocity
V_E = 240 # km/s
# Typical velocity
V_0 = 230 # km/s

K = V_0**3*np.pi*(np.sqrt(np.pi)*erf(V_ESC/V_0) - 2*(V_ESC/V_0)*np.exp(-(V_ESC/V_0)**2)) # (km/s)^3

def eta(q,e):
    """
    Inverse mean speed. See Appendix B.

    `q` is the momentum transfer (GeV), and `e` is the energy transferred to the electron (GeV).

    Returns the inverse mean speed in units of s/cm
    """
    v_min = q/(2*M_X) + e/q

    if v_min < V_ESC - V_E:
        # The 1e-5 is to convert s/km -> s/cm
        return 1e-5*V_0**2*np.pi/(2*V_E*K)*(-4*np.exp(-V_ESC**2/V_0**2)*V_E + np.sqrt(np.pi)*V_0*(erf((v_min+V_E)/V_0) - erf((v_min-V_E)/V_0)))
    elif V_ESC - V_E < v_min < V_ESC + V_E:
        return 1e-5*V_0**2*np.pi/(2*V_E*K)*(-2*np.exp(-V_ESC**2/(V_0**2))*(V_ESC - v_min + V_E) + np.sqrt(np.pi)*V_0*(erf(V_ESC/V_0) - erf((v_min-V_E)/V_0)))
    else:
        return 0

def get_differential_rate(e):
    """
    Returns the differential event rate (events/keV/kg/s) for a dark matter particle
    to scatter an electron with energy `e` (keV).
    """
    # keV -> GeV
    e /= 1e6
    def func(ln_q,e):
        # FIXME: Assume here that F_DM(q) = 1 and f_crystal(q,e) = 1
        q = np.exp(ln_q)
        return (e/q)*eta(q,e)

    # See Equation 3.13
    # The 1e-6 is to convert events/GeV -> events/keV
    ln_q_min = np.log(1e-10)
    ln_q_max = np.log(100)
    return 1e-6*(RHO_X/M_X)*N_CELL*SIGMA_E*ALPHA*(M_E**2/MU**2)*quad(func,ln_q_min,ln_q_max,args=(e))[0]/e

e = np.linspace(0.001,100,1000)
rate = np.array(list(map(get_differential_rate,e)))

plt.plot(e,rate*24*60*60)
plt.xlabel("Energy Transferred (keV)")
plt.ylabel("Rate (Events/kg/keV/day)")
plt.show()
