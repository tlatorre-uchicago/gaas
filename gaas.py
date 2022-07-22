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

# From https://arxiv.org/pdf/1509.01598.pdf Equation 3.13

# Dark matter energy density
RHO_X = 0.4 # GeV/cm^3
# N_CELL = M_Target/M_cell is the number of unit cells in the target crystal
# FIXME: This is for Germanium, need to determine it for GaAs
N_CELL = 135.33 # GeV
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
