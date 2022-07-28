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
from scipy.special import erf
from scipy.optimize import fmin, bisect
from scipy.interpolate import interp2d, RegularGridInterpolator
import matplotlib as mpl

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

# Minimum energy required to produce a photon (in this case we take the D band
# from https://arxiv.org/pdf/1904.09362.pdf)
E_MIN = 0.93 # eV

# Escape velocity
V_ESC = 600 # km/s
# Earth velocity
V_E = 240 # km/s
# Typical velocity
V_0 = 230 # km/s

# speed of light
SPEED_OF_LIGHT = 299792 # km/s

K = V_0**3*np.pi*(np.sqrt(np.pi)*erf(V_ESC/V_0) - 2*(V_ESC/V_0)*np.exp(-(V_ESC/V_0)**2)) # (km/s)^3

## import QEdark data
# From https://github.com/tientienyu/QEdark/blob/main/QEdark-python/QEdark_f2.ipynb
nq = 900
nE = 500

dQ = 0.02*ALPHA*M_E # GeV
dE = 0.1e-9 # GeV
qq = (np.arange(nq)+1)*dQ
ee = (np.arange(nE)+1)*dE

qbincenters = (qq[1:] + qq[:-1])/2
ebincenters = (ee[1:] + ee[:-1])/2
QQ, EE = np.meshgrid(qbincenters,ebincenters)

fcrys = {'Si': np.transpose(np.resize(np.loadtxt('Si_f2.txt',skiprows=1),(nE,nq))),
         'Ge': np.transpose(np.resize(np.loadtxt('Ge_f2.txt',skiprows=1),(nE,nq)))}

f_crystal = RegularGridInterpolator((qq,ee),fcrys['Ge'],bounds_error=False,fill_value=0)

def eta(q,e):
    """
    Inverse mean speed. See Appendix B.

    `q` is the momentum transfer (GeV), and `e` is the energy transferred to the electron (GeV).

    Returns the inverse mean speed in units of s/cm
    """
    v_min = (q/(2*M_X) + e/q)*SPEED_OF_LIGHT

    if v_min < 0:
        return 0
    elif v_min < V_ESC - V_E:
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
    def func(q,e):
        # FIXME: Assume here that F_DM(q) = 1 and f_crystal(q,e) = 1
        return (e/q)*eta(q,e)/q*f_crystal([q,e])

    def get_v_min(q,e):
        v_min = (q/(2*M_X) + e/q)*SPEED_OF_LIGHT
        return v_min

    # First, we try to find the minimum of vmin as a function of the momenum
    # transfer `q`. The reason is that the mean inverse speed `eta` function,
    # is only non-zero over a narrow range where 0 < v_min < V_ESC + V_E and
    # the quad integration later will fail if you give it points far away from
    # this (since it evaluates the function at the two edges and sees both are
    # zero. Therefore, we need to give it a set of points where the function
    # will be non-zero to make sure that it converges.
    xopt = fmin(lambda x: get_v_min(x,e), 1e-10, xtol=1e-6, ftol=1e-4, maxfun=100000)[0]

    # First we try to find the maximum point where v_min < V_ESC + V_E
    try:
        qmax = bisect(lambda x: get_v_min(x,e) - (V_ESC + V_E), xopt, 1, xtol=1e-20)
    except Exception:
        qmax = 1e2

    # Now we try to find the minimum point where v_vmin < V_ESC + V_E
    try:
        qmin = bisect(lambda x: get_v_min(x,e) - (V_ESC + V_E), 0, xopt, xtol=1e-20)
    except Exception:
        qmin = 1e-6

    # See Equation 3.13
    # The 1e-6 is to convert events/GeV -> events/keV
    return (1e-6*(RHO_X/M_X)*N_CELL*SIGMA_E*ALPHA*(M_E**2/MU**2)*quad(func,qmin,qmax,points=[xopt],epsrel=1e-10,epsabs=1e-20,args=(e))[0]/e)*(SPEED_OF_LIGHT*1e5)**2

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import ROOT
    import argparse

    parser = argparse.ArgumentParser("plot the number of expected dark matter interactions as a function of energy")
    parser.add_argument("-o","--output",type=str,help="output filename",default=None)
    args = parser.parse_args()

    e = np.logspace(np.log(E_MIN*1e-3),1,10)
    rate = np.array(list(map(get_differential_rate,e)))

    plt.figure()
    f_crystal_test = f_crystal(list(zip(QQ.flatten(),EE.flatten())))
    print(f_crystal_test[f_crystal_test != 0])
    plt.hist2d(QQ.flatten()*1e6,EE.flatten()*1e9,bins=[qq*1e6,ee*1e9],weights=f_crystal(list(zip(QQ.flatten(),EE.flatten()))),norm=mpl.colors.LogNorm(vmin=1e-3,vmax=10))
    plt.xlabel("q (keV)")
    plt.ylabel("E (eV)")
    plt.title("Germanium Form Factor")
    plt.colorbar()

    if args.output is not None:
        f = ROOT.TFile(args.output,"recreate")
        h = ROOT.TH1D("h","Rate of dark matter interactions",len(e)-1,e)
        for i in range(1,len(e)):
            x = h.GetBinCenter(i)
            h.SetBinContent(i,np.interp(x,e,rate))
        h.Write()
        f.Close()

    total_rate = np.trapz(rate*24*60*60,e)

    print("total rate is %.2e events/kg/day" % total_rate)

    plt.figure()
    plt.plot(e,rate*24*60*60)
    plt.gca().set_xscale("log")
    plt.xlabel("Energy Transferred (keV)")
    plt.ylabel("Rate (Events/kg/keV/day)")
    plt.show()
