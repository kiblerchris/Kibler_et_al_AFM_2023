import numpy as np
from bigleaf import esat_slope, psychrometric_constant, kPa2Pa
from air_density_functions import moist_air_density
from scipy import constants
from warnings import warn

#Calculate TL-TA (Equation 6) for the sensitivity analysis
def calc_tl_ta(rn, rah, ef, rho, cp):
    left = (rn * rah) / (rho * cp)
    right = (1 - ef)
    out = left * right
    return out

#Calculate TC-TA (Equation 6) for the eddy covariance data
def calc_tc_ta(rn, le, g, rah, rho, cp):
    ef = le/(rn-g)
    left = ((rn - g) * rah) / (rho * cp)
    right = (1 - ef)
    tc_ta = left * right
    return tc_ta

#Calculate the coefficients of the quartic formula based on Equation 22 (Tc,nc)
def calc_quartic_coefs(SW_IN, SW_OUT, LW_IN, G, TA, rha, emiss, rho, cp, sigma):
    a = (-1 * emiss * sigma * rha)/(rho * cp)
    d = -1
    rabs = SW_IN - SW_OUT + (LW_IN * emiss) - G
    e = TA + ((rabs * rha)/(rho * cp))
    return a,d,e

#Solve the quartic formula by finding the roots and selecting the reasonable value
def calc_quartic_tlnc(x):
    
    #Calculate polynomial coefficients
    coef_a, coef_d, coef_e = calc_quartic_coefs(SW_IN = x.SW_IN, 
                                               SW_OUT = x.SW_OUT, 
                                               LW_IN = x.LW_IN, 
                                               G = x.G,
                                               TA = x.TA_1_2_1 + 273.15, 
                                               rha = x.opt_rha, 
                                               emiss = 0.98,
                                               rho = x.rho,
                                               cp = x.cp,
                                               sigma = constants.sigma)
    
    #Calculate roots of the polynomial
    roots = np.polynomial.Polynomial([coef_e, coef_d, 0, 0, coef_a]).roots()

    #Select real roots between 0-100 C and return value in C
    real_roots = roots[np.isreal(roots) & (roots > 273.15) & (roots < 273.15 + 100)]
    if len(real_roots) == 1:
        return real_roots[0].real - 273.15
    else:
        warn("Multiple roots returned: " + str(tuple(real_roots)))
        return np.nan