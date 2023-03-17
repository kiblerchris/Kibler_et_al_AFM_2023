import pandas as pd
import numpy as np
import warnings
from scipy.optimize import brute
from air_density_functions import moist_air_density, cp_moist

#Define constants
von_karman = 0.41 #Source: Young et al. (2021)
kinematic_viscosity = 1.461e-5 #Source: Verhoef et al. (1997)
molecular_thermal_diffusivity = 2.06e-5 #Source: Verhoef et al. (1997)
prandtl = kinematic_viscosity/molecular_thermal_diffusivity #Source: Verhoef et al. (1997)

#Brute force optimization that can handle nan values
def nanbrute(fun, ranges, args):
    
    """
    Brute force optimization that can handle NaN values
    
    Dependencies
    ----------
    import numpy as np
    from scipy.optimize import brute
    import warnings

    Parameters
    ----------
    fun : function object
        Function to optimize
    ranges : range
        Range of values to test
    args : variable
        Other arguments to be passed to the function

    Returns
    -------
    min_val : float
        Optimum value based on brute force optimization
    """
    
    brute_out = brute(fun, ranges, args, full_output = True)
    in_vals = brute_out[2]
    diff_vals = brute_out[3]
    
    if sum(np.isfinite(diff_vals)) > 0: #test for any finite values
        min_val = in_vals[diff_vals == np.nanmin(diff_vals)] #select minimum value
        if len(min_val) > 1: 
            warnings.warn('Multiple minimum values returned. Output is mean of returned values:' + str(min_val), Warning)
            return np.mean(min_val)
        elif len(min_val) == 1:
            return min_val[0]
        elif len(min_val) == 0: #shouldn't be possible
            print('No values returned')
            return np.nan
    else:
        return np.nan

#Calculate z0m following Young et al. (2021, Equation 17)
def calc_z0m(u, ustar, z, d, k):
    
    """
    Calculate roughness length for momentum.
    Source: Adam Young et al., Seasonality in aerodynamic resistance across a range of North American ecosystems (2021), Equation 17

    Parameters
    ----------
    u : float
        Wind speed [m/s]
    ustar : float
        Friction velocity [m/s]
    z : float
        Measurement height [m]
    d : float
        Zero-plane displacement height [m]
    k : float
        Von Karman Constant [0.41, unitless]

    Returns
    -------
    out : float
        Roughness length for momentum [m]
    """
    
    top = z - d
    bottom = np.exp((k * u)/ustar)
    out = top/bottom
    return out

#Calculate Reynold's number following Young et al. (2021, Equation 10)
def calc_reynolds(ustar, z0m, v):
    
    """
    Calculate Reynold's number.
    Source: Adam Young et al., Seasonality in aerodynamic resistance across a range of North American ecosystems (2021), Equation 10

    Parameters
    ----------
    ustar : float
        Friction velocity [m/s]
    z0m : float
        Roughness length for momentum [m]
    v : float
        Kinematic viscocity [m2/s]

    Returns
    -------
    out : float
        Reynold's number [unitless]
    """
    
    out = (ustar * z0m)/v
    return out

#Calculate aerodynamic resistance from kB^-1 following Young et al. (2021, Equations 2-7)
def calc_rha_from_kb1(u, ustar, k, kb1):
    
    """
    Calculate aerodynamic resistance to heat transfer based on an estimate of kB-1.
    Source: Adam Young et al., Seasonality in aerodynamic resistance across a range of North American ecosystems (2021), Equations 2-7

    Parameters
    ----------
    u : float
        Wind speed [m/s]
    ustar : float
        Friction velocity [m/s]
    k : float
        Von Karman constant [0.41, unitless]
    kb1 : float
        kB-1 parameter, which is ln(z0m/zoh) [unitless]

    Returns
    -------
    out : float
        Aerodynamic resistance to heat transfer [s/m]
    """
    
    ram = u/(ustar ** 2)
    rbh = (1/(k * ustar)) * kb1
    out = ram + rbh
    return out

#Calculate d using a vertical gradient of wind measurements following Young et al. (2021), Equation 16
#Based on Monteith and Unsworth (2008), Equation 16.31
def gradient_d(d, u1, u2, z1, z2, ustar):
    
    #Calculate both sides of Equation 16
    left = (von_karman * (u1 - u2))/ustar
    right = np.log((z1 - d)/(z2 - d)) if ((z2 - d) != 0 and ((z1 - d)/(z2 - d)) > 0) else np.nan
    
    #Calculate the difference between them to minimize later on
    diff = abs(left - right)
    
    #Make sure output isn't all nan values
    if sum(np.isfinite(diff)) > 0: #test for any finite values
        mean_diff = np.nanmean(diff) #calculate mean difference across all observations
        return mean_diff
    else:
        return np.nan

#Calculate aerodynamic resistance from eddy covariance data
def calc_rha_from_temp(pa, ta, rh, tc_ta, h):
    
    """
    Calculate aerodynamic resistance to heat transfer based on temperature and sensible heat flux measurements.

    Parameters
    ----------
    pa : float
        Air pressure [kPa]
    ta : float
        Air temperature [C]
    rh : float
        Relative humidity [unitless number scaled from 0-100]
    tc_ta : float
        The difference between surface temperature and air temperature [C]
    h : float
        Sensible heat flux [W/m2]

    Returns
    -------
    out : float
        Aerodynamic resistance to heat transfer [s/m]
    """
    
    cp = cp_moist(RH = rh, PA = pa, TA = ta)
    rho = moist_air_density(TA = ta, PA = pa, RH = rh)
    out = (rho * cp * tc_ta)/h
    return out

#Predict rH using selected method. An argument dictionary selects the appropriate columns from each data set.
def predict_rha(df, arg_dict, method, d_range = slice(0, 10, 0.1)):
    
    df_rha = df.copy()
    
    if method == 'thom_1972':
        kb1 = 1.35 * von_karman * ((100 * df_rha[arg_dict['USTAR_col']])**(1/3))
    elif method == 'log1':
        kb1 = np.log(1)
    elif method == 'log10':
        kb1 = np.log(10)
    elif method == 'log100':
        kb1 = np.log(100)
    else:
        #Optimize value of d for each month
        monthly_d_pred = (df_rha
                  .groupby([arg_dict['year_col'], arg_dict['month_col']])
                  .apply(func = lambda x: nanbrute(gradient_d, ranges = (d_range,), args = (x[arg_dict['WS_1_col']], x[arg_dict['WS_2_col']], arg_dict['z1'], arg_dict['z2'], x[arg_dict['USTAR_col']],)))
                  .to_frame(name = 'd_pred'))
    
        #Add monthly d values to data frame
        df_rha = df_rha.join(monthly_d_pred, how = 'left', on = [arg_dict['year_col'], arg_dict['month_col']])

        z0m = calc_z0m(u = df_rha[arg_dict['plant_WS_col']], ustar = df_rha[arg_dict['USTAR_col']], z = arg_dict['plant_z'], d = df_rha['d_pred'], k = von_karman)
        reynolds = calc_reynolds(ustar = df_rha[arg_dict['USTAR_col']], z0m = z0m, v = kinematic_viscosity)
        
        if method == "brutsaert_1982":    
            kb1 = 2.46 * (reynolds ** 0.25) - 2
        elif method == "sheppard_1958_v1":
            kb1 = np.log((von_karman * df_rha[arg_dict['USTAR_col']] * z0m)/(molecular_thermal_diffusivity))
        elif method == 'sheppard_1958_v2':
            kb1 = np.log(prandtl * reynolds) #As described in Hong et al. (2012)
        elif method == 'owen_thomson_1963':
            kb1 = von_karman * 0.52 * (8 * reynolds)**0.45 * prandtl**0.8
        elif method == 'zeng_dickinson_1998':
            kb1 = 0.13 * (reynolds**0.45)
        elif method == 'zilitinkevich_1995':
            kb1 = von_karman * 0.1 * np.sqrt(reynolds)
        elif method == 'kanda_2007':
            kb1 = (1.29 * (reynolds**0.25)) - np.log(7.4)
        elif method == 'kondo_1975':
            kb1 = (von_karman * 3 * (prandtl**(2/3))) + (np.log((reynolds * (prandtl**(1/3)))/3))
        else: 
            raise NameError('Method not recognized')
        
    rha_pred = calc_rha_from_kb1(u = df_rha[arg_dict['plant_WS_col']], ustar = df_rha[arg_dict['USTAR_col']], k = von_karman, kb1 = kb1)
    
    return rha_pred

#Calculate aerodynamic resistance using kB-1 formula from Thom (1972) as described in Verhoef et al. (1997)
def rha_from_thom_1972(USTAR_col, U_col):
    
    """
    Calculate aerodynamic resistance to heat transfer using kB-1 method described by Thom (1972).
    Source: A.S. Thom, Momentum, mass, and heat exchange in vegetation (1972)
    See also: A. Verhoef et al., Some practical notes on the parameter kB-1 for sparse vegetation (1997), Equation 11

    Parameters
    ----------
    U_col : float
        Wind speed [m/s]
    USTAR_col : float
        Friction velocity [m/s]
    
    Constants
    ----------
    von_karman : float
        Von Karman constant [0.41, unitless]

    Returns
    -------
    opt_rha : float
        Aerodynamic resistance to heat transfer [s/m]
    """
    
    von_karman = 0.41
    kb1_from_thom = 1.35 * von_karman * ((100 * USTAR_col)**(1/3))
    opt_rha = calc_rha_from_kb1(u = U_col, ustar = USTAR_col, k = von_karman, kb1 = kb1_from_thom)
    return opt_rha

#Example arg_dict:

# arg_dict = {'WS_1_col': 'WS_1_1_1', 
#             'WS_2_col': 'WS_1_2_1',
#             'plant_WS_col': 'WS_1_2_1',
#             'z1': 14,
#             'z2': 8, 
#             'plant_z': 8,
#             'USTAR_col': 'USTAR', 
#             'H_col': 'H', 
#             'plant_TA_col': 'TA_1_2_1', 
#             'TC_TA_col': 'IR_tc_ta',
#             'PA_col': 'PA',
#             'year_col': 'year', 
#             'month_col': 'month'}