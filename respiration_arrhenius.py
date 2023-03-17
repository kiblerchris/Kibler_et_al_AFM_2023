import pandas as pd
import numpy as np

def rleaf_dark(leaf_temp, Rtref, ref_temp):
    
    """
    Calculate leaf dark respiration as a function of leaf temperature
    Source: Heskel et al. (2016): Convergence in the temperature response of leaf respiration across biomes and plant functional types, Eq. 3

    Parameters
    ----------
    leaf_temp : float
        Leaf temperature [C]
    Rtref : float
        Respiration at reference leaf temperature [umol CO2/(m2 s1)]
    ref_temp : float
        Reference leaf temperature [C]

    Returns
    -------
    leaf_resp_dark : float
        Leaf dark respiration [umol CO2/(m2 s1)]
    """
    
    leaf_resp_dark = Rtref * np.exp((0.1012 * (leaf_temp - ref_temp)) - (0.0005 * (leaf_temp**2 - ref_temp**2)))
    return leaf_resp_dark

#Adjust leaf dark respiration estimate to account for light inhibition
def rleaf_light(rleaf_dark, leaf_temp):
    
    """
    Calculate leaf light respiration from leaf dark respiration as a function of leaf temperature
    Source: Mathias and Trugman (2022): Climate change impacts plant carbon balance, increasing mean future carbon use efficiency but decreasing total forest extent at dry range edges, Eq. 1

    Parameters
    ----------
    rleaf_dark : float
        Leaf dark respiration [umol CO2/(m2 s1)]
    leaf_temp : float
        Leaf temperature [C]

    Returns
    -------
    leaf_resp_light : float
        Leaf light respiration [umol CO2/(m2 s1)]
    """
    
    leaf_resp_light = rleaf_dark * (0.0039 * leaf_temp + 0.6219) 
    return leaf_resp_light

#Calculate leaf light respiration
def lightResp(leaf_temp, Rtref, ref_temp):
    
    """
    Calculate leaf light respiration from leaf temperature

    Parameters
    ----------
    leaf_temp : float
        Leaf temperature [C]
    Rtref : float
        Respiration at reference leaf temperature [umol CO2/(m2 s1)]
    Ea : float
        Activation energy
    ref_temp : float
        Reference leaf temperature [C]

    Returns
    -------
    leaf_resp_light : float
        Leaf light respiration [umol CO2/(m2 s1)]
    """
    
    leaf_resp_dark = rleaf_dark(leaf_temp = leaf_temp, Rtref = Rtref, ref_temp = ref_temp)
    leaf_resp_light = rleaf_light(rleaf_dark = leaf_resp_dark, leaf_temp = leaf_temp)
    return leaf_resp_light

#Calculate the percent change in respiration due to evaporative cooling
def carbonDifference(df, tc_col, tcnc_col, Rtref, ref_temp = 25):
    vals = df.copy()
    
    #Calculate respiration for modeled TL
    vals['resp_tc'] = lightResp(leaf_temp = vals[tc_col], Rtref = Rtref, ref_temp = ref_temp)
    
    #Calculate respiration for modeled TL,nc
    vals['resp_tcnc'] = lightResp(leaf_temp = vals[tcnc_col], Rtref = Rtref, ref_temp = ref_temp)
    
    #Caclulate difference between respiration predictions
    vals['resp_diff'] = vals.resp_tcnc - vals.resp_tc

    #Combine outputs into dataframe
    vals['resp_perc_change'] = ((vals.resp_tc - vals.resp_tcnc)/vals.resp_tcnc) * -100

    #Calculate monthly means
    resp_monthly_mean = vals[['resp_tc', 'resp_tcnc', 'resp_diff', 'resp_perc_change', 'month', 'year']].groupby(['month', 'year']).mean()
    
    return vals, resp_monthly_mean