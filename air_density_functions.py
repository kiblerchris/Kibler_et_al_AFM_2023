import numpy as np

Kelvin = 273.15, 
kPa2Pa = 1000, 

def calc_ea(airtemp, RH):
    
    """
    Calculate partial vapor pressure using Tetens' formula.
    Source: Richard Allen et al., FAO Paper 56 (1998), Equations 11 and 19
    See also: Roland Stull, 2017: Practical Meteorology: An Algebra-based Survey of Atmospheric Science (Equation 4.2)

    Parameters
    ----------
    airtemp : float
        Air temperature [C]
    RH : float
        Relative humidity [unitless number scaled from 0-100]

    Returns
    -------
    ea : float
        Partial vapor pressure [kPa]
        
    NOTE: Validated using NOAA calculator (https://www.weather.gov/epz/wxcalc_vaporpressure), and nearly identical to Stull (2017) Equation 4.2.
    """
    
    es = 0.61078 * np.exp((17.27 * airtemp)/(airtemp + 237.3))
    ea = (RH / 100) * es #Bonan, 2016 (Equation 3.32)
    return ea

def moist_air_density(TA, PA, RH):
    
    """
    Calculate air density of moist air.
    Source: Gordon Bonan, Ecological Climatology (2016), Equation 3.29

    Parameters
    ----------
    PA : float
        Air pressure [kPa]
    TA : float
        Air temperature [C]
    RH : float
        Relative humidity [unitless number scaled from 0-100]
        
    Constants
    ----------
    M : float
        Molecular mass of dry air [kg/mol]
    R : float
        Universal gas constant [J/(mol K)]

    Returns
    -------
    mad : float
        Mean air density [kg/m3]

    NOTE: P/RT is in units of [mol/m3] and M is [kg/mol] so MAD is [kg/m3].
    Validated using STP example from Bonan (2016) page 51. STP is defined on page 45.
    Other option: (1000 * pa)/(287.058 * ta) * (1 - ((1 - 0.622) * ea)/(1000 * pa)), nearly identical with slight coefficient difference (0.37%) on the left side.
    """
    
    M = 28.97 / 1000 #Convert from [g mol-1] to [kg mol-1] (see 3.12)
    R = 8.314 #[J K-1 mol-1]
    p = PA * kPa2Pa[0]
    ea = calc_ea(airtemp = TA, RH = RH) * kPa2Pa[0]
    mad = (p/(R*(TA + Kelvin[0])))*M*(1 - (0.378 * (ea/p))) #P/RT is [mol/m3] (see 3.11) and M is [kg/mol] so MAD is [kg/m3]
    return mad

def cp_moist(RH, PA, TA, opt = 'v2', cpd = 1005, cpv = 1875):
    
    """
    Calculate specific heat at constant pressure of moist air.
    v1 Source: https://www.caee.utexas.edu/prof/Novoselac/classes/ARE383/Handouts/Chapter%207_Thermodynamic%20Properties%20of%20Moist%20Air.pdf
    v2 Source: Roland Stull, 2017: Practical Meteorology: An Algebra-based Survey of Atmospheric Science (Equation 3I.2)
    v3 Source: Roland Stull, 2017: Practical Meteorology: An Algebra-based Survey of Atmospheric Science (Equation 3I.3)

    Parameters
    ----------
    PA : float
        Air pressure [kPa]
    TA : float
        Air temperature [C]
    RH : float
        Relative humidity [unitless number scaled from 0-100]
        
    Constants
    ----------
    cpd : float
        Specific heat for dry air at constant pressure at 27 °C (Source: Stull, 2017, Appendix B.4)
    cpv : float
        Specific heat for water vapor at constant pressure at 15 °C (Source: Stull, 2017, Appendix B.4)

    Returns
    -------
    cp : float
        Specific heat at constant pressure of moist air [J/(kg K)]
        
    NOTE: Correlation between three methods exceeds 0.999. v1 and v3 are mathematically identical. v2 seems to be more robust and exhibited a mean difference of 0.65%.
    A representative value from Bonan (2016) page 46 is 1010 J/(kg K).
    """
    
    ea = calc_ea(airtemp = TA, RH = RH)
    w = 0.622 * (ea/(PA - ea)) #Mixing ratio from Stull, 2017 (Equation 4.4)
    if opt == 'v1':
        cp = cpd + (cpv * w) #https://www.caee.utexas.edu/prof/Novoselac/classes/ARE383/Handouts/Chapter%207_Thermodynamic%20Properties%20of%20Moist%20Air.pdf
    elif opt == 'v2':
        cp = (1 - w) * cpd * (1 + ((cpv/cpd) * w)) #Stull, 2017 (Equation 3I.2)
    elif opt == 'v3':
        cp = cpd * (1 + (1.84 * w)) #Stull, 2017 (Equation 3I.3)
    return cp