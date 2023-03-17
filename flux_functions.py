import pandas as pd
import datetime as dt
import numpy as np
from dateutil.relativedelta import relativedelta
from bigleaf import RH_to_VPD
from air_density_functions import cp_moist, moist_air_density

#Add time index to data frame
def setTimeIndex(dataframe, timestart_col, timeend_col, timezone):
    df = dataframe.copy()
    
    #Convert start and end times to date/time objects
    df = df.assign(time_start = pd.to_datetime(df[timestart_col], format = "%Y%m%d%H%M").dt.tz_localize(timezone),
                   time_end = pd.to_datetime(df[timeend_col], format = "%Y%m%d%H%M").dt.tz_localize(timezone))
    
    #Assign the start time as the index
    df.index = pd.DatetimeIndex(df['time_start'])
    return df

#Remove days with any precipitation on that day or the day before
def noDailyPrecip(dataframe, p_col):
    df = dataframe.copy()
    df['date'] = df.index.date
    
    #Calculate maximum precipitation for date
    max_p = (df.groupby(df.index.date)
             .max()
             .loc[:, [p_col]]
             .rename({p_col: 'max_P'}, axis = 1))

    #Calculate maximum precipitation for day before date
    max_p_previous_day = (df.groupby(df.index.date + relativedelta(days=1))
         .max()
         .loc[:, [p_col]]
         .rename({p_col: 'max_P_previous_day'}, axis = 1))
    
    #Add maximum precipitation columns to dataframe
    df = pd.merge(df, max_p, how = 'left', left_on = 'date', right_index = True)
    df = pd.merge(df, max_p_previous_day, how = 'left', left_on = 'date', right_index = True)
    
    #Select only records with no recorded precipitation in either column
    df = df[np.isclose(df.max_P, 0, atol = 1e-5) & np.isclose(df.max_P_previous_day, 0, atol = 1e-5)]
    return df

#Full preprocessing routine for Ameriflux sites
def preprocessAmeriflux(ameriflux_path, heights, time_zone, noRain = True, ustar_thresh = None, time_start = None, time_end = None, first_month = 5, last_month = 9):
    
    #Load CSV downloaded from Ameriflux
    df = pd.read_csv(ameriflux_path, skiprows = [0,1], na_values = -9999.0, low_memory = False)
    
    #Convert the time columns to Python date/time format and set the index as the start time
    df = setTimeIndex(df, timestart_col = 'TIMESTAMP_START', timeend_col = 'TIMESTAMP_END', timezone = time_zone)
    
    #Filter days with no rain and days after any rainfall
    #This must always be before the time filter!
    if noRain == True:
        df = noDailyPrecip(df, p_col = 'P')
    
    #Filter by u* threshold
    if ustar_thresh:
        df = df[df.USTAR > ustar_thresh]
    
    #Filter by time of day
    if time_start:
        df = df.between_time(time_start, time_end)
    
    #Filter by month
    df = df[(df.index.month >= first_month) & (df.index.month <= last_month)]
    
    #Calculate VPD for each set of air temperature and relative humidity sensors
    for h in range(1, heights + 1):
        df[f'VPD_1_{h}_1'] = RH_to_VPD(RH = df[f'RH_1_{h}_1']/100, TA = df[f'TA_1_{h}_1'], formula = "Allen_1998")        
    
    #Create column with Rn-G
    df['Rn_G'] = df['NETRAD'] - df['G']
    
    #Calculate evaporative fraction as LE/(Rn-G)
    df['Evap_Frac'] = df['LE']/(df['NETRAD'] - df['G'])
    
    #Calculate specific heat of moist air
    df['cp'] = cp_moist(RH = df['RH_1_2_1'], TA = df['TA_1_2_1'], PA = df['PA'])
    
    #Calculate moist air density
    df['rho'] = moist_air_density(RH = df['RH_1_2_1'], TA = df['TA_1_2_1'], PA = df['PA'])
    
    #Calculate energy balance closure ratio
    df['Closure'] = (df['H'] + df['LE'])/(df['NETRAD'] - df['G'])

#     #Calculate the difference between leaf temperature and air temperature
#     if canopy_radtemp_col:
#         df['IR_tc_ta'] = df[canopy_radtemp_col] - df[canopy_airtemp_col]
    
    #Sort by date/time
    df = df.sort_index()
    
    #Create new columns with date/time metadata
    df['year'] = df.index.year
    df['month'] = df.index.month
    df['day'] = df.index.day
    df['time'] = df.index.hour + df.index.minute/60
    
    return df