import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

def calcBias(df, pred_col, obs_col):
    vals = df.copy()
    bias = np.mean(vals[pred_col] - vals[obs_col])
    return bias

#Create 1:1 line for range of observed x values
#bias_vals = np.arange(np.nanmin(vals[xcol]).round(1), np.nanmax(vals[xcol]).round(1), 0.1)

#Empirical fit line
#bias_mod = smf.ols(ycol + ' ~ ' + xcol, data = vals).fit().params

#Mean difference between empirical fit line and 1:1 line
#bias = np.mean((bias_vals * bias_mod[1] + bias_mod[0]) - bias_vals).round(2)