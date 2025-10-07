"""
GBR Statistics Module - Moriasi

Various bivariate statistics functions typically used as Goodness Of Fit measures for
hydrologic models.

Functions modules contained here are designed to be called on pairs of DataFrames,
(observed and predicted) with equivalent columns.

All functions should work in various scenarios, such as:

* Dataframes with columns for different constituents,
* Dataframes with columns for different sites
* Daily, monthly, annual time steps
* Other timeperiods, such data aggregated to water year

Broadly speaking, these statistics will work with data that
"""

import scipy
from scipy import stats

def intersect(obs,pred):
    """
    Return the input pair of dataframes (obs,pred) with a common index made up of the intersection of
    the input indexes
    """
    if hasattr(obs,'intersect'):
        return obs.intersect(pred)
    idx = obs.index.intersection(pred.index)
    return obs.ix[idx],pred.ix[idx]

def nse(obs,pred):
    """
    Nash-Sutcliffe Efficiency
    """
    obs,pred = intersect(obs,pred)
    pred = pred.ix[obs.index] # Filter values not present in
    numerator = ((obs-pred)**2).sum()
    denominator = ((obs-obs.mean())**2).sum()
    return 1 - numerator/denominator

def PBIAS(obs,pred):
    obs,pred = intersect(obs,pred)
    top = (obs-pred).sum()
    bottom = obs.sum()
    return (top/bottom)*100
    
def rsr (obs,pred):
    obs,pred = intersect(obs,pred)
    rmse = (((obs-pred)**2).sum())**(1/2)
    stdev_obs = (((obs-obs.mean())**2).sum())**(1/2)
    return rmse/stdev_obs
# RSR could alternatively be calculated as: RSR = (1-nse)**(1/2)

def r2 (obs,pred):
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(obs,pred)
    return r_value**2

RSR_RATINGS={
    'TSS':(0.5,0.6,0.7),
    'TN':(0.5,0.6,0.7),
    'TP':(0.5,0.6,0.7),
    'Flow':(0.5,0.6,0.7)
}

PBIAS_RATINGS={
    'TSS':(10,15,20),
    'TN':(15,20,30),
    'TP':(15,20,30),
    'Flow':(5,10,15)
}

NSE_RATINGS={
    'TSS':(0.45,0.7,0.8),
    'TN':(0.35,0.5,0.65),
    'TP':(0.35,0.5,0.65),
    'Flow':(0.5,0.7,0.8)
}

R2_RATINGS={
    'TSS':(0.4,0.65,0.8),
    'TN':(0.3,0.6,0.7),
    'TP':(0.4,0.65,0.8),
    'Flow':(0.6,0.75,0.85)
}

def rsr_rating (value,category):
    ratings = RSR_RATINGS[category]
    
    if 0 <= value <= ratings[0]:
        return 'Very good'
    if value <= ratings[1]: 
        return 'Good'
    if value <= ratings[2]: 
        return 'Satisfactory'
    return 'Unsatisfactory'

def pbias_rating (value,category):
    v = abs(value)    
    ratings = PBIAS_RATINGS[category]

    if v < ratings[0]:
        return 'Very good'
    if v < ratings[1]: 
        return 'Good'
    if v < ratings[2]: 
        return 'Satisfactory'
    return 'Unsatisfactory'

def nse_rating (value,category):
    ratings = NSE_RATINGS[category]
    if value < ratings[0]:
        return 'Unsatisfactory'
    if value < ratings[1]:
        return 'Satisfactory'
    if value < ratings[2]:
        return 'Good'
    return 'Very good'

def r2_rating (value,category):
    ratings = R2_RATINGS[category]
    if value < ratings[0]:
        return 'Unsatisfactory'
    if value < ratings[1]:
        return 'Satisfactory'
    if value < ratings[2]:
        return 'Good'
    return 'Very good'



