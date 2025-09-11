
import os
from glob import glob
import pandas as pd
import numpy as np
import logging
logger = logging.getLogger(__name__)

# 
THRESHOLD = 1e-2
THRESHOLD_FRAC = 1e-3

def findResultsFiles(path):
    entries = glob(os.path.join(path,'*'))
    result = []
    
    for entry in entries:
        if os.path.isdir(entry):
            result += findResultsFiles(entry)
        else:
            result.append(entry)
    return result
                  
def findFiles(resultsSet,base=''):
    return findResultsFiles(os.path.join(base,resultsSet))

def fileComparison(lhs,rhs):
    if not lhs.endswith('.csv'):
        return None# ['SKIPPED']

    dsL = pd.read_csv(lhs,error_bad_lines=False)
    if not os.path.exists(rhs):
        return 'file missing'

    dsR = pd.read_csv(rhs,error_bad_lines=False)
    oCols = list(dsL.columns[np.where(dsL.dtypes=='object')])
    if len(oCols)==len(dsL.columns):
        return None
    
    dsL = dsL.sort_index()#inplace=True)
    dsR = dsR.sort_index()#inplace=True)

    if len(dsL.index) != len(dsR.index) or np.any(dsL.index!=dsR.index):
        return 'mismatched index'

    if len(oCols):
        dsL.sort_values(by=oCols,inplace=True)
        dsR.sort_values(by=oCols,inplace=True)
        if set(dsL.index)==set(range(len(dsL.index))):
            dsL['new_index'] = list(range(len(dsL.index)))
            dsL = dsL.set_index('new_index')
            dsR['new_index'] = list(range(len(dsL.index)))
            dsR = dsR.set_index('new_index')
    else:
        dsL.sort_values(by=dsL.columns,inplace=True)
        dsR.sort_values(by=dsR.columns,inplace=True)

    if len(dsL.columns) != len(dsR.columns) or np.any(dsL.columns != dsR.columns):
        return 'mismatched columns', set(dsL.columns)-set(dsR.columns)

    bothNan = np.logical_and(pd.isnull(dsL),pd.isnull(dsR))
    result = np.logical_and(dsL!=dsR,np.logical_not(bothNan))
#    rows = result.transpose().any()
    numericCols = list(set(dsL.columns)-set(oCols))
    numericL = dsL[numericCols]
    numericR = dsR[numericCols]

    delta = numericL-numericR
    relative = delta / numericL
    rows = np.logical_and(np.abs(delta)>THRESHOLD,np.abs(relative)>THRESHOLD_FRAC).transpose().any()
    if rows.any():
        result = dsL[rows].copy()
        for col in numericCols:
            result['RHS_'+col] = dsR[rows][col]
            result['DIFF_'+col] = delta[rows][col]
            result['REL_DIFF_'+col] = relative[rows][col]
            
        return (result,dsL[rows],dsR[rows])
    return None

def test_comparison(lhs,rhs):
    try:
        return fileComparison(lhs,rhs)
    except Exception as e:
        return str(e)

def compare(resultsSet1,resultsSet2):
    files1 = findFiles(resultsSet1)
    files2 = [f.replace(resultsSet1,resultsSet2) for f in files1]
    logger.info('Comparing %s %s (%d files)'%(resultsSet1, resultsSet2,len(files1)),end='')
    pairs = zip(files1,files2)
   
    tmp = {lhs.split('/')[-1]:test_comparison(lhs,rhs) for lhs,rhs in pairs}
    logger.info(' DONE')
    return {key:val for key,val in tmp.items() if not val is None}, [k for k,v in tmp.items() if v is None]

    