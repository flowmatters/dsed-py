import os
import numpy as np
import pandas as pd
from dsed.const import *

R_SQD_THRESHOLD_NUT=0.99
SSQUARES_THRESHOLD=1

def df_problems(df):
#    return df[(df['r-squared']<R_SQD_THRESHOLD_NUT)|np.isnan(df['r-squared'])|(df['ssquares']>SSQUARES_THRESHOLD)]
    return df[(df['r-squared']<R_SQD_THRESHOLD_NUT)|np.isnan(df['r-squared'])]

def grouped_lengths(df,problems,nonzero,*args):
    if not len(args):
        return [
            {
                'orig':len(df),
                'prob':len(problems),
                'nonzero':len(nonzero)
            }
        ]

    first = args[0]
    values = set(df[first])

    result = []
    for v in values:
        res = grouped_lengths(*[d[d[first]==v] for d in [df,problems,nonzero]],*args[1:])
        for row in res:
            row[first]=v
        result += res
    return result

def problem_counts(df,*args):
    problems = df_problems(df)
    nonzero = df[df['sum-orig']>0]

    return pd.DataFrame(grouped_lengths(df,problems,nonzero,*args))

if __name__=='__main__':
    print('Compare')

