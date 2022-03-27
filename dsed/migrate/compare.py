import os
import numpy as np
import pandas as pd
from dsed.const import *
import openwater.examples.catchment_model_comparison as cmc
from dsed import ow

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

def _arg_parser():
  parser = cmc._arg_parser()
  return parser

def dsed_model_comparison(m,source_files,ow_dir,component=None,con=None):
  return cmc.model_comparison(ow.OpenwaterDynamicSednetResults,m,source_files,ow_dir,component,con)

if __name__=='__main__':
  import veneer.extract_config as ec
  args = ec._parsed_args(_arg_parser())
  cmc.compare_main(dsed_model_comparison,**args)

