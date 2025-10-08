import pandas as pd
import logging
logger = logging.getLogger(__name__)

def read_ragged_csv(fn,*args,**kwargs):
    logger.warning('Reading ragged CSV file %s',fn)
    import csv

    with open(fn,'r') as fp:
      lines=list(csv.reader(fp))
      header = lines[0]
      extra_names = ['a','b','c']
    if 'header' in kwargs:
        del kwargs['header']
    tbl = pd.read_csv(fn,skiprows=1,names=header+extra_names,*args,**kwargs)
    tbl = tbl.drop(columns=extra_names)
    return tbl


def read_source_csv(fn,*args,**kwargs):
    try:
        return pd.read_csv(fn,*args,**kwargs)
    except pd.errors.ParserError:
        return read_ragged_csv(fn,*args,**kwargs)

