'''
Testing Dynamic Sednet preprocessors
'''

import veneer
import os
import json
import pandas as pd
import numpy as np
import sys
import string
from dsed import preprocessors
from .general import TestServer, write_junit_style_results, arg_or_default
from datetime import datetime
import traceback
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.NOTSET)
logger.propagate = True

veneer.general.PRINT_SCRIPTS=True

def assert_all_equal(exp,act):
    if not (exp==act).all():
        logger.critical("Expected: %s,\n Actual: %s"%(str(exp),str(act)))
    assert (exp==act).all()

def assert_empty(results):
    if len(results):
        logger.debug('Differences:')
        logger.debug('\n'.join([str(r) for r in results]))
    assert len(results)==0

def _compare(expected,result,label):
    orig_cols = expected.columns
    unitless_cols = [col.split(' (')[0] for col in orig_cols]

    expected = expected.rename(columns=dict(zip(orig_cols,unitless_cols)))
    results_reordered = result[expected.columns]

    assert_all_equal(expected.columns,results_reordered.columns)
    
    results=[]
    for col in expected.columns:
        if results_reordered.dtypes[col]==np.dtype('float64'):
            exp = expected[col]
            res = results_reordered[col]
            if not (np.abs(exp-res)<1e-6).all():
                results.append(col)
        elif results_reordered.dtypes[col]==np.dtype('int64'):
            exp = expected[col]
            res = results_reordered[col]
            if not (exp==res).all():
                results.append(col)
        else:
            exp = expected[col].fillna('')
            res = results_reordered[col].fillna('')
            if not (exp==res).all():
                results.append(col)
    if len(results):
        expected.to_csv(label+'_expected.csv')
        results_reordered.to_csv(label+'_actual.csv')
    assert_empty(results)

def preprocessor_test(context,project_file,preprocessor,preprocess_params,expected):
    try:
        logger.info('Running %s for %s'%(preprocessor.__name__,project_file))
        v = context.start_for_test(project_file)
        result = preprocessor(v,**preprocess_params)
        logger.info('Preprocessor completed')
        failed = False
        if isinstance(expected,dict):
            for key,expected_df in expected.items():
                try:
                    _compare(expected_df,result[key],project_file.split('.')[0]+'_'+key)
                except:
                    failed = True
        else:
            _compare(expected,result,project_file.split('.')[0])
        assert not failed
    finally:
        context.shutdown()

if __name__=='__main__':
    test_fn = arg_or_default(1)
    veneer_path = os.path.abspath(arg_or_default(2))
    source_version = arg_or_default(3,'4.1.1')

    tests =json.load(open(test_fn,'r'))
    wd = os.getcwd()
    results = {}
    context = TestServer(port=44444)

    for test in tests:
        preprocessor = 'run_%s'%test['preprocessor']
        project_fn = test['project']
        args = test['parameters']
        expected_fns = test['expected_results']

        if isinstance(expected_fns,str):
            expected = pd.read_csv(expected_fns)
        else:
            expected = {key:pd.read_csv(fn) for key,fn in expected_fns.items()}
        label = test.get('label','unlabelled')
        label = "%s (%s)"%(preprocessor,label)
        logger.info("================= %s ================="%label)
        param_subst = {
            'pwd':os.getcwd().replace('\\','/')
        }
        for k,v in args.items():
            if isinstance(v,str):
                args[k] = string.Template(v).substitute(param_subst)

        try:
            start_t = datetime.now()
            preprocessor_test(context,project_fn,getattr(preprocessors,preprocessor),args,expected)
            logger.info("SUCCESS: %s"%label)
            success=True
            msg=None
        except Exception as e:
            logger.error('FAILED: %s with %s'%(label,str(e)))
            logger.error('\n'.join(traceback.format_tb(e.__traceback__)))
            success=False
            msg = str(e)
        finally:
            end_t = datetime.now()
            elapsed = (end_t - start_t).total_seconds()
            os.chdir(wd)
            results[label]={
                'success':success,
                'elapsed':elapsed,
                'message':msg
            }

    write_junit_style_results(results,'preprocessor_test_results.xml','preprocessor tests')

