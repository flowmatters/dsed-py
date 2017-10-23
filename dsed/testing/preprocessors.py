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
from .general import TestExecutionOptions, write_junit_style_results
from datetime import datetime

veneer.general.PRINT_SCRIPTS=True

def preprocessor_test(context,project_file,preprocessor,preprocess_params,expected):
    try:
        print('Running %s for %s'%(preprocessor.__name__,project_file))
        v = context.start_for_test(project_file)
        result = preprocessor(v,**preprocess_params)
        print('Preprocessor completed')

        orig_cols = expected.columns
        unitless_cols = [col.split(' (')[0] for col in orig_cols]
        expected = expected.rename(columns=dict(zip(orig_cols,unitless_cols)))
        results_reordered = result[expected.columns]

        assert (expected.columns==results_reordered.columns).all()

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
        assert len(results)==0
    finally:
        context.shutdown()

if __name__=='__main__':
    test_fn = sys.argv[1]
    veneer_path = os.path.abspath(sys.argv[2])
    source_version = '4.1.1' if len(sys.argv)<4 else sys.argv[3]

    tests =json.load(open(test_fn,'r'))
    wd = os.getcwd()
    results = {}
    context = TestExecutionOptions(veneer_path,source_version,44444,None)

    for test in tests:
        preprocessor = 'run_%s'%test['preprocessor']
        project_fn = test['project']
        args = test['parameters']
        expected = pd.read_csv(test['expected_results'])
        label = test.get('label','unlabelled')
        label = "%s (%s)"%(preprocessor,label)
        print("================= %s ================="%label)
        param_subst = {
            'pwd':os.getcwd()
        }
        for k,v in args.items():
            if isinstance(v,str):
                args[k] = string.Template(v).substitute(param_subst)

        try:
            start_t = datetime.now()
            preprocessor_test(context,project_fn,getattr(preprocessors,preprocessor),args,expected)
            print("SUCCESS: %s"%label)
            success=True
            msg=None
        except Exception as e:
            print('FAILED: %s with %s'%(label,str(e)))
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

