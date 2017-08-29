'''
Functionality to support regression testing of dynamic sednet
'''
import os
import sys
import json
from datetime import datetime

import tempfile
import shutil
from veneer.manage import create_command_line, start, kill_all_now
import veneer
from .compare_results import compare
import pandas as pd

RUN_NAME='regression_test'

def simulation_test(project_file,
                    expected_results_folder,
                    veneer_path,
                    source_version='4.1.1',
                    port=44444,
                    temp_dir=None):
    processes=None
    delete_after=False

    if not temp_dir:
        delete_after=True
        temp_dir = tempfile.mkdtemp('_dsed_test')

    try:
        veneer_cmd_path = create_command_line(veneer_path,source_version,dest=os.path.join(temp_dir,'veneer_cmd'),force=False)
        processes,ports = start(project_file,veneer_exe=veneer_cmd_path,
                                overwrite_plugins=True,ports=port,
                                remote=False,script=True,debug=True)

        v = veneer.Veneer(port=ports[0])
        output_path = os.path.join(temp_dir,'test_outputs')
        if os.path.exists(output_path):
            shutil.rmtree(output_path)
        result, run_url = v.run_model(ModelOutputPath=output_path,runName=RUN_NAME,IsRegressionTestRun=False)
        assert result==302

        scen_name = v.model.get('scenario.Name')
        full_results_path = os.path.join(output_path,scen_name,RUN_NAME)
        errors, check_ok = compare(expected_results_folder,full_results_path)
        if len(errors):
            print("==== ERRORS ====")
            for fn,error in errors.items():
                print("* %s"%fn)
                print(error)
                print("-"*40)
            raise Exception('%d differences.'%len(errors))
        if not len(check_ok):
            raise Exception('no results!')

    finally:
        if processes:
            kill_all_now(processes)

        if delete_after:
            shutil.rmtree(temp_dir)

def write_junit_style_results(results,fn):
    from junit_xml import TestSuite,TestCase

    test_cases = []
    for k,v in results.items():
        tc = TestCase(k,'RegressionTest',v['elapsed'],v['message'],'')
        print(v)
        if not v['success']:
            tc.add_failure_info(v['message'])
        test_cases.append(tc)

    suite = TestSuite('dynamic sednet regression tests',test_cases)
    with open(fn,'w') as f:
        TestSuite.to_file(f,[suite])

if __name__=='__main__':
    test_fn = sys.argv[1]
    veneer_path = os.path.abspath(sys.argv[2])
    source_version = '4.1.1' if len(sys.argv)<4 else sys.argv[3]

    tests = pd.read_csv(test_fn)
    wd = os.getcwd()
    results = {}
    for ix in range(len(tests)):
        row = tests.irow(ix)
        os.chdir(row.Folder)
        try:
            start_t = datetime.now()
            simulation_test(row.ProjectFn,
                            row.ExpectedResults,
                            veneer_path,
                            source_version)
            print("SUCCESS: " + row.Folder)
            success=True
            msg=None
        except Exception as e:
            print('FAILED: %s with %s'%(row.Folder,str(e)))
            success=False
            msg = str(e)
        finally:
            end_t = datetime.now()
            elapsed = (end_t - start_t).total_seconds()
            os.chdir(wd)
            results[row.Folder]={
                'success':success,
                'elapsed':elapsed,
                'message':msg
            }

    write_junit_style_results(results,'regression_test_results.xml')