'''
Functionality to support regression testing of dynamic sednet
'''
import os
import sys
import json

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
            assert False
        assert len(check_ok)

    finally:
        if processes:
            kill_all_now(processes)

        if delete_after:
            shutil.rmtree(temp_dir)

if __name__=='__main__':
    test_fn = sys.argv[1]
    veneer_path = sys.argv[2]
    source_version = '4.1.1' if len(sys.argv)<4 else sys.argv[3]

    tests = pd.read_csv(test_fn)
    wd = os.getcwd()
    results = {}
    for ix in range(len(tests)):
        row = tests.irow(ix)
        os.chdir(row.Folder)
        try:
            simulation_test(row.ProjectFn,
                            row.ExpectedResults,
                            veneer_path,
                            source_version)
            print("SUCCESS: " + row.Folder)
            results[row.Folder]='SUCCESS'
        except Exception as e:
            print('FAILED: '+row.Folder)
            results[row.Folder]='FAILURE: %s'%str(e)
        finally:
            os.chdir(wd)
    print(json.dumps(results))