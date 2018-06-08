'''
Functionality to support regression testing of dynamic sednet
'''
import os
import sys
import json
from datetime import datetime
from glob import glob

import tempfile
import shutil
from veneer.manage import create_command_line, start, kill_all_now, print_from_all
from .compare_results import compare
from .common import Veneer
from .testing.general import write_junit_style_results
import pandas as pd

RUN_NAME='regression_test'
REGRESSION_TEST_RUN_OPTIONS={
    'AssumeCommandLine':False, # Refers to Source regression test system. Outputs go to temp directory
    'AssumeCommandLine':True,
    'DoNamedNodeRecording':True,
    'DoRegionalTimeSeriesReporting':True
}

# TODO
# * Set PreRunCatchments=True, RunNetworksInParallel=True
# * Time run...

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
        veneer_cmd_path = create_command_line(veneer_path,source_version,dest=os.path.join(temp_dir,'veneer_cmd'),force=True)

        start_load = datetime.now()
        processes,ports,((o,_),(e,_)) = start(project_file,veneer_exe=veneer_cmd_path,
                                overwrite_plugins=True,ports=port,
                                remote=False,script=True,debug=True,
                                return_io=True)
        end_load = datetime.now()
        elapsed_load = (end_load - start_load).total_seconds()
        print('Loaded in %s seconds'%elapsed_load)

        v = Veneer(port=ports[0])

        # Set go fast options
        v.model.source_scenario_options("PerformanceConfiguration","ProcessCatchmentsInParallel",True)
        v.configure_options({
            'RunNetworksInParallel':True,
            'PreRunCatchments':True,
            'ParallelFlowPhase':True
        })

        output_path = os.path.join(temp_dir,'test_outputs')
        if os.path.exists(output_path):
            shutil.rmtree(output_path)
        print('Writing results to %s'%output_path)
        start_sim = datetime.now()
        result, run_url = v.run_model(ModelOutputPath=output_path,runName=RUN_NAME,**REGRESSION_TEST_RUN_OPTIONS)
        print(result,run_url)
        assert result==302
        assert run_url
        end_sim = datetime.now()
        elapsed_sim = (end_sim - start_sim).total_seconds()
        print('Run complete in %s seconds'%elapsed_sim)
        print_from_all(e,'ERROR')
        print_from_all(o,'OUTPUT')

        print('Results',glob(os.path.join(output_path,'*')))
        scen_name = v.model.get('scenario.Name')
        full_results_path = os.path.join(output_path,scen_name,RUN_NAME)
        errors, check_ok = compare(expected_results_folder,full_results_path)
        print('Comparison complete')
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

if __name__=='__main__':
    test_fn = sys.argv[1]
    veneer_path = os.path.abspath(sys.argv[2])
    source_version = '4.1.1' if len(sys.argv)<4 else sys.argv[3]

    tests = pd.read_csv(test_fn)
    wd = os.getcwd()
    results = {}
    for ix in range(len(tests)):
        row = tests.iloc[ix]
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

    write_junit_style_results(results,'regression_test_results.xml','dynamic sednet regression tests')
