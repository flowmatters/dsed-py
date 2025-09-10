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
from dsed.compare_results import compare
from dsed.common import Veneer
from .general import TestServer, write_junit_style_results, arg_or_default
import pandas as pd
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.NOTSET)
logger.propagate = True

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

def simulation_test(context,
                    project_file,
                    expected_results_folder):
    try:
        logger.info('Running simulation test for %s'%project_file)
        start_load = datetime.now()
        v = context.start_for_test(project_file)
        end_load = datetime.now()
        elapsed_load = (end_load - start_load).total_seconds()
        logger.info('Loaded in %s seconds'%elapsed_load)

        # Set go fast options
        v.model.source_scenario_options("PerformanceConfiguration","ProcessCatchmentsInParallel",True)
        v.configure_options({
            'RunNetworksInParallel':True,
            'PreRunCatchments':True,
            'ParallelFlowPhase':True
        })
        v.model.simulation.configure_assurance_rule('Off','Data Sources result units must be commensurate with their usages')

        output_path = os.path.join(context.temp_dir,'test_outputs')
        if os.path.exists(output_path):
            shutil.rmtree(output_path)
        logger.info('Writing results to %s'%output_path)
        start_sim = datetime.now()
        result, run_url = v.run_model(ModelOutputPath=output_path,runName=RUN_NAME,**REGRESSION_TEST_RUN_OPTIONS)
        logger.debug(f'{result}, {run_url}')
        assert result==302
        assert run_url
        end_sim = datetime.now()
        elapsed_sim = (end_sim - start_sim).total_seconds()
        logger.info('Run complete in %s seconds'%elapsed_sim)

        logger.info(f'Results {glob(os.path.join(output_path,'*'))}')
        scen_name = v.model.get('scenario.Name')
        full_results_path = os.path.join(output_path,scen_name,RUN_NAME)
        errors, check_ok = compare(expected_results_folder,full_results_path)
        logger.info('Comparison complete')
        if len(errors):
            messages = ["==== ERRORS ===="]
            for fn,error in errors.items():
                messages.append("* %s"%fn)
                messages.append(str(error))
                df = error[0]
                if hasattr(df,'columns'):
                    for col in df.columns:
                        if df[col].dtype != 'object':
                            continue
                        messages.append('Differences cover %s: %s'%(col,str(set(df[col]))))
                messages.append("-"*40)
            logger.error('\n'.join(messages))
            error_fn = 'ERRORS_'+project_file.split('.')[0]+'.txt'
            open(error_fn,'w').write('\n'.join(messages))
            raise Exception('%d differences.'%len(errors))
        if not len(check_ok):
            raise Exception('no results!')

    finally:
        context.shutdown()

if __name__=='__main__':
    test_fn = arg_or_default(1)
    server = TestServer()

    tests = pd.read_csv(test_fn)
    wd = os.getcwd()
    results = {}
    for ix in range(len(tests)):
        row = tests.iloc[ix]
        os.chdir(row.Folder)
        try:
            start_t = datetime.now()
            simulation_test(server,
                            row.ProjectFn,
                            row.ExpectedResults)
            logger.info("SUCCESS: " + row.Folder)
            success=True
            msg=None
        except Exception as e:
            logger.error('FAILED: %s with %s'%(row.Folder,str(e)))
            import traceback
            traceback.print_tb(e.__traceback__)
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
