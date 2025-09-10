import os
import sys
import tempfile
import shutil
from veneer.manage import create_command_line, start, kill_all_now, print_from_all
from datetime import datetime
from dsed import Veneer
import time

DEFAULT_SOURCE_PATH='C:\\Program Files\\eWater'
PLUGINS=[
    'Dynamic_SedNet.dll',
    'GBR_DynSed_Extension.dll'
]

def arg_or_default(i,default=None):
    if len(sys.argv)<(i+1):
        return default
    return sys.argv[i]

def write_junit_style_results(results,fn,suite,test_type='RegressionTest',):
    from junit_xml import TestSuite,TestCase

    test_cases = []
    for k,v in results.items():
        tc = TestCase(k,test_type,v['elapsed'],v['message'],'')
        logger.debug(v)
        if not v['success']:
            tc.add_failure_info(v['message'])
        test_cases.append(tc)

    suite = TestSuite(suite,test_cases)
    with open(fn,'w') as f:
        TestSuite.to_file(f,[suite])

class TestServer(object):
    def __init__(self,veneer_path=None,source_version=None,port=44444,temp_dir=None):
        self.veneer_path = veneer_path if veneer_path else os.path.abspath(arg_or_default(2,''))
        self.source_version=source_version if source_version else arg_or_default(3,'4.1.1')
        self.port = port
        self.temp_dir = temp_dir
        self.source_path=arg_or_default(4,DEFAULT_SOURCE_PATH)
        if self.source_path:
            self.source_version=None

        self.plugin_path = arg_or_default(5)
        if self.plugin_path:
            self.plugins=[os.path.abspath(os.path.join(self.plugin_path,p)) for p in PLUGINS]
        else:
            self.plugins=[]

        self.error_streams = []
        self.output_streams = []


    def start_for_test(self,project_fn):
        self.processes=None
        self.delete_after=False

        if not self.temp_dir:
            self.delete_after=True
            self.temp_dir = tempfile.mkdtemp('_dsed_test')

        dest = os.path.join(self.temp_dir,'veneer_cmd')
        logger.debug(f'{self.veneer_path}, {self.source_version}, {dest}')
        veneer_cmd_path = create_command_line(self.veneer_path,self.source_version,
                                              dest=dest,force=True,source_path=self.source_path)

        start_load = datetime.now()
        self.processes,ports,((self.output_streams,_),(self.error_streams,_)) = start(project_fn,veneer_exe=veneer_cmd_path,
                                overwrite_plugins=False,ports=self.port,
                                remote=False,script=True,debug=True,
                                return_io=True,additional_plugins=self.plugins)
        end_load = datetime.now()
        elapsed_load = (end_load - start_load).total_seconds()
        logger.info('Loaded in %s seconds'%elapsed_load)

        return Veneer(port=ports[0])

    def shutdown(self):
        print_from_all(self.error_streams,'ERROR')
        print_from_all(self.output_streams,'OUTPUT')
        if self.processes:
            logger.info('Shutting down server')
            kill_all_now(self.processes)
            logger.info('Pause for the process to end...')
            time.sleep(2)
            logger.info('...Resuming')
            self.processes = None

        if self.delete_after:
            shutil.rmtree(self.temp_dir)
            self.temp_dir = None