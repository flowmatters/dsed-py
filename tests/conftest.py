'''
Shared fixtures.

Regression tests against Source need a Source run's summary CSVs and the equivalent Openwater
model and results. They are skipped unless these environment variables are set:

* DSED_TEST_SOURCE_REPORTS - directory of Source summary CSVs (RawResults.csv, DSScenarioRunInfo.xml, ...)
* DSED_TEST_OW_MODEL - the Openwater model file (.h5), with its .meta.json etc alongside
* DSED_TEST_OW_OUTPUTS - optional, the Openwater results file. Defaults to <model>_outputs.h5
* DSED_TEST_OW_BIN - optional, directory of Openwater binaries (passed to openwater.discovery)
* DSED_TEST_KNOWN_DIFFERENCES - optional, JSON file of known model differences for this model pair
  (see dsed.testing.source_reports.load_known_differences). Without it, reports must match Source exactly.

Keep the data, and the known differences file, outside this repository. eg

  DSED_TEST_SOURCE_REPORTS=/data/source-runs/<scenario> \\
  DSED_TEST_OW_MODEL=/data/openwater/<model>.h5 \\
  DSED_TEST_KNOWN_DIFFERENCES=/data/source-runs/<scenario>.known_differences.json \\
  pytest tests
'''
import os
import pytest

def _env_path(name):
    value = os.environ.get(name)
    return os.path.expanduser(value) if value else None

@pytest.fixture(scope='session')
def source_reports_dir():
    path = _env_path('DSED_TEST_SOURCE_REPORTS')
    if not path:
        pytest.skip('DSED_TEST_SOURCE_REPORTS not set')
    return path

@pytest.fixture(scope='session')
def ow_reports(source_reports_dir):
    model_fn = _env_path('DSED_TEST_OW_MODEL')
    if not model_fn:
        pytest.skip('DSED_TEST_OW_MODEL not set')

    from openwater import discovery
    bin_path = _env_path('DSED_TEST_OW_BIN')
    if bin_path:
        discovery.set_exe_path(bin_path)
    discovery.discover()

    from dsed.ow.standard_reports import reports_for
    from dsed.testing.source_reports import source_run_period
    start, end = source_run_period(source_reports_dir)
    return reports_for(model_fn, _env_path('DSED_TEST_OW_OUTPUTS'), start=start, end=end)

@pytest.fixture(scope='session')
def compare(source_reports_dir, ow_reports):
    '''compare(table_name) -> comparison of the Openwater report with the Source CSV, computed once per table.'''
    from dsed.testing.source_reports import compare_tables, read_source_table
    cache = {}
    def _compare(table):
        if table not in cache:
            ow_table = getattr(ow_reports, table)()
            cache[table] = compare_tables(table, read_source_table(source_reports_dir, table), ow_table)
        return cache[table]
    return _compare

@pytest.fixture(scope='session')
def known_differences():
    from dsed.testing.source_reports import load_known_differences
    return load_known_differences(_env_path('DSED_TEST_KNOWN_DIFFERENCES'))
