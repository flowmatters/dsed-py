
import pandas as pd
from openwater.discovery import discover, set_exe_path
from openwater.results import OpenwaterResults
from openwater.template import ModelFile
import dsed.migrate as migrate
from veneer.utils import split_network
from veneer.extensions import _feature_id
from assertpy import assert_that
import os
import json
import geopandas as gpd

FLOW_RSQUARED_THRESHOLD = 0.98
R_SQD_THRESHOLD_FLOW=0.95
SSQUARES_THRESHOLD=1e-3
R_SQD_THRESHOLD_PEST=0.99
RUNOFF_RSQ_THRESHOLD=0.98
RUNOFF_DELTA_THRESHOLD=1e4
GULLY_MODEL='DynamicSednetGullyAlt'

def migration_test(path, link_renames={}):
    # Expection <path> to contain files needed to generation OW model
    # and <path>/results to contain comparison results
    discover()
    model, meta, network = migrate.build_ow_model(path, link_renames)
    links, nodes, catchments = split_network(network)

    # write
    time_period = pd.date_range(meta['start'], meta['end'])

    # run
    results = model.run('', '', time_period)

    # read results
    comparison = migrate.SourceOWComparison(meta, results, path, catchments)

    r_squareds = comparison.compare_flows()
    n_bad = len(r_squareds.index[r_squareds < FLOW_RSQUARED_THRESHOLD])
    frac_bad = n_bad / len(r_squareds)
    assert_that(frac_bad).is_less_than_or_equal_to(0.05)

class SourceImplementation(object):
    def __init__(self, directory):
        self.directory = directory

    def _load_csv(self, f):
        return pd.read_csv(os.path.join(self.directory, f.replace(' ', '')+'.csv'), index_col=0, parse_dates=True)


class RegressionTest(object):
    def __init__(self, expected, actual):
        self.expected = expected
        self.actual = actual
        self.results = {}
        self.comparison = migrate.SourceOWComparison(actual.meta,
                                                     actual.results,
                                                     expected.directory,
                                                     actual.catchments)

    def run(self):
        pass
