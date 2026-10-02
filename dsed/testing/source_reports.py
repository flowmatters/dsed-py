'''
Compare Openwater standard reports with the summary CSVs written by Dynamic SedNet in Source.
'''
import os
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd

RTOL = 1e-4
ATOL = 1.0

MATCH = 'match'
MISMATCH = 'mismatch'
MISSING = 'missing in OW'
EXTRA = 'extra in OW'
ZERO_ONLY = {'missing in OW (zero)', 'extra in OW (zero)'}

# Source CSV, key columns, value column
TABLES = {
    'raw_summary_table': ('RawResults.csv',
                          ['Constituent', 'ModelElementType', 'ModelElement', 'FU', 'BudgetElement', 'Process'],
                          'Total_Load_in_Kg'),
    'climate_table': ('climateTable.csv', ['Catchment', 'FU', 'Element'], 'Depth_m'),
    'overall_summary_table': ('OverallSummaryTable.csv', ['Constituent', 'MassBalanceElement'], 'Total_Load_in_Kg'),
}

def source_run_period(source_dir):
    '''(start, end) Timestamps of the Source run that produced the CSVs in source_dir.'''
    root = ET.parse(os.path.join(source_dir, 'DSScenarioRunInfo.xml')).getroot()
    start = pd.Timestamp(root.findtext('startRecDate') or root.findtext('startDate'))
    end = pd.Timestamp(root.findtext('endDate'))
    return start, end

def read_source_table(source_dir, table):
    return pd.read_csv(os.path.join(source_dir, TABLES[table][0]))

def compare_tables(table, source, openwater, rtol=RTOL, atol=ATOL):
    '''
    Join Source and Openwater versions of a standard report on their key columns.

    Returns one row per key with columns 'source', 'openwater' and 'status'. Status is one of
    'match', 'mismatch', 'missing in OW', 'extra in OW', or the '(zero)' variants of the last two
    when the unmatched row is zero.
    '''
    _, keys, value = TABLES[table]
    source = source.copy()
    openwater = openwater.copy()
    for df in (source, openwater):
        for k in keys:
            df[k] = df[k].astype(str)

    s = source.groupby(keys)[value].sum().rename('source')
    o = openwater.groupby(keys)[value].sum().rename('openwater')
    joined = pd.concat([s, o], axis=1)

    src = joined.source.to_numpy()
    ow = joined.openwater.to_numpy()
    close = np.abs(ow - src) <= np.maximum(atol, rtol * np.abs(src))
    status = np.where(close, MATCH, MISMATCH).astype(object)
    status[np.isnan(ow)] = np.where(src[np.isnan(ow)] == 0, 'missing in OW (zero)', MISSING)
    status[np.isnan(src)] = np.where(ow[np.isnan(src)] == 0, 'extra in OW (zero)', EXTRA)
    joined['status'] = status
    return joined.reset_index()

def summarise_differences(comparison, group):
    '''Count and total the non-matching rows of a comparison, grouped by the given columns.'''
    bad = comparison[~comparison.status.isin({MATCH} | ZERO_ONLY)]
    summary = bad.groupby(group + ['status']).agg(n=('source', 'size'),
                                                  source=('source', 'sum'),
                                                  openwater=('openwater', 'sum'))
    summary['ratio'] = summary.openwater / summary.source
    return summary.reset_index()

def load_known_differences(path):
    '''
    Known differences between a particular Openwater model and Source model, from a JSON file.

    Differences that come from the models (rather than from reporting) are specific to a model pair,
    so they are kept with the data, not in the code. The file holds a list of entries like:

      {"reason": "Model: ...", "where": {"FU": ["..."], "BudgetElement": ["..."]}}

    where "where" maps RawResults columns to the values that the difference applies to.
    Returns a list of (reason, {column: set(values)}).
    '''
    import json
    if not path:
        return []
    with open(path) as fp:
        entries = json.load(fp)
    return [(e['reason'], {col: set(values) for col, values in e['where'].items()}) for e in entries]

def matches_filter(row, filters):
    return all(row[col] in values for col, values in filters.items())

def explained_by(row, known_differences):
    '''Reasons, from known_differences, that cover a row of a raw results comparison.'''
    return [reason for reason, filters in known_differences if matches_filter(row, filters)]

def overall_elements_explained(raw_comparison, known_differences):
    '''
    (Constituent, MassBalanceElement) pairs of the overall summary that can differ from Source because
    of known raw results differences: Supply, Loss and Residual through rows with that Process, and
    Export through Node Yield rows.
    '''
    differences = raw_comparison[~raw_comparison.status.isin({MATCH} | ZERO_ONLY)]
    explained = set()
    for _, row in differences.iterrows():
        if not explained_by(row, known_differences):
            continue
        if row.Process in ('Supply', 'Loss', 'Residual'):
            explained.add((row.Constituent, row.Process))
        if row.BudgetElement == 'Node Yield':
            explained.add((row.Constituent, 'Export'))
    return explained
