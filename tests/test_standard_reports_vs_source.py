'''
Regression tests: Openwater standard reports vs the equivalent Source summary CSVs.

Reporting should reproduce Source wherever the Openwater model reproduces the Source time series.
Differences that come from the models rather than reporting are specific to a model pair, so they
are listed in a known differences file kept with the data (see dsed.testing.source_reports.load_known_differences).
test_known_differences_still_differ fails when one goes away, so the file stays current.

See conftest.py for the environment variables that configure the data.
'''
import pytest
from dsed.testing.source_reports import (
    MATCH, ZERO_ONLY, summarise_differences, explained_by, matches_filter, overall_elements_explained
)

def _differences(comparison):
    return comparison[~comparison.status.isin({MATCH} | ZERO_ONLY)]

def _unexplained(rows, known_differences):
    return rows[[not explained_by(row, known_differences) for _, row in rows.iterrows()]]

def test_climate_table_matches_source(compare):
    differences = _differences(compare('climate_table'))
    assert len(differences) == 0, summarise_differences(differences, ['FU', 'Element']).to_string()

def test_raw_results_match_source_except_known_differences(compare, known_differences):
    unexplained = _unexplained(_differences(compare('raw_summary_table')), known_differences)
    summary = summarise_differences(unexplained, ['ModelElementType', 'Constituent', 'FU', 'BudgetElement'])
    assert len(unexplained) == 0, '\n' + summary.to_string()

def test_raw_results_have_no_unexpected_rows(compare, known_differences):
    comparison = compare('raw_summary_table')
    zero_rows = _unexplained(comparison[comparison.status.isin(ZERO_ONLY)], known_differences)
    summary = zero_rows.groupby(['ModelElementType', 'FU', 'BudgetElement', 'status']).size()
    assert len(zero_rows) == 0, '\n' + summary.to_string()

def test_known_differences_still_differ(compare, known_differences):
    comparison = compare('raw_summary_table')
    unmatched = comparison[comparison.status != MATCH]
    gone = [reason for reason, filters in known_differences
            if not any(matches_filter(row, filters) for _, row in unmatched.iterrows())]
    assert not gone, f'No longer differ from Source - remove from the known differences file: {gone}'

def test_overall_summary_matches_source_except_known_differences(compare, known_differences):
    explained = overall_elements_explained(compare('raw_summary_table'), known_differences)
    differences = _differences(compare('overall_summary_table'))
    unexplained = differences[[(row.Constituent, row.MassBalanceElement) not in explained
                               for _, row in differences.iterrows()]]
    assert len(unexplained) == 0, '\n' + unexplained.to_string()

def test_overall_summary_is_derived_from_raw_results(ow_reports):
    raw = ow_reports.raw_summary_table()
    overall = ow_reports.overall_summary_table().set_index(['Constituent', 'MassBalanceElement']).Total_Load_in_Kg
    raw = raw[raw.Constituent != 'Flow']
    for process in ['Supply', 'Loss', 'Residual']:
        expected = raw[raw.Process == process].groupby('Constituent').Total_Load_in_Kg.sum()
        for constituent, value in expected.items():
            assert overall[(constituent, process)] == pytest.approx(value)
