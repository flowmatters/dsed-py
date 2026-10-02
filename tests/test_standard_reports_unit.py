import pandas as pd
import pytest
from types import SimpleNamespace

from dsed.ow import vocabulary as v
from dsed.ow.budgets import window_total, value_at_window_end, value_before_window_start, _tabulate
from dsed.ow import source_format as sf
from dsed.testing.source_reports import compare_tables

def raw_row(constituent, element_type, element, fu, budget_element, process, value):
    return dict(Constituent=constituent, ModelElementType=element_type, ModelElement=element, FU=fu,
                BudgetElement=budget_element, Process=process, Total_Load_in_Kg=value)

# Budgets

@pytest.fixture
def daily():
    index = pd.date_range('2000-01-01', '2000-01-10')
    return pd.DataFrame({'a': range(1, 11), 'b': range(11, 21)}, index=index)

def window(start=None, end=None):
    return SimpleNamespace(start=pd.Timestamp(start) if start else None,
                           end=pd.Timestamp(end) if end else None)

def test_window_total_includes_both_ends(daily):
    assert window_total(daily, window('2000-01-03', '2000-01-05')).to_dict() == {'a': 3 + 4 + 5, 'b': 13 + 14 + 15}
    assert window_total(daily, window()).to_dict() == {'a': 55, 'b': 155}

def test_state_like_values_at_window_boundaries(daily):
    w = window('2000-01-03', '2000-01-05')
    assert value_at_window_end(daily, w).to_dict() == {'a': 5, 'b': 15}
    assert value_before_window_start(daily, w).to_dict() == {'a': 2, 'b': 12}

def test_tabulate_keeps_nodes_of_the_element_type_and_clears_dummy_tags():
    # A model used at catchments (with a cgu) and at nodes (without)
    index = pd.MultiIndex.from_tuples([
        ('Catchment A', 'FU A', 'dummy-node_name'),
        ('dummy-catchment', 'dummy-cgu', 'Dam'),
    ], names=['catchment', 'cgu', 'node_name'])
    series = pd.Series([1.0, 2.0], index=index)

    catchments = _tabulate(series, 'PassLoadIfFlow', v.CATCHMENT, {'constituent': 'TN'})
    assert catchments[['catchment', 'cgu', 'value', 'constituent']].values.tolist() == [['Catchment A', 'FU A', 1.0, 'TN']]
    assert catchments.node_name.isna().all()

    nodes = _tabulate(series, 'PassLoadIfFlow', v.NODE, {})
    assert nodes.node_name.tolist() == ['Dam']
    assert nodes.catchment.isna().all() and nodes.cgu.isna().all()

# Source format

META = dict(
    sediments=['Sediment - Fine', 'Sediment - Coarse'],
    particulate_nutrients=['N_Particulate'],
    dissolved_nutrients=['N_DIN', 'P_FRP'],
    pesticides=[],
    pesticide_cgus=['Crop'],
    ts_load={'combos': [['Crop', 'N_DIN']]},
)

@pytest.mark.parametrize('model,element,constituent,cgu,expected', [
    ('DeliveryRatio', v.HILLSLOPE, 'Sediment - Fine', 'Crop', [('Hillslope surface soil', 'Supply')]),
    ('EmcDwc', v.HILLSLOPE, 'Sediment - Fine', 'Crop', [('Hillslope surface soil', 'Supply')]),
    ('SednetParticulateNutrientGeneration', v.HILLSLOPE, 'N_Particulate', 'Graze', [('Hillslope no source distinction', 'Supply')]),
    ('SednetParticulateNutrientGeneration', v.BASEFLOW, 'N_Particulate', 'Graze', [('Undefined', 'Supply')]),
    ('SednetDissolvedNutrientGeneration', v.QUICKFLOW, 'N_DIN', 'Graze', [('Diffuse Dissolved', 'Supply')]),
    ('EmcDwc', v.QUICKFLOW, 'N_DIN', 'Water', [('Undefined', 'Supply')]),
    ('EmcDwc', v.BASEFLOW, 'N_DIN', 'Crop', [('Seepage', 'Supply'), ('DWC Contributed Seepage', 'Other')]),
    ('EmcDwc', v.QUICKFLOW, 'P_FRP', 'Crop', [('Hillslope no source distinction', 'Supply')]),
    ('EmcDwc', v.BASEFLOW, 'P_FRP', 'Crop', [('Seepage', 'Supply')]),
    ('ApplyScalingFactor', v.LEACHED, 'N_DIN', 'Crop',
     [('Leached', 'Other'), ('TimeSeries Contributed Seepage', 'Other'), ('Seepage', 'Supply')]),
    ('PassLoadIfFlow', v.QUICKFLOW, 'P_FRP', 'Crop', []),
])
def test_catchment_labels(model, element, constituent, cgu, expected):
    assert sf.catchment_labels(model, element, constituent, cgu, META) == expected

def link_row(element, process, value, model='InstreamFineSediment', constituent='Sediment - Fine'):
    return dict(element_type=v.LINK, catchment='Catchment A', cgu=None, node_name=None, constituent=constituent,
                budget_element=element, process=process, value=value, units=v.KG, model=model, variable='x')

def test_link_rows_report_net_remobilisation_as_supply():
    table = pd.DataFrame([
        link_row(v.CHANNEL_DEPOSITION, v.LOSS, -5.0),
        link_row(v.CHANNEL_DEPOSITION, v.LOSS, 3.0, constituent='N_Particulate', model='InstreamParticulateNutrient'),
    ])
    rows = sf._link_rows(table).set_index('Constituent')
    assert (rows.BudgetElement == 'Channel Remobilisation').all()
    assert (rows.ModelElementType == 'Catchment').all() and (rows.FU == 'Stream').all()
    assert rows.Total_Load_in_Kg.to_dict() == {'Sediment - Fine': 5.0, 'N_Particulate': 0.0}

def test_link_rows_only_report_flood_plain_deposition_of_particulates():
    table = pd.DataFrame([
        link_row(v.FLOOD_PLAIN_DEPOSITION, v.LOSS, 5.0),
        link_row(v.FLOOD_PLAIN_DEPOSITION, v.LOSS, 1.0, model='InstreamDissolvedNutrientDecay', constituent='N_DIN'),
    ])
    assert sf._link_rows(table).Constituent.tolist() == ['Sediment - Fine']

def test_water_reported_in_litres():
    table = pd.DataFrame([link_row(v.OUTFLOW, v.YIELD, 2.0, model='StorageRouting', constituent=v.FLOW)])
    table['units'] = v.M3
    assert sf._in_source_units(table).value.tolist() == [2000.0]

def test_climate_table_reports_quickflow_in_metres_and_zero_for_fus_without_area():
    climate = pd.DataFrame([
        dict(catchment='Catchment A', hru=fu, variable=var, value=value, units='mm')
        for fu in ['FU A', 'FU B']
        for var, value in [('rainfall', 1000.0), ('actualET', 600.0), ('runoff', 300.0), ('baseflow', 100.0)]
    ])
    areas = pd.DataFrame(dict(catchment=['Catchment A', 'Catchment A'], cgu=['FU A', 'FU B'], area=[10.0, 0.0]))
    table = sf.climate_table(climate, areas).set_index(['FU', 'Element']).Depth_m

    assert table[('FU A', 'Rainfall')] == pytest.approx(1.0)
    assert table[('FU A', 'Runoff (QuickFlow)')] == pytest.approx(0.2)
    assert table[('FU A', 'Baseflow')] == pytest.approx(0.1)
    assert (table.loc['FU B'] == 0.0).all()

def test_overall_summary_sums_raw_by_process_and_exports_outlet_node_yield():
    raw = pd.DataFrame([
        raw_row('TN', 'Catchment', 'Catchment A', 'FU A', 'Gully', 'Supply', 10.0),
        raw_row('TN', 'Catchment', 'Catchment A', 'FU A', 'Hillslope surface soil', 'Supply', 5.0),
        raw_row('TN', 'Link', 'Catchment A', 'Link', 'Stream Decay', 'Loss', 2.0),
        raw_row('TN', 'Link', 'Catchment A', 'Stream', 'Residual Link Storage', 'Residual', 1.0),
        raw_row('TN', 'Link', 'Catchment A', 'Stream', 'Link Yield', 'Yield', 12.0),
        raw_row('TN', 'Node', 'Outlet', 'Node', 'Node Yield', 'Yield', 12.0),
        raw_row('TN', 'Node', 'Dam', 'Node', 'Node Yield', 'Yield', 7.0),
        raw_row('Flow', 'Link', 'Catchment A', 'Stream', 'Link Initial Load', 'Supply', 100.0),
    ])
    result = sf.overall_summary(raw, {'Outlet'}).set_index('MassBalanceElement').Total_Load_in_Kg

    assert result.to_dict() == {'Supply': 15.0, 'Loss': 2.0, 'Residual': 1.0, 'Export': 12.0}

def test_overall_summary_reports_every_element_for_every_constituent():
    raw = pd.DataFrame([raw_row('TP', 'Catchment', 'Catchment A', 'FU A', 'Gully', 'Supply', 1.0)])
    result = sf.overall_summary(raw, set())

    assert list(result.MassBalanceElement) == ['Supply', 'Loss', 'Residual', 'Export']
    assert list(result.Total_Load_in_Kg) == [1.0, 0.0, 0.0, 0.0]

# Comparison with Source

def test_compare_tables_classifies_rows():
    keys = dict(Constituent='TN', MassBalanceElement=None)
    source = pd.DataFrame([
        dict(keys, MassBalanceElement='Supply', Total_Load_in_Kg=100.0),
        dict(keys, MassBalanceElement='Loss', Total_Load_in_Kg=50.0),
        dict(keys, MassBalanceElement='Residual', Total_Load_in_Kg=0.0),
        dict(keys, MassBalanceElement='Export', Total_Load_in_Kg=10.0),
    ])
    openwater = pd.DataFrame([
        dict(keys, MassBalanceElement='Supply', Total_Load_in_Kg=100.000001),
        dict(keys, MassBalanceElement='Loss', Total_Load_in_Kg=60.0),
        dict(keys, MassBalanceElement='Other', Total_Load_in_Kg=3.0),
    ])
    status = compare_tables('overall_summary_table', source, openwater).set_index('MassBalanceElement').status
    assert status.to_dict() == {
        'Supply': 'match',
        'Loss': 'mismatch',
        'Residual': 'missing in OW (zero)',
        'Export': 'missing in OW',
        'Other': 'extra in OW',
    }
