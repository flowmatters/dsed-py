'''
Source Dynamic SedNet summary report formats, from Openwater budget tables (dsed.ow.budgets).

Everything here exists to reproduce the conventions of the summary CSVs written by Dynamic SedNet in Source
(RawResults.csv, climateTable.csv, OverallSummaryTable.csv): its labels, units, element types, zero rows and
other quirks. Comments note the Source behaviour each step reproduces, so steps can be retired when that
format is no longer needed.
'''
from itertools import product
import numpy as np
import pandas as pd
from dsed.const import M3_TO_L, MM_TO_M
from dsed.ow import vocabulary as v

VALUE_COL = 'Total_Load_in_Kg'
RAW_COLUMNS = ['Constituent','ModelElementType','ModelElement','FU','BudgetElement','Process',VALUE_COL]
CLIMATE_COLUMNS = ['Catchment','FU','Element','Depth_m']
OVERALL_COLUMNS = ['Constituent','MassBalanceElement',VALUE_COL]

# Source reports values this small as zero
EFFECTIVELY_ZERO = 1e-15

# Source labels for link and node budget terms: (budget_element, process) -> (ModelElementType, FU, BudgetElement, Process)
LINK_LABELS = {
  (v.INFLOW,v.IN_FLOW):                ('Link','Stream','Link In Flow','In Flow'),
  (v.OUTFLOW,v.YIELD):                 ('Link','Stream','Link Yield','Yield'),
  (v.STORAGE,v.INITIAL):               ('Link','Stream','Link Initial Load','Supply'),
  (v.STORAGE,v.RESIDUAL):              ('Link','Stream','Residual Link Storage','Residual'),
  # Source reports these reach processes against the catchment, with FU 'Stream'
  (v.STREAMBANK,v.SUPPLY):             ('Catchment','Stream','Streambank','Supply'),
  (v.POINT_SOURCE,v.SUPPLY):           ('Catchment','Stream','Point Source','Supply'),
  (v.CHANNEL_DEPOSITION,v.LOSS):       ('Catchment','Stream','Channel Remobilisation','Supply'),
  # Source reports the channel bed store at the end of the run as deposition
  (v.CHANNEL_STORAGE,v.RESIDUAL):      ('Link','Link','Stream Deposition','Loss'),
  (v.FLOOD_PLAIN_DEPOSITION,v.LOSS):   ('Link','Link','Flood Plain Deposition','Loss'),
  (v.STREAM_DECAY,v.LOSS):             ('Link','Link','Stream Decay','Loss'),
}
NODE_LABELS = {
  (v.OUTFLOW,v.YIELD):                 'Node Yield',
  (v.EXTRACTION,v.LOSS):               'Extraction',
  (v.RAINFALL,v.SUPPLY):               'Rainfall',
  (v.EVAPORATION,v.LOSS):              'Evaporation',
  (v.RESERVOIR_DEPOSITION,v.LOSS):     'Reservoir Deposition',
  (v.STORAGE,v.INITIAL):               'Node Initial Load',
  (v.STORAGE,v.RESIDUAL):              'Residual Node Storage',
}
SOURCE_PROCESS = {v.INITIAL:'Supply'}  # Source counts initial storage as supply

# Source only reports flood plain deposition for sediment and particulate nutrients
FLOOD_PLAIN_MODELS = {'InstreamFineSediment','InstreamParticulateNutrient'}

CLIMATE_LABELS = {
  'rainfall':'Rainfall',
  'actualET':'Actual ET',
  'baseflow':'Baseflow',
  'quickflow':'Runoff (QuickFlow)',
}

def catchment_labels(model, budget_element, constituent, cgu, meta):
  '''
  Source (BudgetElement, Process) labels for a catchment generation budget term.

  Source labels depend on which Source model generated a load, so this reconstructs that from the
  Openwater model and budget element. Returns a list, since Source reports some loads under several labels,
  or an empty list for terms Source doesn't report.
  '''
  sediments = set(meta['sediments'])
  dissolved = set(meta['dissolved_nutrients'])
  pesticides = set(meta['pesticides'])
  cropping_cgus = set(meta.get('pesticide_cgus') or [])
  ts_load = meta.get('ts_load') or {}
  ts_load_combos = {tuple(combo) for combo in ts_load.get('combos',[])}

  if budget_element == v.GULLY:
    return [('Gully','Supply')]
  if budget_element == v.HILLSLOPE:
    if constituent in sediments:
      # Includes the dry weather load of Cropping Sediment, which Source doesn't separate
      return [('Hillslope surface soil','Supply')]
    return [('Hillslope no source distinction','Supply')]
  if model == 'SednetDissolvedNutrientGeneration':
    return [('Diffuse Dissolved','Supply')]
  if model == 'SednetParticulateNutrientGeneration':
    return [('Undefined','Supply')]
  if budget_element == v.LEACHED:
    # Source's Leached should be before the seepage delivery ratio; Openwater only has it after
    return [('Leached','Other'),('TimeSeries Contributed Seepage','Other'),('Seepage','Supply')]
  if model == 'ApplyScalingFactor':
    return [('Hillslope no source distinction','Supply')]
  if model == 'EmcDwc':
    if (cgu, constituent) in ts_load_combos:
      # Dry weather component of a time series load model (eg GBR DIN)
      labels = [('Seepage','Supply')]
      if budget_element == v.BASEFLOW:
        labels.append(('DWC Contributed Seepage','Other'))
      return labels
    if (cgu in cropping_cgus) and \
       ((constituent in pesticides) or (constituent in dissolved and constituent.startswith('P'))):
      # Cropping pesticide and dissolved phosphorus models
      if budget_element == v.QUICKFLOW:
        return [('Hillslope no source distinction','Supply')]
      return [('Seepage','Supply')]
    return [('Undefined','Supply')]
  return []

def _zero_rows(**columns):
  '''Rows of zeros for every combination of the given column values (Source reports these explicitly).'''
  names = list(columns.keys())
  values = [[val] if isinstance(val,str) else list(val) for val in columns.values()]
  table = pd.DataFrame(list(product(*values)),columns=names)
  table[VALUE_COL] = 0.0
  return table

def _catchment_rows(table, meta):
  if not len(table):
    return pd.DataFrame(columns=RAW_COLUMNS)
  keys = ['model','budget_element','constituent','cgu']
  combos = table[keys].drop_duplicates()
  labels = []
  for _, row in combos.iterrows():
    for element, process in catchment_labels(row.model, row.budget_element, row.constituent, row.cgu, meta):
      labels.append(dict(row, BudgetElement=element, Process=process))
  labelled = table.merge(pd.DataFrame(labels,columns=keys+['BudgetElement','Process']),on=keys)
  labelled = labelled.rename(columns={'catchment':'ModelElement','cgu':'FU','constituent':'Constituent','value':VALUE_COL})
  labelled['ModelElementType'] = 'Catchment'
  return labelled[RAW_COLUMNS]

def _link_rows(table):
  table = table[~((table.budget_element==v.FLOOD_PLAIN_DEPOSITION)&(~table.model.isin(FLOOD_PLAIN_MODELS)))]
  keys = list(LINK_LABELS.keys())
  table = table[[(e,p) in LINK_LABELS for e,p in zip(table.budget_element,table.process)]].copy()
  labels = [LINK_LABELS[(e,p)] for e,p in zip(table.budget_element,table.process)]
  table[['ModelElementType','FU','BudgetElement','Process']] = pd.DataFrame(labels,index=table.index)
  # Source reports remobilisation from the channel store (net channel deposition < 0) as supply
  remob = table.BudgetElement=='Channel Remobilisation'
  table.loc[remob,'value'] = np.where(table.loc[remob,'value']>0,0.0,table.loc[remob,'value'].abs())
  table = table.rename(columns={'catchment':'ModelElement','constituent':'Constituent','value':VALUE_COL})
  return table[RAW_COLUMNS]

def _node_rows(table):
  rows = []
  for (element, process), label in NODE_LABELS.items():
    subset = table[(table.budget_element==element)&(table.process==process)].copy()
    subset['BudgetElement'] = label
    subset['Process'] = SOURCE_PROCESS.get(process,process)
    rows.append(subset)
  inflows = table[(table.budget_element==v.INFLOW)&(table.process==v.SUPPLY)]
  # Source reports injected flows as 'Node Injected Inflow', and also as the node's yield
  flows = inflows[inflows.model=='Input']
  rows.append(flows.assign(BudgetElement='Node Injected Inflow',Process='Supply'))
  rows.append(flows.assign(BudgetElement='Node Yield',Process='Yield'))
  loads = inflows[inflows.model!='Input']
  rows.append(loads.assign(BudgetElement='Node Injected Mass',Process='Supply'))
  result = pd.concat(rows,ignore_index=True)
  result['ModelElementType'] = 'Node'
  result['FU'] = 'Node'
  result = result.rename(columns={'node_name':'ModelElement','constituent':'Constituent','value':VALUE_COL})
  return result[RAW_COLUMNS]

def _in_source_units(table):
  '''Source reports water in litres.'''
  table = table.copy()
  water = table.units==v.M3
  table.loc[water,'value'] = table.loc[water,'value'] * M3_TO_L
  return table

def raw_results(budget):
  '''
  Source's RawResults table from a Budget (dsed.ow.budgets.Budget).
  '''
  impl = budget.impl
  meta = impl.meta
  catchments = [c for c in budget.results.dim('catchment') if not c.startswith('dummy-')]

  catchment_table = _in_source_units(budget.catchment_table())
  link_table = _in_source_units(budget.link_table())
  node_table = _in_source_units(budget.node_table())

  links = _link_rows(link_table)

  # Source reports the yield of outlet confluences, which Openwater doesn't model as nodes
  outlets = {o['catchment']:o['name'] for o in impl.outlets() if 'Confluence' in o['kind']}
  outlet_yields = links[(links.BudgetElement=='Link Yield')&(links.ModelElement.isin(outlets.keys()))].copy()
  outlet_yields['ModelElement'] = outlet_yields['ModelElement'].map(outlets)
  outlet_yields[['ModelElementType','FU','BudgetElement']] = ['Node','Node','Node Yield']

  storages = list(impl.model.model.parameters('Storage')['node_name']) if 'Storage' in budget.results.models() else []

  tables = [
    _catchment_rows(catchment_table, meta),
    links,
    _node_rows(node_table),
    outlet_yields,
    # Rows Source reports even though they are always zero
    _zero_rows(Constituent=meta['sediments'],ModelElementType='Catchment',ModelElement=catchments,
               FU='Water',BudgetElement='Undefined',Process='Supply'),
    _zero_rows(Constituent=meta['sediments'],ModelElementType='Catchment',ModelElement=catchments,
               FU=meta['gully_cgus'],BudgetElement='Hillslope sub-surface soil',Process='Supply'),
    _zero_rows(Constituent=[c for c in meta['sediments'] if 'Coarse' in c],ModelElementType='Catchment',
               ModelElement=catchments,FU='Stream',BudgetElement='Channel Remobilisation',Process='Supply'),
    _zero_rows(Constituent=meta['dissolved_nutrients']+meta['pesticides'],ModelElementType='Node',
               ModelElement=storages,FU='Node',BudgetElement='Reservoir Decay',Process='Loss'),
    _zero_rows(Constituent=v.FLOW,ModelElementType='Node',ModelElement=storages,
               FU='Node',BudgetElement='Infiltration',Process='Loss'),
  ]
  raw = pd.concat([t for t in tables if len(t)],ignore_index=True)
  raw = raw.groupby(RAW_COLUMNS[:-1],as_index=False,sort=False)[VALUE_COL].sum()
  raw.loc[raw[VALUE_COL].abs()<EFFECTIVELY_ZERO,VALUE_COL] = 0.0
  return raw[RAW_COLUMNS]

MASS_BALANCE_PROCESSES = ['Supply','Loss','Residual']

def overall_summary(raw, outlet_nodes):
  '''
  Source's OverallSummaryTable from a RawResults table.

  As in Source: Supply, Loss and Residual are the raw rows summed by Process, Export is the Node Yield at the
  outlet nodes, and Flow is excluded.
  '''
  raw = raw[raw.Constituent!=v.FLOW]
  constituents = sorted(set(raw.Constituent))

  by_process = raw[raw.Process.isin(MASS_BALANCE_PROCESSES)].groupby(['Constituent','Process'])[VALUE_COL].sum()
  export = raw[(raw.BudgetElement=='Node Yield')&(raw.ModelElement.isin(outlet_nodes))].groupby('Constituent')[VALUE_COL].sum()

  rows = []
  for c in constituents:
    for p in MASS_BALANCE_PROCESSES:
      rows.append((c,p,by_process.get((c,p),0.0)))
    rows.append((c,'Export',export.get(c,0.0)))
  return pd.DataFrame(rows,columns=OVERALL_COLUMNS)

def climate_table(climate, fu_areas):
  '''
  Source's climateTable from a Budget climate table (dsed.ow.budgets.Budget.climate_table).

  fu_areas: DataFrame with columns catchment, cgu, area.
  '''
  depths = climate.pivot_table(index=['catchment','hru'],columns='variable',values='value',aggfunc='sum') * MM_TO_M
  # Source reports quickflow, as runoff less baseflow
  depths['quickflow'] = depths['runoff'] - depths['baseflow']
  depths = depths[list(CLIMATE_LABELS.keys())]
  # Source reports zero depths for FUs with no area
  areas = fu_areas.set_index(['catchment','cgu'])['area']
  has_area = areas.reindex(depths.index).fillna(0.0) > 0
  depths = depths.mul(has_area, axis=0)

  table = depths.rename(columns=CLIMATE_LABELS).rename_axis(columns='Element').stack().rename('Depth_m').reset_index()
  table = table.rename(columns={'catchment':'Catchment','hru':'FU'})
  return table[CLIMATE_COLUMNS].sort_values(['Catchment','FU','Element']).reset_index(drop=True)
