'''
Budgets of Openwater Dynamic SedNet results, in Openwater's terms.

A budget table has one row per budget term per model element, with columns:

  element_type   - Catchment, Link or Node
  catchment      - for catchment and link terms
  cgu            - for catchment terms
  node_name      - for node terms
  constituent    - including 'Flow' for water
  budget_element - eg Hillslope, Gully, Streambank, Outflow (see dsed.ow.vocabulary)
  process        - Initial, Supply, In Flow, Loss, Yield or Residual
  value          - total over the reporting window (or the stored amount, for Initial and Residual)
  units          - kg or m3
  model, variable - the Openwater model type and variable the term came from

Budget tables are independent of Source's reporting conventions. dsed.ow.source_format converts them
to Source's summary report formats.
'''
import logging
import pandas as pd
from openwater.file import _tabulate_model_scalars_from_file
from dsed.const import PER_SECOND_TO_PER_DAY, MM_TO_M
from dsed.ow import vocabulary as v
from dsed.ow.structure import FINE_SEDIMENT, COARSE_SEDIMENT

logger = logging.getLogger(__name__)

BUDGET_COLUMNS = ['element_type','catchment','cgu','node_name','constituent','budget_element','process',
                  'value','units','model','variable']

# Which tag identifies the model element, for each element type
ELEMENT_TAGS = {
  v.CATCHMENT:'catchment',
  v.LINK:'catchment',
  v.NODE:'node_name'
}
LOCATION_TAGS = ['catchment','link_name','node_name']

# Openwater tags that are part of the model structure, rather than of the budget
STRUCTURAL_CONSTITUENTS = {'NLeached':'N_DIN'}

def window_total(timeseries, window):
  '''Total of each time series over the reporting window.'''
  if window.start is not None:
    timeseries = timeseries[timeseries.index>=window.start]
  if window.end is not None:
    timeseries = timeseries[timeseries.index<=window.end]
  return timeseries.sum()

def value_at_window_end(timeseries, window):
  '''Value of each time series on the last day of the reporting window.'''
  if window.end is not None:
    timeseries = timeseries[timeseries.index<=window.end]
  return timeseries.iloc[-1]

def value_before_window_start(timeseries, window):
  '''Value of each time series on the day before the reporting window starts.'''
  timeseries = timeseries[timeseries.index<window.start]
  return timeseries.iloc[-1]

def _as_set(value):
  if isinstance(value,(list,tuple,set,frozenset)):
    return set(value)
  return {value}

def _fixed_tags(ow, model):
  '''Tags with a single value for every node of the model (Openwater stores these on the map, not as dimensions).'''
  attrs = ow.model['/MODELS/%s/map'%model].attrs
  return {k:(val.decode() if hasattr(val,'decode') else val) for k,val in attrs.items()
          if k not in ('DIMS','PROCESSES')}

def _query_tags(ow, model, filters):
  '''
  The tags to pass to Openwater's results functions for `filters`, or None if no node of the model can match.
  '''
  dims = ow.dims_for_model(model)
  fixed = _fixed_tags(ow, model)
  query = {}
  for tag, value in filters.items():
    allowed = _as_set(value)
    if tag in dims:
      allowed = allowed & set(ow.dim(tag))
      if not allowed:
        return None
      query[tag] = list(allowed) if len(allowed)>1 else allowed.pop()
    elif tag in fixed:
      if fixed[tag] not in allowed:
        return None
    else:
      return None
  return query

def _tabulate(series, model, element_type, fixed):
  '''Rows (one per model element) from a reduced series indexed by the model's tags.'''
  element_tag = ELEMENT_TAGS[element_type]
  if element_tag not in series.index.names:
    if fixed.get(element_tag) is None:
      return pd.DataFrame()

  table = series.rename('value').reset_index()
  for tag, value in fixed.items():
    if tag not in table.columns:
      table[tag] = value

  # Openwater gives nodes 'dummy-<tag>' for tags they don't have. Keep the nodes of this element type
  # (a real value for its identifying tag, and dummies for the other location tags), then clear the dummies.
  table = table[~table[element_tag].astype(str).str.startswith('dummy-')]
  for tag in LOCATION_TAGS:
    if tag != element_tag and tag in table.columns:
      table = table[table[tag].astype(str).str.startswith('dummy-')]
  tag_columns = [c for c in table.columns if c != 'value']
  table = table.copy()
  table[tag_columns] = table[tag_columns].mask(table[tag_columns].astype(str).apply(lambda col: col.str.startswith('dummy-')))
  if 'hru' in table.columns and 'cgu' not in table.columns:
    table = table.rename(columns={'hru':'cgu'})
  return table

class BudgetItem(object):
  '''
  A budget term from a time series model variable.

  `reduce` turns the time series into one value per node (default: total over the reporting window).
  `filters` restrict the nodes used; `assign` sets tag values on the resulting rows.
  '''
  def __init__(self, model, variable, element_type, budget_element, process,
               conversion_factor=PER_SECOND_TO_PER_DAY, units=v.KG, reduce=window_total,
               filters=None, assign=None):
    self.model = model
    self.variable = variable
    self.element_type = element_type
    self.budget_element = budget_element
    self.process = process
    self.conversion_factor = conversion_factor
    self.units = units
    self.reduce = reduce
    self.filters = filters or {}
    self.assign = assign or {}

  def __call__(self, budget):
    ow = budget.results
    if self.model not in ow.models():
      return pd.DataFrame()
    query = _query_tags(ow, self.model, self.filters)
    if query is None:
      return pd.DataFrame()
    timeseries = ow.all_time_series(self.model, self.variable, **query)
    if not timeseries.shape[1]:
      return pd.DataFrame()
    reduced = self.reduce(timeseries, budget).copy() * self.conversion_factor
    reduced.name = None
    table = _tabulate(reduced, self.model, self.element_type, _fixed_tags(ow, self.model))
    return self._finish(table)

  def _finish(self, table):
    if not len(table):
      return pd.DataFrame()
    for tag, value in self.assign.items():
      table[tag] = value
    if 'constituent' not in table.columns:
      raise Exception(f'No constituent for {self.model}.{self.variable} ({self.budget_element})')
    table['constituent'] = table['constituent'].replace(STRUCTURAL_CONSTITUENTS)
    table['element_type'] = self.element_type
    table['budget_element'] = self.budget_element
    table['process'] = self.process
    table['units'] = self.units
    table['model'] = self.model
    table['variable'] = self.variable
    return table.reindex(columns=BUDGET_COLUMNS)

class StateBudgetItem(BudgetItem):
  '''
  A budget term from a model's initial or final states (process Initial or Residual).

  States are only stored at the start and end of the run. When the reporting window starts after the run
  starts (Initial) or ends before the run ends (Residual), the states don't apply. The item then uses
  `timeseries` (the name of a model output that tracks the state) if given, otherwise it reports nothing and
  logs a warning.
  '''
  def __init__(self, model, state, element_type, budget_element, process, timeseries=None,
               conversion_factor=1.0, **kwargs):
    super().__init__(model, state, element_type, budget_element, process,
                     conversion_factor=conversion_factor, **kwargs)
    self.timeseries = timeseries

  @property
  def selection(self):
    return 'initial' if self.process==v.INITIAL else 'final'

  def __call__(self, budget):
    ow = budget.results
    if self.model not in ow.models():
      return pd.DataFrame()
    if not budget.states_apply(self.selection):
      return self._from_timeseries(budget)

    query = _query_tags(ow, self.model, self.filters)
    if query is None:
      return pd.DataFrame()
    states = budget.get_states(self.selection, self.model)
    for tag, value in query.items():
      states = states[states[tag].isin(_as_set(value))]
    states = states.drop(columns=[c for c in states.columns if c.startswith('_')])
    tags = [c for c in states.columns if c in ow.dims_for_model(self.model)]
    series = states.set_index(tags)[self.variable] * self.conversion_factor
    table = _tabulate(series, self.model, self.element_type, _fixed_tags(ow, self.model))
    return self._finish(table)

  def _from_timeseries(self, budget):
    if self.timeseries is None:
      if (self.model, self.variable) in budget.warned:
        return pd.DataFrame()
      budget.warned.add((self.model, self.variable))
      logger.warning(f'Skipping {self.budget_element} ({self.process}) from {self.model}.{self.variable}: '
                     'states are only available when the reporting window matches the run')
      return pd.DataFrame()
    reduce = value_at_window_end if self.selection=='final' else value_before_window_start
    item = BudgetItem(self.model, self.timeseries, self.element_type, self.budget_element, self.process,
                      conversion_factor=self.conversion_factor, units=self.units, reduce=reduce,
                      filters=self.filters, assign=self.assign)
    table = item(budget)
    if len(table):
      table['variable'] = self.variable
    return table

def _evaluate(items, budget):
  tables = [t for t in (item(budget) for item in items) if len(t)]
  if not tables:
    return pd.DataFrame(columns=BUDGET_COLUMNS)
  return pd.concat(tables, ignore_index=True)

class Budget(object):
  '''
  Budget tables for an Openwater Dynamic SedNet model run, over a reporting window.

  Parameters:
  * ow_impl - an OpenwaterDynamicSednetResults object
  * start, end - optional reporting window (defaults to the whole run)
  '''
  def __init__(self, ow_impl, start=None, end=None):
    self.impl = ow_impl
    self.results = ow_impl.results
    self.start = pd.Timestamp(start) if start is not None else None
    self.end = pd.Timestamp(end) if end is not None else None
    self._tables = {}
    self.warned = set()

  @property
  def time_period(self):
    '''(start, end) of the reporting window, for use with OpenwaterResults time_period arguments.'''
    return (self.start, self.end)

  def states_apply(self, selection):
    '''Whether the model's initial/final states coincide with the start/end of the reporting window.'''
    run = self.impl.time_period
    if selection=='initial':
      return self.start is None or self.start<=run[0]
    return self.end is None or self.end>=run[-1]

  def get_states(self, selection, model, **tags):
    f = self.results.model if selection=='initial' else self.results.results
    mmap = self.impl.model.model._map_model_dims(model)
    return _tabulate_model_scalars_from_file(f, model, mmap, 'states', **tags)

  def _cached(self, name, items_fn):
    if name not in self._tables:
      self._tables[name] = _evaluate(items_fn(), self)
    return self._tables[name]

  def catchment_table(self):
    '''Constituent generation, by catchment and CGU.'''
    def items():
      return [BudgetItem(c['model'], c['variable'], v.CATCHMENT, c['budget_element'], v.SUPPLY,
                         filters=c['filters'], assign=c['assign'])
              for c in self.impl.generation_components()]
    return self._cached('catchment', items)

  def link_table(self):
    '''Flow and constituent routing, and in-stream processes, by link (identified by catchment).'''
    return self._cached('link', self._link_items)

  def node_table(self):
    '''Storages, extractions and injected inflows, by node.'''
    return self._cached('node', self._node_items)

  def table(self):
    '''All budget terms.'''
    return pd.concat([self.catchment_table(), self.link_table(), self.node_table()], ignore_index=True)

  def _link_items(self):
    flow = dict(assign={'constituent':v.FLOW}, units=v.M3)
    items = [
      BudgetItem('StorageRouting','inflow',v.LINK,v.INFLOW,v.IN_FLOW,**flow),
      BudgetItem('StorageRouting','outflow',v.LINK,v.OUTFLOW,v.YIELD,**flow),
      StateBudgetItem('StorageRouting','S',v.LINK,v.STORAGE,v.INITIAL,timeseries='storage',**flow),
      StateBudgetItem('StorageRouting','S',v.LINK,v.STORAGE,v.RESIDUAL,timeseries='storage',**flow),
    ]

    for c in self.impl.meta['constituents']:
      model, inflow = self.impl.transport_model(c,'upstream')
      if model is None:
        continue
      _, outflow = self.impl.transport_model(c,'downstream')
      _, store = self.impl.transport_store(c)
      constituent = self._constituent_args(model,c)
      items += [
        BudgetItem(model,inflow,v.LINK,v.INFLOW,v.IN_FLOW,**constituent),
        BudgetItem(model,outflow,v.LINK,v.OUTFLOW,v.YIELD,**constituent),
        StateBudgetItem(model,store,v.LINK,v.STORAGE,v.INITIAL,**constituent),
        StateBudgetItem(model,store,v.LINK,v.STORAGE,v.RESIDUAL,**constituent),
      ]

      bank_model, bank_variable = self.impl.streambank_output(c)
      if bank_model is not None:
        items.append(BudgetItem(bank_model,bank_variable,v.LINK,v.STREAMBANK,v.SUPPLY,
                                **self._constituent_args(bank_model,c)))

      processes = [
        ('floodplain_deposition',v.FLOOD_PLAIN_DEPOSITION,v.LOSS),
        ('channel_deposition',v.CHANNEL_DEPOSITION,v.LOSS),
        ('decay',v.STREAM_DECAY,v.LOSS),
        ('point_source',v.POINT_SOURCE,v.SUPPLY),
      ]
      for process, element, role in processes:
        p_model, p_variable = self.impl.instream_process(c,process)
        if p_model is not None:
          items.append(BudgetItem(p_model,p_variable,v.LINK,element,role,**self._constituent_args(p_model,c)))

      store_model, store_variable = self.impl.instream_process(c,'channel_store')
      if store_model is not None:
        items += [StateBudgetItem(store_model,store_variable,v.LINK,v.CHANNEL_STORAGE,process,
                                  **self._constituent_args(store_model,c))
                  for process in (v.INITIAL,v.RESIDUAL)]
    return items

  def _constituent_args(self, model, c):
    '''Select the nodes of `model` for constituent `c` (by tag where the model has one, otherwise all nodes).'''
    ow = self.results
    has_tag = model in ow.models() and \
      (('constituent' in ow.dims_for_model(model)) or ('constituent' in _fixed_tags(ow, model)))
    if has_tag:
      return dict(filters={'constituent':c},assign={'constituent':c})
    return dict(assign={'constituent':c})

  def _node_items(self):
    flow = dict(assign={'constituent':v.FLOW}, units=v.M3)
    items = [
      BudgetItem('Storage','outflow',v.NODE,v.OUTFLOW,v.YIELD,**flow),
      BudgetItem('Storage','rainfallVolume',v.NODE,v.RAINFALL,v.SUPPLY,**flow),
      BudgetItem('Storage','evaporationVolume',v.NODE,v.EVAPORATION,v.LOSS,**flow),
      StateBudgetItem('Storage','currentVolume',v.NODE,v.STORAGE,v.INITIAL,timeseries='volume',**flow),
      StateBudgetItem('Storage','currentVolume',v.NODE,v.STORAGE,v.RESIDUAL,timeseries='volume',**flow),

      BudgetItem('PartitionDemand','outflow',v.NODE,v.OUTFLOW,v.YIELD,**flow),
      BudgetItem('PartitionDemand','extraction',v.NODE,v.EXTRACTION,v.LOSS,**flow),
      BudgetItem('Input','output',v.NODE,v.INFLOW,v.SUPPLY,filters={'variable':'inflow'},**flow),

      BudgetItem('VariablePartition','output2',v.NODE,v.OUTFLOW,v.YIELD),
      BudgetItem('VariablePartition','output1',v.NODE,v.EXTRACTION,v.LOSS),
      BudgetItem('PassLoadIfFlow','inputLoad',v.NODE,v.INFLOW,v.SUPPLY),
      BudgetItem('PassLoadIfFlow','outputLoad',v.NODE,v.OUTFLOW,v.YIELD),
    ]
    for model in ['LumpedConstituentRouting','StorageParticulateTrapping']:
      items += [
        BudgetItem(model,'outflowLoad',v.NODE,v.OUTFLOW,v.YIELD),
        StateBudgetItem(model,'storedMass',v.NODE,v.STORAGE,v.INITIAL),
        StateBudgetItem(model,'storedMass',v.NODE,v.STORAGE,v.RESIDUAL),
      ]
    items.append(BudgetItem('StorageParticulateTrapping','trappedMass',v.NODE,v.RESERVOIR_DEPOSITION,v.LOSS,
                            conversion_factor=1.0))
    return items

  def climate_table(self, rr_model='Sacramento'):
    '''
    Climate and runoff depths (mm) over the reporting window, by catchment and HRU, with columns
    catchment, hru, variable (the rainfall runoff model variable), value, units.
    '''
    tables = []
    for variable in ['rainfall','actualET','runoff','baseflow']:
      tbl = self.results.table(rr_model,variable,'catchment','hru','sum','sum',time_period=self.time_period)
      tbl = tbl[tbl.index!='dummy-catchment']
      tbl.index.name = 'catchment'
      tbl = tbl.reset_index().melt(id_vars=['catchment'],var_name='hru',value_name='value')
      tbl['variable'] = variable
      tables.append(tbl)
    result = pd.concat(tables, ignore_index=True)
    result['units'] = 'mm'
    return result[['catchment','hru','variable','value','units']]

# ---------------------------------------------------------------------------------------------------
# Work in progress: budget for a single Source model, to be rebuilt as a view of Budget tables.

GULLY_MODELS = {
    'DERM RATIO': 'DynamicSednetGullyAlt',
    'SEDNET POWER': 'DynamicSednetGully'
}

HILLSLOPE_SURFACE = 'Hillslope surface soil'
HILLSLOPE_SUBSURFACE = 'Hillslope sub-surface soil'
TOTAL = 'Total'

def _gully_model(results, cgu):
    gully_types = results.meta.get('gully_types', {}) or {}
    return GULLY_MODELS[gully_types.get(cgu, 'DERM RATIO')]

def _catchment_series(ow, model, variable, **tags):
    '''
    Time series of model.variable, one column per catchment, or None if the model isn't present.

    Only filters on tags that are dimensions of the model. Openwater drops a dimension when a
    model only ever has one value for it (eg DeliveryRatio when only one FU uses it), so the
    caller must know, from the model structure, that the remaining tags are satisfied.
    '''
    if model not in ow.models():
        return None
    dims = ow.dims_for_model(model)
    tags = {k: val for k, val in tags.items() if k in dims}
    return ow.time_series(model, variable, 'catchment', **tags)

def cropping_sediment_budget(results, cgu, constituent=FINE_SEDIMENT, catchment=None):
    '''
    Budget for Source's 'Cropping Sediment (Sheet & Gully) - GBR' model, from Openwater results.

    Work in progress: not yet consistent with the standard reports, which (like Source) include the
    dry weather load in 'Hillslope surface soil'.

    Parameters:
    * results - an OpenwaterDynamicSednetResults object
    * cgu - the functional unit, eg 'Dryland Cropping'
    * constituent - 'Sediment - Fine' (default) or 'Sediment - Coarse'
    * catchment - optional. If given, return daily time series for that catchment.

    Returns a DataFrame with columns:
    * 'Hillslope surface soil' - delivered load from the cropping time series
    * 'Hillslope sub-surface soil' - dry weather (DWC) load
    * 'Gully' - delivered gully load
    * 'Total' - total generated load (equals the sum of the above)

    If catchment is None: one row per catchment, total load (kg) over the simulation.
    If catchment is given: one row per day, load in kg/day.

    Loads are *delivered* loads (after sediment delivery ratios), matching Source.
    '''
    if constituent not in (FINE_SEDIMENT, COARSE_SEDIMENT):
        raise ValueError(f'constituent must be {FINE_SEDIMENT!r} or {COARSE_SEDIMENT!r}')

    fine = constituent == FINE_SEDIMENT
    ow = results.results
    tags = dict(cgu=cgu, constituent=constituent)

    sources = {
        HILLSLOPE_SUBSURFACE: ('EmcDwc', 'totalLoad'),
        v.GULLY: (_gully_model(results, cgu), 'fineLoad' if fine else 'coarseLoad'),
        TOTAL: ('Sum', 'out'),
    }
    # Only FUs with a cropping soil loss time series have the time series (sheet) nodes
    if cgu in (results.meta.get('timeseries_sediment') or []):
        sources[HILLSLOPE_SURFACE] = ('DeliveryRatio', 'output')

    columns = {}
    for element, (model, variable) in sources.items():
        ts = _catchment_series(ow, model, variable, **tags)
        if ts is None:
            raise Exception(f'No {model} results for {cgu}/{constituent}. Is this FU using Cropping Sediment?')
        columns[element] = ts * PER_SECOND_TO_PER_DAY  # kg/s -> kg/day

    budget = pd.DataFrame({element: ts[catchment] if catchment is not None else ts.sum()
                           for element, ts in columns.items()})
    if HILLSLOPE_SURFACE not in budget.columns:
        budget[HILLSLOPE_SURFACE] = 0.0
    return budget[[HILLSLOPE_SURFACE, HILLSLOPE_SUBSURFACE, v.GULLY, TOTAL]]
