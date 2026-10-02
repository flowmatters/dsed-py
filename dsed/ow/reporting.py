from openwater.examples import OpenwaterCatchmentModelResults
from openwater.results import OpenwaterResults
from openwater.template import ModelFile
from .structure import dissolved_nutrient_ts_load
import json
import geopandas as gpd
import pandas as pd
import os
import logging
import shutil

# Reach transport models, indexed by model type, then by the constituent's
# 'position' relative to the reach: 'upstream' (load entering from the upstream
# reach), 'downstream' (load leaving this reach) and 'lateral' (the load
# entering from the subcatchment generation models — i.e. the conceptual
# "catchment output" total).
TRANSPORT_FLUXES = {
    'LumpedConstituentRouting':       {'upstream':'inflowLoad',          'downstream':'outflowLoad',   'lateral':'lateralLoad'},
    'ConstituentDecay':               {'upstream':'inflowLoad',          'downstream':'outflowLoad',   'lateral':'lateralLoad'},
    'InstreamDissolvedNutrientDecay': {'upstream':'incomingMassUpstream','downstream':'loadDownstream','lateral':'incomingMassLateral'},
    'InstreamParticulateNutrient':    {'upstream':'incomingMassUpstream','downstream':'loadDownstream','lateral':'incomingMassLateral'},
    'InstreamCoarseSediment':         {'upstream':'upstreamMass',        'downstream':'loadDownstream','lateral':'lateralMass'},
    'InstreamFineSediment':           {'upstream':'upstreamMass',        'downstream':'loadDownstream','lateral':'lateralMass'},
}

TRANSPORT_POSITIONS = ('upstream','downstream','lateral')

# Mass stored in the reach by each transport model (a state, so only available at the start and end of a run)
TRANSPORT_STORES = {
    'LumpedConstituentRouting':       'storedMass',
    'ConstituentDecay':               'storedMass',
    'InstreamDissolvedNutrientDecay': 'totalStoredMass',
    'InstreamParticulateNutrient':    'instreamStoredMass',
    'InstreamCoarseSediment':         'totalStoredMass',
    'InstreamFineSediment':           'totalStoredMass',
}

# Other in-stream processes, by transport model. Values ending in 'Store'/'StoredMass' are states.
INSTREAM_PROCESSES = {
    'InstreamFineSediment': {
        'floodplain_deposition':'loadToFloodplain',
        'channel_deposition':'loadToChannelDeposition', # net: negative when remobilising
        'channel_store':'channelStoreFine',
    },
    'InstreamCoarseSediment': {
        'channel_store':'channelStore',
    },
    'InstreamParticulateNutrient': {
        'floodplain_deposition':'loadToFloodplain',
        'channel_deposition':'loadDeposited', # net: negative when remobilising
        'channel_store':'channelStoredMass',
    },
    'InstreamDissolvedNutrientDecay': {
        'floodplain_deposition':'loadToFloodplain',
        'decay':'decayedLoad',
        'point_source':'loadFromPointSource',
    },
    'ConstituentDecay': {
        'decay':'decayedLoad',
    },
}
INSTREAM_PROCESS_NAMES = ('floodplain_deposition','channel_deposition','channel_store','decay','point_source')

# Downstream output flux for any model that can match a reporting node — both
# transport models (above) and generation models. flux_tags_for_node() uses
# this when a Veneer node name matches an OW node directly.
DOWNSTREAM_FLUXES = {m: f['downstream'] for m, f in TRANSPORT_FLUXES.items()}
DOWNSTREAM_FLUXES.update({
    'EmcDwc':'outflowLoad',
    'Sum':'outflowLoad',
    'SednetDissolvedNutrientGeneration':'totalLoad',
    'SednetParticulateNutrientGeneration':'totalLoad',
    'PassLoadIfFlow':'outputLoad',
    'StorageParticulateTrapping':'outflowLoad',
    'VariablePartition':'output2',  # ??????
})

NETWORK_SIDECARS = ('nodes', 'links', 'catchments')
SIDECAR_SUFFIXES = ('.meta.json',) + tuple(f'.{c}.json' for c in NETWORK_SIDECARS)


class _MetaDict(dict):
  '''Meta dict that falls back to HDF5-derived values for a few well-known keys.

  Keys like `start`, `end`, `constituents`, `fus` can be reconstructed from the
  model file itself. Any other missing key raises KeyError with context pointing
  the user at the absent .meta.json sidecar.'''

  def __init__(self, sidecar_data, fallbacks, sidecar_path):
    super().__init__(sidecar_data)
    self._fallbacks = fallbacks
    self._sidecar_path = sidecar_path

  def _fallback(self, key):
    fn = self._fallbacks.get(key)
    if fn is None:
      return None
    val = fn()
    if val is not None:
      self[key] = val
    return val

  def __getitem__(self, key):
    if super().__contains__(key):
      return super().__getitem__(key)
    val = self._fallback(key)
    if val is not None:
      return val
    raise KeyError(
      f'Meta key {key!r} not found in sidecar ({self._sidecar_path}) '
      f'and has no HDF5 fallback. The sidecar may be missing (e.g. for a '
      f'clipped model) or lack this dsed-specific classification.')

  def get(self, key, default=None):
    try:
      return self[key]
    except KeyError:
      return default

  def __contains__(self, key):
    if super().__contains__(key):
      return True
    return self._fallback(key) is not None


class OpenwaterDynamicSednetModel(object):
  def __init__(self,fn):
    self.fn = fn
    self.ow_model_fn = self.filename_from_base('.h5')
    self.open_model()

  def filename_from_base(self,fn):
    return self.fn.replace('.h5','')+fn

  @property
  def meta(self):
    if not hasattr(self, '_meta'):
      meta_fn = self.filename_from_base('.meta.json')
      sidecar = {}
      if os.path.exists(meta_fn):
        with open(meta_fn) as fp:
          sidecar = json.load(fp)
      self._meta = _MetaDict(sidecar, self._meta_fallbacks(), meta_fn)
    return self._meta

  def _meta_fallbacks(self):
    def _time_period():
      tp = getattr(self.model, 'time_period', None)
      return tp if tp is not None and len(tp) else None
    def start():
      tp = _time_period()
      return tp[0].isoformat() if tp is not None else None
    def end():
      tp = _time_period()
      return tp[-1].isoformat() if tp is not None else None
    def _dim(name):
      try:
        vals = self.model.dim(name)
      except Exception:
        return None
      return list(vals) if vals is not None else None
    return {
      'start': start,
      'end': end,
      'constituents': lambda: _dim('constituent'),
      'fus': lambda: _dim('cgu'),
    }

  @property
  def dates(self):
    if not hasattr(self, '_dates'):
      tp = getattr(self.model, 'time_period', None)
      if tp is not None and len(tp):
        self._dates = tp
      else:
        self._dates = pd.date_range(self.meta['start'], self.meta['end'])
    return self._dates

  @property
  def nodes(self):
    self._init_network()
    return self._nodes

  @property
  def links(self):
    self._init_network()
    return self._links

  @property
  def catchments(self):
    self._init_network()
    return self._catchments

  @property
  def network(self):
    self._init_network()
    return self._network

  def _init_network(self):
    if hasattr(self, '_network'):
      return
    missing = [c for c in NETWORK_SIDECARS
               if not os.path.exists(self.filename_from_base('.'+c+'.json'))]
    if missing:
      raise FileNotFoundError(
        f'Network sidecar(s) missing for {self.ow_model_fn}: ' +
        ', '.join(f'.{c}.json' for c in missing) +
        '. Network-dependent reporting is unavailable (e.g. for a clipped '
        'or transformed model that did not copy its sidecars).')
    from veneer.general import _extend_network
    self._nodes = gpd.read_file(self.filename_from_base('.nodes.json'))
    self._links = gpd.read_file(self.filename_from_base('.links.json'))
    self._catchments = gpd.read_file(self.filename_from_base('.catchments.json'))
    raw = [json.load(open(self.filename_from_base('.'+c+'.json'),'r'))
           for c in NETWORK_SIDECARS]
    network = {
        'type':'FeatureCollection',
        'features':sum([r['features'] for r in raw],[])
    }
    if 'crs' in raw[0]:
      network['crs']=raw[0]['crs']
    self._network = _extend_network(network)

  def run(self,results_fn,overwrite=False):
    self.model.run(self.dates,results_fn,overwrite=overwrite)
    return OpenwaterDynamicSednetResults(self.ow_model_fn,results_fn)

  def open_model(self):
    _ensure_uncompressed(self.ow_model_fn)
    self.model = ModelFile(self.ow_model_fn)

  def copy_to(self,dest_fn):
    shutil.copyfile(self.ow_model_fn,dest_fn)
    dest_base = dest_fn.replace('.h5','')
    for suffix in SIDECAR_SUFFIXES:
      src = self.filename_from_base(suffix)
      if os.path.exists(src):
        shutil.copyfile(src, dest_base + suffix)
    return OpenwaterDynamicSednetModel(dest_fn)

  def catchment_for_node(self,node,exact=True):
    '''
    Find the catchment (and hence link) to use as a reporting proxy for a give node.

    Catchment will be the catchment immediately downstream of the node in the original Source model.

    Hence you would use upstream fluxes on the catchment transport model to get the equivalent fluxes from the node.

    Parameters:
    * node - the name of the node
    * exact - When false, the system will find a node with a name that matches the given node name.
              If more than one node matches and exception will be raised.

    Notes:
    * If there is more than one node downstream of the given node, an exception will be raised.

    Returns:
    * The name of the catchment
    '''
    if exact:
      node = self.network.by_name(node)
    else:
      nodes = self.network.match_name(f'.*{node}.*')
      if len(nodes) == 0:
        raise Exception('No nodes matching %s'%node)
      if len(nodes) > 1:
        raise Exception('Multiple nodes matching %s'%node)
      node = nodes[0]

    ds_links = self.network.downstream_links(node)
    assert len(ds_links)==1
    ds_link = ds_links[0]
    catchments = self.network['features'].find_by_link(ds_link['properties']['id'])
    assert len(catchments)==1
    catchment = catchments[0]
    return catchment['properties']['name']

  def generation_model(self,c,fu):
    EMC = 'EmcDwc','totalLoad'
    SUM = 'Sum','out'
    NONE = None,None

    if c in self.meta['sediments']:
        if fu in (self.meta['usle_cgus']+self.meta['cropping_cgus']+self.meta['gully_cgus']):
            return SUM
        return NONE

    if c in self.meta['pesticides']:
        if fu in self.meta['cropping_cgus']:
            return SUM
        return EMC

    if c in self.meta['dissolved_nutrients']:
        if fu in ['Water']: #,'Conservation','Horticulture','Other','Urban','Forestry']:
            return EMC

        if dissolved_nutrient_ts_load(self.meta['ts_load'],cgu=fu,constituent=c):
            return SUM

        pesticide_cgus = self.meta.get('pesticide_cgus',[])
        if (fu == 'Sugarcane') and (fu in pesticide_cgus):
            if c=='N_DIN':
                return SUM
            elif c=='N_DON':
                return EMC
            elif c.startswith('P'):
                return EMC

        if (fu == 'Bananas') and (c=='N_DIN'):
            return SUM

        if fu in self.meta['cropping_cgus'] or fu in pesticide_cgus:
            if c.startswith('P'):
                return 'PassLoadIfFlow', 'outputLoad'

        return 'SednetDissolvedNutrientGeneration', 'totalLoad'

    if c in self.meta['particulate_nutrients']:
        if (fu != 'Sugarcane') and (c == 'P_Particulate'):
            if (fu in self.meta['cropping_cgus']) or (fu in self.meta.get('timeseries_sediment',[])):
                return SUM

        for fu_cat in ['cropping_cgus','hillslope_emc_cgus','gully_cgus','erosion_cgus']:
            if fu in self.meta.get(fu_cat,[]):
                return 'SednetParticulateNutrientGeneration', 'totalLoad'

    return EMC

  def generation_components(self):
    '''
    The constituent generation budget terms, as a list of dicts with:
      budget_element - from dsed.ow.vocabulary (Hillslope, Gully, Quickflow, Baseflow, Leached)
      model, variable - the model type and output carrying the term
      filters - tags a node must match ({tag: value or list of values})
      assign - tag values to give the resulting rows (eg the constituent, for models without a constituent tag)

    Generation is made of several node types (eg a gully model, time series nodes and EmcDwc nodes summed
    together), so the role of each node is inferred here from model types, constituents and meta.
    '''
    from . import vocabulary as v
    meta = self.meta
    sediments = meta['sediments']
    particulates = meta['particulate_nutrients']
    dissolved = meta['dissolved_nutrients']
    pesticides = meta['pesticides']
    components = []
    def add(element,model,variable,filters=None,assign=None):
      components.append(dict(budget_element=element,model=model,variable=variable,
                             filters=filters or {},assign=assign or {}))

    sediment_vars = list(zip(sediments,['fineLoad' if 'Fine' in s else 'coarseLoad' for s in sediments]))
    for gully_model in ['DynamicSednetGullyAlt','DynamicSednetGully']:
      for sed,var in sediment_vars:
        add(v.GULLY,gully_model,var,assign={'constituent':sed})
    for sed in sediments:
      usle_var = 'totalFineLoad' if 'Fine' in sed else 'totalCoarseLoad'
      add(v.HILLSLOPE,'USLEFineSedimentGeneration',usle_var,assign={'constituent':sed})
      # Cropping time series (after delivery ratios), and EMC/DWC hillslope or cropping DWC loads
      add(v.HILLSLOPE,'DeliveryRatio','output',filters={'constituent':sed})
      add(v.HILLSLOPE,'EmcDwc','totalLoad',filters={'constituent':sed})

    for c in particulates:
      add(v.HILLSLOPE,'SednetParticulateNutrientGeneration','hillslopeContribution',filters={'constituent':c})
      add(v.GULLY,'SednetParticulateNutrientGeneration','gullyContribution',filters={'constituent':c})
      add(v.BASEFLOW,'SednetParticulateNutrientGeneration','slowflowConstituent',filters={'constituent':c})

    for c in dissolved:
      add(v.QUICKFLOW,'SednetDissolvedNutrientGeneration','quickflowConstituent',filters={'constituent':c})
      add(v.BASEFLOW,'SednetDissolvedNutrientGeneration','slowflowConstituent',filters={'constituent':c})

    for c in particulates + dissolved + pesticides:
      add(v.QUICKFLOW,'EmcDwc','quickLoad',filters={'constituent':c})
      add(v.BASEFLOW,'EmcDwc','slowLoad',filters={'constituent':c})
      # Time series loads, after scaling (sediment scaling nodes are intermediate steps, not budget terms)
      add(v.QUICKFLOW,'ApplyScalingFactor','output',filters={'constituent':c})

    if dissolved:
      add(v.LEACHED,'ApplyScalingFactor','output',filters={'constituent':'NLeached'},assign={'constituent':'N_DIN'})

    # Time series loads used without scaling
    for c in pesticides:
      add(v.QUICKFLOW,'PassLoadIfFlow','outputLoad',filters={'constituent':c})
    for c in dissolved:
      direct_cgus = [fu for fu in meta['fus'] if self.generation_model(c,fu)[0]=='PassLoadIfFlow']
      if direct_cgus:
        add(v.QUICKFLOW,'PassLoadIfFlow','outputLoad',filters={'constituent':c,'cgu':direct_cgus})

    return components

  def _transport_model_type(self,c):
    if c in self.meta['pesticides']:
      return 'ConstituentDecay'
    if c in self.meta['dissolved_nutrients']:
      return 'InstreamDissolvedNutrientDecay'
    if c in self.meta['particulate_nutrients']:
      return 'InstreamParticulateNutrient'
    if c == 'Sediment - Coarse':
      return 'InstreamCoarseSediment'
    if c == 'Sediment - Fine':
      return 'InstreamFineSediment'
    return None

  def transport_model(self,c,position='downstream'):
    '''
    Return (model_type, flux) for the reach transport model handling constituent `c`.

    position:
      'downstream' (default) - flux leaving the reach to the next reach downstream
      'upstream'             - flux entering the reach from the upstream reach
      'lateral'              - flux entering the reach from the subcatchment generation models
                               (i.e. the conceptual catchment output total — see catchment_output())
    '''
    if position not in TRANSPORT_POSITIONS:
      raise ValueError(
        f'Unknown position {position!r}; expected one of {TRANSPORT_POSITIONS}')
    model = self._transport_model_type(c)
    if model is None:
      return None, None
    return model, TRANSPORT_FLUXES[model][position]

  def transport_store(self,c):
    '''
    Return (model_type, state) for the mass of `c` stored in the reach, or (None, None).
    '''
    model = self._transport_model_type(c)
    if model is None:
      return None, None
    return model, TRANSPORT_STORES[model]

  def instream_process(self,c,process):
    '''
    Return (model_type, variable) for an in-stream process acting on constituent `c`, or (None, None)
    if that process isn't modelled for `c`.

    process: one of INSTREAM_PROCESS_NAMES
      'floodplain_deposition' - load deposited on the floodplain
      'channel_deposition'    - net load deposited in the channel (negative when remobilising)
      'channel_store'         - mass in the channel bed store (a state)
      'decay'                 - load lost to decay
      'point_source'          - load added by point sources
    '''
    if process not in INSTREAM_PROCESS_NAMES:
      raise ValueError(f'Unknown process {process!r}; expected one of {INSTREAM_PROCESS_NAMES}')
    model = self._transport_model_type(c)
    variable = INSTREAM_PROCESSES.get(model,{}).get(process)
    if variable is None:
      return None, None
    return model, variable

  def outlets(self):
    '''
    The network's outlet nodes, as a list of dicts with:
      name      - node name
      catchment - catchment of the link flowing into the outlet
      kind      - the node's Source node type, eg 'ConfluenceNodeModel', 'ExtractionNodeModel'
    '''
    result = []
    for node in self.network.outlet_nodes():
      props = node['properties']
      links = self.network.upstream_links(props['id'])._list
      assert len(links)==1, f'Expected one link into outlet {props["name"]}, got {len(links)}'
      result.append(dict(
        name=props['name'],
        catchment=links[0]['properties']['name'].replace('link for catchment ',''),
        kind=props.get('icon','').split('/')[-1]
      ))
    return result

  def catchment_output(self,c):
    '''
    Return (model_type, flux) representing the total `c` load leaving the subcatchments
    (the lateral input to the reach transport model — there is no node in the graph
    that represents this total directly).
    '''
    return self.transport_model(c, position='lateral')

  def streambank_output(self,c):
    '''
    Return (model_type, flux) for the streambank erosion source of constituent `c`.

    - Sediment - Fine / Sediment - Coarse: BankErosion model fluxes
    - Particulate nutrients: derived inside InstreamParticulateNutrient (loadFromStreambank)
    - Everything else: (None, None)
    '''
    if c == 'Sediment - Fine':
      return 'BankErosion', 'bankErosionFine'
    if c == 'Sediment - Coarse':
      return 'BankErosion', 'bankErosionCoarse'
    if c in self.meta['particulate_nutrients']:
      return 'InstreamParticulateNutrient', 'loadFromStreambank'
    return None, None

  def cgu_output(self,c,fu):
    '''
    Return (model_type, flux) representing the load of `c` generated by CGU `fu`.

    Thin alias for generation_model(); name parallels catchment_output().
    '''
    return self.generation_model(c, fu)

class OpenwaterDynamicSednetResults(OpenwaterCatchmentModelResults):
    def __init__(self, fn, res_fn=None):
        self.model = OpenwaterDynamicSednetModel(fn)
        self.ow_results_fn = res_fn or self.model.filename_from_base('_outputs.h5')
        self.time_period = self.model.dates

        if _file_exists(self.ow_results_fn):
          self.open_files()
        else:
          logging.info('No results file found for %s. Running model'%self.ow_results_fn)
          self.run_model()

    @property
    def meta(self):
        return self.model.meta

    @property
    def catchments(self):
        return self.model.catchments

    @property
    def network(self):
        return self.model.network

    def run_model(self):
        self.model.run(self.ow_results_fn, overwrite=True)
        self.open_results()

    def open_files(self):
        self.open_results()

    def open_results(self):
        _ensure_uncompressed(self.ow_results_fn)
        self.results = OpenwaterResults(self.model.ow_model_fn,
                                        self.ow_results_fn,
                                        self.time_period)

    def generation_model(self,c,fu):
      return self.model.generation_model(c,fu)

    def transport_model(self,c,position='downstream'):
      return self.model.transport_model(c,position=position)

    def transport_store(self,c):
      return self.model.transport_store(c)

    def instream_process(self,c,process):
      return self.model.instream_process(c,process)

    def outlets(self):
      return self.model.outlets()

    def generation_components(self):
      return self.model.generation_components()

    def catchment_output(self,c):
      return self.model.catchment_output(c)

    def streambank_output(self,c):
      return self.model.streambank_output(c)

    def cgu_output(self,c,fu):
      return self.model.cgu_output(c,fu)

    def catchment_for_node(self,node,exact=True):
       return self.model.catchment_for_node(node,exact=exact)

    def openwater_node(self,node,exact=False):
        '''
        Find the name of a matching node in the Openwater model if one and only one exists

        If exact is False, the system will find a node with a name that matches the given node name.
        If more than one node matches and exception will be raised.
        If no nodes match, None will be returned.
        '''
        actual_nodes = self.results.dim('node_name')
        if exact:
            if node in actual_nodes:
                return node
            return None
        matches = [n for n in actual_nodes if node in n]
        if len(matches) == 0:
            return None
        if len(matches) > 1:
            raise Exception('Multiple nodes matching %s'%node)
        return matches[0]

    def flux_tags_for_node(self,node,consistuent,exact_node_match=False):
        '''
        Find the flux tags for a given node.

        Parameters:
        * node - the name of the node
        * exact - When false, the system will find a node with a name that matches the given node name.
                  If more than one node matches and exception will be raised.

        Notes:
        * If there is more than one node downstream of the given node, an exception will be raised.

        Returns:
        * A tuple containing:
          * A model name
          * A flux (variable) name
          * A dictionary of the tags to identify the model instance corresponding to the node
        '''
        tags = {}
        tags['constituent'] = consistuent
        ow_node = self.openwater_node(node,exact=exact_node_match)
        ow_model = self.model.model
        if ow_node is None: # Match a catchment
            catchment = self.catchment_for_node(node,exact=exact_node_match)
            tags['catchment'] = catchment
            transport_model,upstream_flux = self.transport_model(consistuent, position='upstream')
            if 'constituent' not in ow_model.dims_for_model(transport_model):
                tags.pop('constituent')
            return transport_model,upstream_flux,tags

        tags['node_name'] = ow_node
        models = ow_model.models_matching(**tags)
        if len(models) == 0:
            tags.pop('constituent')
            models = ow_model.models_matching(**tags)
        assert len(models)==1
        model = models[0]
        flux = DOWNSTREAM_FLUXES[model]
        return model,flux,tags

def _file_exists(fn):
    if os.path.exists(fn):
        return True
    gzfn = fn + '.gz'
    return os.path.exists(gzfn)

def _ensure_uncompressed(fn):
    if not _file_exists(fn):
        raise Exception('File not found (compressed or uncompressed): %s'%fn)
    if os.path.exists(fn):
        return
    gzfn = fn + '.gz'
    os.system('gunzip %s'%gzfn)
    assert os.path.exists(fn)
