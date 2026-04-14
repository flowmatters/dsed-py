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

UPSTREAM_FLUXES={
    'LumpedConstituentRouting':'inflowLoad',
    'ConstituentDecay':'inflowLoad',
    'InstreamDissolvedNutrientDecay':'incomingMassUpstream',
    'InstreamParticulateNutrient':'incomingMassUpstream',
    'InstreamCoarseSediment':'upstreamMass',
    'InstreamFineSediment':'upstreamMass'
}

DOWNSTREAM_FLUXES={
    'LumpedConstituentRouting':'outflowLoad',
    'ConstituentDecay':'outflowLoad',
    'InstreamDissolvedNutrientDecay':'loadDownstream',
    'InstreamParticulateNutrient':'loadDownstream',
    'InstreamCoarseSediment':'loadDownstream',
    'InstreamFineSediment':'loadDownstream',
    'EmcDwc':'outflowLoad',
    'Sum':'outflowLoad',
    'SednetDissolvedNutrientGeneration':'totalLoad',
    'SednetParticulateNutrientGeneration':'totalLoad',
    'SednetDissolvedNutrientGeneration':'totalLoad',
    'PassLoadIfFlow':'outputLoad',
    'StorageParticulateTrapping':'outflowLoad',
    'VariablePartition':'output2' # ??????
}

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

  def transport_model(self,c):
    LCR = 'LumpedConstituentRouting','outflowLoad'
    if c in self.meta['pesticides']:
      # was LCR
      return 'ConstituentDecay', 'outflowLoad'
    if c in self.meta['dissolved_nutrients']:
        return 'InstreamDissolvedNutrientDecay', 'loadDownstream'
    if c in self.meta['particulate_nutrients']:
        return 'InstreamParticulateNutrient', 'loadDownstream'
    if c == 'Sediment - Coarse':
        return 'InstreamCoarseSediment', 'loadDownstream'
    if c == 'Sediment - Fine':
        return 'InstreamFineSediment', 'loadDownstream'
    assert False

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

    def transport_model(self,c):
      return self.model.transport_model(c)

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
            transport_model,downstream_flux = self.transport_model(consistuent)
            upstream_flux = UPSTREAM_FLUXES[transport_model]
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
