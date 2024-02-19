from openwater.examples import OpenwaterCatchmentModelResults
from openwater.results import OpenwaterResults
from openwater.template import ModelFile
from .structure import dissolved_nutrient_ts_load
import json
import geopandas as gpd
import pandas as pd
import os

class OpenwaterDynamicSednetResults(OpenwaterCatchmentModelResults):
    def __init__(self, fn, res_fn=None):
        self.fn = fn
        self.ow_model_fn = self.filename_from_base('.h5')
        self.meta = json.load(open(self.filename_from_base('.meta.json')))
        self.init_network(fn)

        self.ow_results_fn = res_fn or self.filename_from_base('_outputs.h5')
        self.dates = pd.date_range(self.meta['start'], self.meta['end'])
        self.time_period = self.dates
        self.open_files()

    def filename_from_base(self,fn):
        return self.fn.replace('.h5','')+fn

    def init_network(self,fn):
        from veneer.general import _extend_network
        self.nodes = gpd.read_file(self.filename_from_base('.nodes.json'))
        self.links = gpd.read_file(self.filename_from_base('.links.json'))
        self.catchments = gpd.read_file(self.filename_from_base('.catchments.json'))
        raw = [json.load(open(self.filename_from_base('.'+c+'.json'),'r')) for c in ['nodes','links','catchments']]
        self.network = {
            'type':'FeatureCollection',
            'features':sum([r['features'] for r in raw],[])
        }
        if 'crs' in raw[0]:
          self.network['crs']=raw[0]['crs']

        self.network = _extend_network(self.network)

    def run_model(self):
        self.model.run(self.dates, self.ow_results_fn, overwrite=True)
        self.open_files()

    def open_files(self):
        _ensure_uncompressed(self.ow_model_fn)
        _ensure_uncompressed(self.ow_results_fn)

        self.results = OpenwaterResults(self.ow_model_fn,
                                        self.ow_results_fn,
                                        self.dates)
        self.model = ModelFile(self.ow_model_fn)

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

def _ensure_uncompressed(fn):
    if os.path.exists(fn):
        return
    gzfn = fn + '.gz'
    if not os.path.exists(gzfn):
        raise Exception('File not found (compressed or uncompressed): %s'%fn)
    os.system('gunzip %s'%gzfn)
    assert os.path.exists(fn)
