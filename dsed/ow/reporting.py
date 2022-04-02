from openwater.examples import OpenwaterCatchmentModelResults

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
            'crs':raw[0]['crs'],
            'features':sum([r['features'] for r in raw],[])
        }
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
