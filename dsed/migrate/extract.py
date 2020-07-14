import io
import json
import os
import pandas as pd
from veneer.actions import get_big_data_source
import veneer

def _BEFORE_BATCH_NOP(slf,x,y):
    pass

STANDARD_SOURCE_ELEMENTS=[
    'Downstream Flow Volume',
    'Upstream Flow Volume'
]

STANDARD_SOURCE_VARIABLES=[
    'Quick Flow',
    'Slow Flow',
    'Total Flow'
]
SOURCE_STORAGE_VARIABLES=[
    'Storage Volume',
    'Water Surface Area',
    'Water Surface Elevation',
    'Regulated Release Volume',
    'Spill Volume',
    'Total Inflow Volume',
    'Total Outflow Volume'
]

CONSTITUENT_VARIABLES=[
    'Downstream Flow Mass',
    'Total Flow Mass',
    'Stored Concentration',
    'Stored Mass'
]

CONSTITUENT_VARIABLE_LABELS=[
    'network',
    'generation',
    'storage_concentration',
    'storage_mass'
]

class SourceExtractor(object):
    def __init__(self,v,dest,results=None,progress=print):
        self.v=v
        self.dest=dest
        self.results = results
        if self.results is None:
            self.results = os.path.join(self.dest,'Results')

        self.progress=progress

    def _ensure(self):
        if not os.path.exists(self.current_dest):
            os.makedirs(self.current_dest)

    def writeable(self,fn):
        self._ensure()
        return open(os.path.join(self.current_dest,fn),'w')

    def write_json(self,fn,data):
        json.dump(data,self.writeable(fn+'.json'))

    def write_csv(self,fn,df):
        self._ensure()
        df.to_csv(os.path.join(self.current_dest,fn+'.csv'))

    def _extract_structure(self):
        fus = set(self.v.model.catchment.get_functional_unit_types())
        constituents = self.v.model.get_constituents()
        constituent_sources = self.v.model.get_constituent_sources()
        assert len(constituent_sources)==1 # We don't use constituent source

        network = self.v.network()
        network_df = network.as_dataframe()

        fu_areas = self.v.retrieve_csv('/tables/fus')
        fu_areas = pd.DataFrame.from_csv(io.StringIO(fu_areas))

        self.write_json('constituents',constituents)
        self.write_json('fus',list(fus))
        self.writeable('network.json').write(network_df.to_json())
        self.write_csv('fu_areas',fu_areas)

    def _extract_runoff_configuration(self):
        runoff_params = self.v.model.catchment.runoff.tabulate_parameters()
        actual_rr_types = set(self.v.model.catchment.runoff.get_param_values('theBaseRRModel'))
        assert len(actual_rr_types) == 1
        sac_parameters = {k:self.v.model.catchment.runoff.get_param_values('theBaseRRModel.%s'%k) for k in self.v.model.find_parameters('TIME.Models.RainfallRunoff.Sacramento.Sacramento')}
        name_columns = self.v.model.catchment.runoff.name_columns
        sac_names = list(self.v.model.catchment.runoff.enumerate_names())
        sac_names = {col:[v[i] for v in sac_names] for i,col in enumerate(name_columns)}
        runoff_parameters = pd.DataFrame(dict(**sac_names,**sac_parameters))

        runoff_inputs = self.v.model.catchment.runoff.tabulate_inputs('Dynamic_SedNet.Models.Rainfall.DynSedNet_RRModelShell')
        self.write_csv('runoff_params',runoff_parameters)

        self.progress('Getting climate data')
        climate = get_big_data_source(self.v,'Climate Data',self.data_sources,self.progress)

        self.write_csv('climate',climate)

    def _extract_routing_configuration(self):
        link_models = self.v.model.link.routing.model_table()
        link_params = self.v.model.link.routing.tabulate_parameters('RiverSystem.Flow.StorageRouting')

        transport_models = self.v.model.link.constituents.model_table()
        transport_params = self.v.model.link.constituents.tabulate_parameters()
        self.write_csv('routing_models',link_models)
        self.write_csv('routing_params',link_params)
        self.write_csv('transportmodels',transport_models)

        for model_type, table in transport_params.items():
            self.write_csv('cr-%s'%model_type,table)

    def _extract_generation_configuration(self):
        self.progress('Getting usle timeseries')
        usle_ts = self.v.data_source('USLE Data')
        usle_timeseries = usle_ts['Items'][0]['Details']

        self.progress('Getting gully timeseries')
        gully_ts = self.v.data_source('Gully Data')
        gully_timeseries = gully_ts['Items'][0]['Details']

        self.progress('Getting cropping metadata')
        cropping_ts = get_big_data_source(self.v,'Cropping Data',self.data_sources,self.progress)

        generation_models = self.v.model.catchment.generation.model_table()
        generation_parameters = self.v.model.catchment.generation.tabulate_parameters()

        self.write_csv('usle_timeseries',usle_timeseries)
        self.write_csv('gully_timeseries',gully_timeseries)
        self.write_csv('cropping',cropping_ts)

        for model_type, table in generation_parameters.items():
            self.write_csv('cg-%s'%model_type,table)

        self.write_csv('cgmodels',generation_models)

    def _extract_demand_configuration(self):
        extraction_params = self.v.model.node.tabulate_parameters(node_types='ExtractionNodeModel')
        
        if not len(extraction_params):
            self.progress('No extraction points. Skipping water user extraction')
            return

        extraction_params = extraction_params['RiverSystem.Nodes.SupplyPoint.ExtractionNodeModel']
        # extractions = list(extraction_params['NetworkElement'])
        self.write_csv('extraction_point_params',extraction_params)

        water_users = self.v.model.node.water_users.names()
        if len(water_users):
            self.progress('Extracting information for %d water users'%len(water_users))

        demands={}
        for wu in water_users:
            d = self.v.model.node.water_users.get_param_values('DemandModel.Name',nodes=wu)[0]
        #     assert len(d) == 1
        #     d = d[0]
            if d.startswith('Time Series Demand #'):
                demands[wu] = self.v.model.node.water_users.get_data_sources('DemandModel.Order',nodes=wu)[0]
            elif d.startswith('Monthly Pattern #'):
                txt = self.v.model.node.water_users.get_param_values('DemandModel.Quantities',nodes=wu)[0]       
                demands[wu] = pd.DataFrame([{'month':ln[0],'volume':float(ln[1])} for ln in [ln.split(' ') for ln in txt.splitlines()]])
            else:
                raise Exception('Unsupported demand model: %s'%d)

        for node,demand in demands.items():
            # print(demand)
            # demand = demand.replace('%2F','/')
            # print(node,"'%s'"%demand)
            if isinstance(demand,pd.DataFrame):
                self.write_csv('monthly-pattern-demand-%s'%node,demand)
                continue

            if demand == '':
                print('No demand time series configured for node: %s'%node)
                continue

            data_source = demand.split('/')[2]
            # print(data_source)
            ds = self.v.data_source(data_source)['Items'][0]['Details']
            self.write_csv('timeseries-demand-%s'%node,ds)

    def _extract_storage_configuration(self):
        params = self.v.model.node.storages.tabulate_parameters()

        if not len(params):
            self.progress('No storages in model')
            return

        self.progress('Extracting information for %d storages'%len(params))
        params = params['RiverSystem.Nodes.StorageNodeModel']

        self.write_csv('storage_params',params)

        for storage in list(params['NetworkElement']):
            lva = self.v.model.node.storages.lva(storage)
            self.write_csv('storage_lva_%s'%storage,lva)

        outlet_links = {}
        outlets={}
        releases={}
        for ne in list(params['NetworkElement']):
            outlet_links[ne] = self.v.model.node.storages.outlets(ne)
            for link in outlet_links[ne]:
                outlets[link] = self.v.model.node.storages.releases(ne,link)
                for rel in outlets[link]:
                    releases[(ne,rel)] = self.v.model.node.storages.release_table(ne,rel)
        storage_meta = {
            'outlet_links':outlet_links,
            'outlets':outlets
        }
        self.write_json('storage_meta',storage_meta)

        for (storage,release),table in releases.items():
            self.write_csv('storage_release_%s_%s'%(storage,release),table)

    def extract_source_config(self):
        self.current_dest = self.dest
        self._ensure()

        self.progress('Getting data sources')
        self.data_sources = self.v.data_sources()

        self._extract_structure()
        self._extract_storage_configuration()
        self._extract_demand_configuration()

        self._extract_runoff_configuration()
        self._extract_generation_configuration()
        self._extract_routing_configuration()

    def _get_recorder_batches(self):
        recorders = []
        for sse in STANDARD_SOURCE_ELEMENTS:
            recorders.append([{'recorder':{'RecordingElement':sse},'retriever':{'RecordingVariable':sse},'label':sse.replace(' ','_').lower()}])
        for ssv in STANDARD_SOURCE_VARIABLES:
            recorders.append([{'recorder':{'RecordingVariable':ssv},'label':ssv.replace(' ','_')}])
        recorders.append([{'recorder':{'RecordingVariable':sv},'label':sv.replace(' ','_').lower()} for sv in SOURCE_STORAGE_VARIABLES])

        constituents = self.v.model.get_constituents()
        for c in constituents:
            recorders.append([{'recorder':{'RecordingVariable':'Constituents@%s@%s'%(c,cv)},'label':'%s%s'%(c,lbl)} for cv,lbl in zip(CONSTITUENT_VARIABLES,CONSTITUENT_VARIABLE_LABELS)])
        # for cv in CONSTITUENT_VARIABLES:
        #     recorders += [{'RecordingVariable':'Constituents@%s@%s'%(c,cv)} for c in constituents]
        return recorders

    def _configure_key_recorders(self):
        recorders = [{'RecordingElement':re} for re in STANDARD_SOURCE_ELEMENTS] + \
            [{'RecordingVariable':rv} for rv in STANDARD_SOURCE_VARIABLES + SOURCE_STORAGE_VARIABLES]

        constituents = self.v.model.get_constituents()
        for cv in CONSTITUENT_VARIABLES:
            recorders += [{'RecordingVariable':'Constituents@%s@%s'%(c,cv)} for c in constituents]

        self.v.configure_recording(enable=recorders)
        self.progress('Configured recorders')

    def extract_source_results(self,start=None,end=None,batches=False,before_batch=_BEFORE_BATCH_NOP):
        self.current_dest = self.results
        self._ensure()

        recording_batches = self._get_recorder_batches()
        if not batches:
            recording_batches = [[item for sublist in recording_batches for item in sublist]]
        for ix,batch in enumerate(recording_batches):
            before_batch(self,ix,batch)
            self.v.drop_all_runs()

            self.progress('Running batch %d of %d, with %d recorders'%(ix+1,len(recording_batches),len(batch)))
            self.progress('Running to get:\n* '+('\n* '.join([r['label'] for r in batch])))

            recorders = [r['recorder'] for r in batch]        
            self.v.configure_recording(enable=recorders,disable=[{}])

            self.v.model.simulation.configure_assurance_rule(level='Warning',category='Data Sources')

            self.v.run_model(start=start,end=end)
            self.progress('Simulation done.')

            run_summary = self.v.retrieve_run()
            results_df = run_summary['Results'].as_dataframe()
            self.progress('Got %d results'%len(results_df))

            for r in batch:
                retriever = r.get('retriever',r['recorder'])
                ts_results = self.v.retrieve_multiple_time_series(run_data=run_summary,criteria=retriever,name_fn=veneer.name_for_location)
                self.write_csv(r['label'],ts_results)

            # self.write_csv('results',results_df)
        return
        # variables = set(results_df.RecordingVariable)

        # downstream = self.v.retrieve_multiple_time_series(run_data=r,criteria={'RecordingVariable':'Downstream Flow Volume'},name_fn=veneer.name_for_location)
        # self.progress('Got downstream flow')
        # self.write_csv('downstream_vol',downstream)

        # upstream = self.v.retrieve_multiple_time_series(run_data=r,criteria={'RecordingVariable':'Upstream Flow Volume'},name_fn=veneer.name_for_location)
        # self.progress('Got upstream flow')
        # self.write_csv('upstream_vol',upstream)

        # runoff = self.v.retrieve_multiple_time_series(run_data=r,criteria={'RecordingVariable':'Quick Flow'},name_fn=veneer.name_for_fu_and_sc)
        # self.write_csv('runoff',runoff)

        # baseflow = self.v.retrieve_multiple_time_series(run_data=r,criteria={'RecordingVariable':'Slow Flow'},name_fn=veneer.name_for_fu_and_sc)
        # self.write_csv('baseflow',baseflow)

        # totalflow = self.v.retrieve_multiple_time_series(run_data=r,criteria={'RecordingVariable':'Total Flow'},name_fn=veneer.name_for_fu_and_sc)
        # self.write_csv('totalflow',totalflow)

        # def download_constituent_outputs(suffix,fn_suffix,name_fn=veneer.name_for_location):
        #     constituent_variables = [v for v in variables if v.startswith('Constituents@') and v.endswith(suffix)]
        #     self.progress(constituent_variables)
        #     for cv in constituent_variables:
        #         con = cv.split('@')[1].replace(' ','')
        #         self.progress(con)
        #         ts = self.v.retrieve_multiple_time_series(run_data=r,criteria={'RecordingVariable':cv},name_fn=name_fn)
        #         self.write_csv(con+fn_suffix,ts)
        #         self.progress('Downloaded %s %s'%(con,fn_suffix))

        # for cv,lbl in zip(CONSTITUENT_VARIABLES,CONSTITUENT_VARIABLE_LABELS):
        #     download_constituent_outputs(cv,lbl)

        # for sv in SOURCE_STORAGE_VARIABLES:
        #     ts = self.v.retrieve_multiple_time_series(run_data=r,criteria={'RecordingVariable':sv},name_fn=veneer.name_for_location)
        #     self.write_csv(sv.replace(' ','_').lower(),ts)

