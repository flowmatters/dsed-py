'''
Functionality to migrate a Source Dynamic Sednet model to Openwater (with limitations)


'''
import io
import os
import json

import pandas as pd
import numpy as np
import geopandas as gpd
from scipy import stats

from dsed.ow import DynamicSednetCatchment

import openwater.nodes as node_types
from openwater.examples import from_source
from openwater import debugging
from openwater.config import Parameteriser, ParameterTableAssignment, DataframeInput, \
                             DataframeInputs, DefaultParameteriser
from openwater.results import OpenwaterResults

from veneer.actions import get_big_data_source
import veneer

SQKM_TO_SQM = 1000*1000
M2_TO_HA = 1e-4
PER_SECOND_TO_PER_DAY=86400
PER_DAY_TO_PER_SECOND=1/PER_SECOND_TO_PER_DAY
G_TO_KG=1e-3

def nop(*args,**kwargs):
    pass

def sum_squares(l1,l2):
    return sum((np.array(l1)-np.array(l2))**2)

def levels_required(table,column,criteria):
    if len(set(table[column]))==1:
        return 0

    subsets = [table[table[criteria[0]]==v] for v in set(table[criteria[0]])]
        
    return max([1+levels_required(subset,column,criteria[1:]) for subset in subsets])

def simplify(table,column,criteria=['Constituent']):
    levels = levels_required(table,column,criteria)
    new_columns = criteria[:levels] + [column]
    return table.drop_duplicates(new_columns)[new_columns]

def extract_source_config(v,dest,progress=print):
    def writeable(fn):
        return open(os.path.join(dest,fn),'w')

    def write_json(fn,data):
        json.dump(data,writeable(fn+'.json'))

    def write_csv(fn,df):
        df.to_csv(os.path.join(dest,fn+'.csv'))

    fus = set(v.model.catchment.get_functional_unit_types())
    constituents = v.model.get_constituents()
    constituent_sources = v.model.get_constituent_sources()
    assert len(constituent_sources)==1 # We don't use constituent source

    network = v.network()
    network_df = network.as_dataframe()

    fu_areas = v.retrieve_csv('/tables/fus')
    fu_areas = pd.DataFrame.from_csv(io.StringIO(fu_areas))

    runoff_params = v.model.catchment.runoff.tabulate_parameters()
    actual_rr_types = set(v.model.catchment.runoff.get_param_values('theBaseRRModel'))
    assert len(actual_rr_types) == 1
    sac_parameters = {k:v.model.catchment.runoff.get_param_values('theBaseRRModel.%s'%k) for k in v.model.find_parameters('TIME.Models.RainfallRunoff.Sacramento.Sacramento')}
    name_columns = v.model.catchment.runoff.name_columns
    sac_names = list(v.model.catchment.runoff.enumerate_names())
    sac_names = {col:[v[i] for v in sac_names] for i,col in enumerate(name_columns)}
    runoff_parameters = pd.DataFrame(dict(**sac_names,**sac_parameters))

    runoff_inputs = v.model.catchment.runoff.tabulate_inputs('Dynamic_SedNet.Models.Rainfall.DynSedNet_RRModelShell')

    progress('Getting data sources')
    data_sources = v.data_sources()
    #[ds['FullName'] for ds in data_sources]
    progress('Getting climate data')
    climate = get_big_data_source(v,'Climate Data',data_sources,progress)

    progress('Getting usle timeseries')
    usle_ts = v.data_source('USLE Data')
    usle_timeseries = usle_ts['Items'][0]['Details']
    
    progress('Getting gully timeseries')
    gully_ts = v.data_source('Gully Data')
    gully_timeseries = gully_ts['Items'][0]['Details']

    progress('Getting cropping metadata')
    cropping_ts = get_big_data_source(v,'Cropping Data',data_sources,progress)

    generation_models = v.model.catchment.generation.model_table()
    generation_parameters = v.model.catchment.generation.tabulate_parameters()

    link_models = v.model.link.routing.model_table()
    link_params = v.model.link.routing.tabulate_parameters()

    transport_models = v.model.link.constituents.model_table()
    transport_params = v.model.link.constituents.tabulate_parameters()

    write_json('constituents',constituents)
    write_json('fus',list(fus))
    writeable('network.json').write(network_df.to_json())
    write_csv('runoff_params',runoff_parameters)
    write_csv('climate',climate)
    write_csv('usle_timeseries',usle_timeseries)
    write_csv('gully_timeseries',gully_timeseries)
    write_csv('cropping',cropping_ts)

    for model_type, table in generation_parameters.items():
        write_csv('cg-%s'%model_type,table)

    write_csv('cgmodels',generation_models)
    write_csv('routing_models',link_models)
    write_csv('routing_models',link_models)
    write_csv('transportmodels',transport_models)

    for model_type, table in transport_params.items():
        write_csv('cr-%s'%model_type,table)

    write_csv('fu_areas',fu_areas)

def extract_source_results(v,dest,progress=print,start=None,end=None):
    def _ensure():
        if not os.path.exists(dest):
            os.makedirs(dest)

    def writeable(fn):
        _ensure()
        return open(os.path.join(dest,fn),'w')

    def write_json(fn,data):
        json.dump(data,writeable(fn+'.json'))

    def write_csv(fn,df):
        _ensure()
        df.to_csv(os.path.join(dest,fn+'.csv'))

    constituents = v.model.get_constituents()
    recorders = [
        {'RecordingElement':'Downstream Flow Volume'},
        {'RecordingElement':'Upstream Flow Volume'}
    ]
    recorders += [{'RecordingVariable':'Constituents@%s@Downstream Flow Mass'%c} for c in constituents]
    recorders += [{'RecordingVariable':'Constituents@%s@Total Flow Mass'%c} for c in constituents]

    v.configure_recording(enable=recorders)
    progress('Configured recorders')

    v.drop_all_runs()
    v.model.simulation.configure_assurance_rule(level='Warning',category='Data Sources')

    v.run_model(start=start,end=end)
    progress('Simulation done.')

    r = v.retrieve_run()
    variables = set(r['Results'].as_dataframe().RecordingVariable)

    downstream = v.retrieve_multiple_time_series(run_data=r,criteria={'RecordingVariable':'Downstream Flow Volume'},name_fn=veneer.name_for_location)
    progress('Got downstream flow')

    upstream = v.retrieve_multiple_time_series(run_data=r,criteria={'RecordingVariable':'Downstream Flow Volume'},name_fn=veneer.name_for_location)
    progress('Got upstream flow')

    write_csv('upstream_vol',upstream)
    write_csv('downstream_vol',downstream)

    def download_constituent_outputs(suffix,fn_suffix,name_fn=veneer.name_for_location):
        constituent_variables = [v for v in variables if v.startswith('Constituents@') and v.endswith(suffix)]
        progress(constituent_variables)
        for cv in constituent_variables:
            con = cv.split('@')[1].replace(' ','')
            progress(con)
            ts = v.retrieve_multiple_time_series(run_data=r,criteria={'RecordingVariable':cv},name_fn=name_fn)
            write_csv(con+fn_suffix,ts)
            progress('Downloaded %s %s'%(con,fn_suffix))

    download_constituent_outputs('Downstream Flow Mass','network')
    download_constituent_outputs('Total Flow Mass','generation',veneer.name_for_fu_and_sc)

def build_ow_model(data_path,start='1986/07/01',end='2014/06/30',
                   link_renames={},
                   progress=print):
  meta = {}
  def load_json(f):
    return json.load(open(os.path.join(data_path,f+'.json')))

  def load_csv(f):
    return pd.read_csv(os.path.join(data_path,f+'.csv'),index_col=0,parse_dates=True)

  network = gpd.read_file(os.path.join(data_path,'network.json'))

  time_period = pd.date_range(start,end)
  meta['start'] = start
  meta['end'] = end

  orig_climate = load_csv('climate')
  cropping = load_csv('cropping')
  cropping = cropping.reindex(time_period)
  constituents = load_json('constituents')
  meta['constituents'] = constituents

  fus = load_json('fus')
  meta['fus'] = fus

  pesticide_cgus = set([c.split('$')[-1] for c in cropping.columns if 'Dissolved_Load_g_per_Ha' in c])
  cg_models = load_csv('cgmodels')
  cg_models = simplify(cg_models,'model',['Constituent','Functional Unit','Catchment'])

  fine_sed_cg = cg_models[cg_models.Constituent=='Sediment - Fine']
  fine_sed_cg = dict(zip(fine_sed_cg['Functional Unit'],fine_sed_cg.model))
  erosion_cgus = [fu for fu,model in fine_sed_cg.items() if model == 'Dynamic_SedNet.Models.SedNet_Sediment_Generation']
  emc_cgus = [fu for fu,model in fine_sed_cg.items() if model == 'RiverSystem.Catchments.Models.ContaminantGenerationModels.EmcDwcCGModel']
  meta['erosion_cgus'] = erosion_cgus

  cr_models = load_csv('transportmodels')
  cr_models = simplify(cr_models,'model',['Constituent'])

  routing_models = load_csv('routing_models')
  routing_models.replace(link_renames,inplace=True)

  RR = node_types.Sacramento
  ROUTING= node_types.StorageRouting

  dissolved_nutrients = [c for c in constituents if '_D' in c or '_F' in c]
  meta['dissolved_nutrients'] = dissolved_nutrients

  particulate_nutrients = [c for c in constituents if '_Particulate' in c]
  meta['particulate_nutrients'] = particulate_nutrients

  sediments = [c for c in constituents if c.startswith('Sediment - ')]
  meta['sediments'] = sediments

  pesticides = [c for c in constituents if not c in dissolved_nutrients+particulate_nutrients+sediments]
  meta['pesticides'] = pesticides

  catchment_template = DynamicSednetCatchment(dissolved_nutrients=[],  #dissolved_nutrients,
                                              particulate_nutrients=[], #particulate_nutrients,
                                              pesticides=pesticides) #pesticides)

  catchment_template.hrus = fus
  catchment_template.cgus = fus
  catchment_template.cgu_hrus = {fu:fu for fu in fus}
  catchment_template.pesticide_cgus = pesticide_cgus
  catchment_template.erosion_cgus = erosion_cgus
  catchment_template.sediment_fallback_cgu = emc_cgus

  catchment_template.rr = RR

  default_cg = catchment_template.cg.copy()

  def generation_model_for_constituent_and_fu(constituent,cgu=None):
      progress(constituent,cgu)
      return default_cg[constituent]

  catchment_template.routing = ROUTING

  tpl_nested = catchment_template.get_template()
  tpl = tpl_nested.flatten()

  template_image = debugging.graph_template(tpl)
  template_image.render('cgu_template',format='png')

  model = from_source.build_catchment_graph(catchment_template,network,progress=nop)
  progress('Model built')

  p = Parameteriser()
  model._parameteriser = p
  p._parameterisers.append(DefaultParameteriser())
  climate_ts = orig_climate.reindex(time_period)
  i = DataframeInputs()
  i.inputter(climate_ts,'rainfall','rainfall for ${catchment}')
  i.inputter(climate_ts,'pet','pet for ${catchment}')

  p._parameterisers.append(i)
  fu_areas = load_csv('fu_areas')
  fu_areas_parameteriser = ParameterTableAssignment(fu_areas,'DepthToRate','area','cgu','catchment')
  p._parameterisers.append(fu_areas_parameteriser)


  sacramento_parameters = load_csv('runoff_params')
  sacramento_parameters['hru'] = sacramento_parameters['Functional Unit']
  sacramento_parameters = sacramento_parameters.rename(columns={c:c.lower() for c in sacramento_parameters.columns})
  runoff_parameteriser = ParameterTableAssignment(sacramento_parameters,RR,dim_columns=['catchment','hru'])
  p._parameterisers.append(runoff_parameteriser)

  routing_params = load_csv('routing')
  routing_params.replace(link_renames,inplace=True)
  routing_params['catchment'] = routing_params.NetworkElement.str.slice(19)
  routing_parameteriser = ParameterTableAssignment(routing_params,ROUTING,dim_columns=['catchment'])
  p._parameterisers.append(routing_parameteriser)

  dwcs = load_csv('cg-GBR_DynSed_Extension.Models.GBR_Pest_TSLoad_Model')
  dwcs = dwcs.rename(columns={'Catchment':'catchment','Functional Unit':'cgu','Constituent':'constituent'})

  dwcs['particulate_scale'] = 0.01 * dwcs['DeliveryRatio'] * 0.01 * dwcs['Fine_Percent']
  dwcs['dissolved_scale'] = 0.01 * dwcs['DeliveryRatioDissolved']

  dwcs['final_scale'] = dwcs['Load_Conversion_Factor']

  cropping_inputs = DataframeInputs()
  scaled_cropping_ts = {}
  for col in cropping.columns:
      constituent, variable, catchment, cgu = col.split('$')
      if variable != 'Dissolved_Load_g_per_Ha' and variable != 'Particulate_Load_g_per_Ha':
          continue
      
      row = dwcs[dwcs.catchment==catchment]
      row = row[row.constituent==constituent]
      row = row[row.cgu==cgu]
      assert len(row)==1
      row = row.iloc[0]
      
      scale = row['%s_scale'%(variable.split('_')[0].lower())] * row['final_scale']

      scaled_cropping_ts[col] = scale * cropping[col]
  scaled_cropping = pd.DataFrame(scaled_cropping_ts)

  combined_cropping_ts = {}
  for col in scaled_cropping.columns:
      constituent, variable, catchment, cgu = col.split('$')
      if variable != 'Dissolved_Load_g_per_Ha': 
          continue
      #  and variable != 'Particulate_Load_g_per_Ha':
      dis_col = col
      part_col = '%s$Particulate_Load_g_per_Ha$%s$%s'%(constituent,catchment,cgu)
      comb_col = '%s$Combined_Load_g_per_Ha$%s$%s'%(constituent,catchment,cgu)
      combined_cropping_ts[comb_col] = scaled_cropping[dis_col] + scaled_cropping[part_col]
  combined_cropping = pd.DataFrame(combined_cropping_ts)

  # Particulate_Load_g_per_Ha or Dissolved_Load_g_per_Ha
  cropping_inputs.inputter(combined_cropping,'inputLoad','${constituent}$$Combined_Load_g_per_Ha$$${catchment}$$${cgu}')
  # TODO NEED TO SCALE BY AREA!
  p._parameterisers.append(cropping_inputs)

  fu_areas_scaled = fu_areas * M2_TO_HA * PER_DAY_TO_PER_SECOND * G_TO_KG
  cropping_ts_scaling = ParameterTableAssignment(fu_areas_scaled,'PassLoadIfFlow','scalingFactor','cgu','catchment')
  p._parameterisers.append(cropping_ts_scaling)

  usle_timeseries = load_csv('usle_timeseries').reindex(time_period)
  usle_timeseries = usle_timeseries.fillna(method='ffill')
  fine_sediment_params = load_csv('cg-Dynamic_SedNet.Models.SedNet_Sediment_Generation')
  assert len(set(fine_sediment_params.Constituent))==1
  assert set(fine_sediment_params.useAvModel) == {False}
  fine_sediment_params = fine_sediment_params.rename(columns={
      'Max_Conc':'maxConc',
      'USLE_HSDR_Fine':'usleHSDRFine',
      'USLE_HSDR_Coarse':'usleHSDRCoarse',
      'Catchment':'catchment',
      'Functional Unit':'cgu',
      'Constituent':'constituent'
  })

  fine_sediment_params = fine_sediment_params.rename(columns={c:c.replace('_','') for c in fine_sediment_params.columns})
  usle_parameters = ParameterTableAssignment(fine_sediment_params,node_types.USLEFineSedimentGeneration,dim_columns=['catchment','cgu'])
  model._parameteriser.append(usle_parameters)

  usle_timeseries_inputs = DataframeInputs()
  usle_timeseries_inputs.inputter(usle_timeseries,'KLSC','KLSC_Total For ${catchment} ${cgu}')
  usle_timeseries_inputs.inputter(usle_timeseries,'KLSC_Fine','KLSC_Fines For ${catchment} ${cgu}')
  usle_timeseries_inputs.inputter(usle_timeseries,'CovOrCFact','C-Factor For ${catchment} ${cgu}')
  model._parameteriser.append(usle_timeseries_inputs)

  fine_sediment_params = fine_sediment_params.rename(columns={
      'GullyYearDisturb':'YearDisturbance',
      'AverageGullyActivityFactor':'averageGullyActivityFactor',
      'GullyManagementPracticeFactor':'managementPracticeFactor',
      'GullySDRFine':'sdrFine',
      'GullySDRCoarse':'sdrCoarse'
  })
  # Area, AnnualRunoff, GullyAnnualAverageSedimentSupply, annualLoad, longtermRunoffFactor
  # dailyRunoffPowerFactor
  gully_parameters = ParameterTableAssignment(fine_sediment_params,node_types.DynamicSednetGully,dim_columns=['catchment','cgu'])
  model._parameteriser.append(usle_parameters)

  emc_dwc = load_csv('cg-RiverSystem.Catchments.Models.ContaminantGenerationModels.EmcDwcCGModel')
  emc_dwc = emc_dwc.rename(columns={
      'Catchment':'catchment',
      'Functional Unit':'cgu',
      'Constituent':'constituent',
      'eventMeanConcentration':'EMC',
      'dryMeanConcentration':'DWC'
  })
  emc_parameteriser = ParameterTableAssignment(emc_dwc,node_types.EmcDwc,dim_columns=['catchment','cgu','constituent'],complete=False)
  model._parameteriser.append(emc_parameteriser)

  model._parameteriser = p

  return model, meta, network

class SourceOWComparison(object):
    def __init__(self,meta,ow_model_fn,ow_results_fn,source_results_path,catchments):
        self.meta = meta
        self.ow_model_fn = ow_model_fn
        self.ow_results_fn = ow_results_fn
        self.source_results_path = source_results_path
        self.time_period = pd.date_range(self.meta['start'],self.meta['end'])
        self.catchments = catchments
        self.results = OpenwaterResults(ow_model_fn,ow_results_fn,self.time_period)

        self.comparison_flows = None
        self.link_outflow = None

    def _load_csv(self,f):
        return pd.read_csv(os.path.join(self.source_results_path,f.replace(' ','')+'.csv'),index_col=0,parse_dates=True)

    def _load_flows(self):
        if self.comparison_flows is not None:
            return

        routing = 'StorageRouting'
        self.link_outflow = self.results.time_series(routing,'outflow','catchment')
        self.comparison_flows = self._load_csv('Results/downstream_vol')
        self.comparison_flows = self.comparison_flows.rename(columns={c:c.replace('link for catchment ','') for c in self.comparison_flows.columns})
        self.comparison_flows = self.comparison_flows / 86400.0

    def comparable_flows(self,sc):
        self._load_flows()

        if not sc in self.comparison_flows.columns or not sc in self.link_outflow.columns:
            return None,None

        orig = self.comparison_flows[sc]
        ow = self.link_outflow[sc]
        common = orig.index.intersection(ow.index)
        orig = orig[common]
        ow = ow[common]
        return orig,ow

    def compare_flow(self,sc):
        orig, ow = self.comparable_flows(sc)
        if orig is None or ow is None:
            return np.nan

        _,_,r_value,_,_ = stats.linregress(orig,ow)
        return r_value**2

    def compare_flows(self):
        self._load_flows()
        return pd.Series([self.compare_flow(c) for c in self.link_outflow.columns],index=self.link_outflow.columns)

    def plot_flows(self,sc):
        import matplotlib.pyplot as plt
        orig, ow = self.comparable_flows(sc)
        plt.figure()
        orig.plot(label='orig')
        ow.plot(label='ow')
        plt.legend()
        
        plt.figure()
        plt.scatter(orig,ow)

    def generation_model(self,c,fu):
        if c in self.meta['sediments']:
            if fu in self.meta['erosion_cgus']:
                return 'Sum','out'
            return 'EmcDwc','totalLoad'

        if c in self.meta['pesticides']:
            return 'Sum','out'

        return 'EmcDwc','totalLoad'

    def transport_model(self,c):
        if c in self.meta['pesticides']:
            return 'LumpedConstituentRouting','outflowLoad'
        assert False

    def compare_constituent_generation(self,constituents=None,progress=print):
        if constituents is None:
            constituents = self.meta['constituents']

        errors = []
        for c in constituents:
            progress(c)
            comparison = self._load_csv('Results/%sgeneration'%c).reindex(self.time_period)
            for fu in self.meta['fus']:
                model,output = self.generation_model(c,fu)
                ow = self.results.time_series(model,output,'catchment',cgu=fu,constituent=c)
                comparison_columns = ['%s: %s'%(fu,catchment) for catchment in ow.columns]
                fu_comparison = comparison[comparison_columns]
                if ow.sum().sum()==0 and fu_comparison.sum().sum()==0:
                    for sc in ow.columns:
                        errors.append({'catchment':sc,'cgu':fu,'constituent':c,'ssquares':0,'sum-ow':0,'sum-orig':0})
                else:
                    for sc in ow.columns:
                        res = {'catchment':sc,'cgu':fu,'constituent':c,'ssquares':0,'sum-ow':0,'sum-orig':0}
                        ow_sc = ow[sc]
                        orig_sc = fu_comparison['%s: %s'%(fu,sc)]
                        if ow_sc.sum()>0 or orig_sc.sum()>0:
                            orig_scaled = (orig_sc*86400)
                            ow_scaled = (ow_sc*86400)
                            res['ssquares'] = sum_squares(orig_scaled,ow_scaled)
                            res['sum-ow'] = ow_scaled.sum()
                            res['sum-orig'] = orig_scaled.sum()
                        errors.append(res)
        return pd.DataFrame(errors)

    def compare_constituent_transport(self,constituents=None,progress=print):
        if constituents is None:
            constituents = self.meta['constituents']

        SOURCE_COL_PREFIX='link for catchment '
        errors = []
        for c in constituents:
            progress(c)
            comparison = self._load_csv('Results/%snetwork'%c).reindex(self.time_period)
            comparison = comparison[[catchment for catchment in comparison.columns if catchment.startswith(SOURCE_COL_PREFIX)]]
            comparison = comparison.rename(columns={catchment:catchment.replace(SOURCE_COL_PREFIX,'') for catchment in comparison.columns})
            comparison = comparison * PER_DAY_TO_PER_SECOND
            model,output = self.transport_model(c)
            ow = self.results.time_series(model,output,'catchment',constituent=c)
            # progress(comparison.columns)
            # progress(ow.columns)
            for sc in ow.columns:
                if not sc in comparison:
                    continue

                res = {'catchment':sc,'constituent':c,'ssquares':0,'sum-ow':0,'sum-orig':0}
                ow_sc = ow[sc]
                orig_sc = comparison[sc]
                if ow_sc.sum()>0 or orig_sc.sum()>0:
                    orig_scaled = (orig_sc*86400)
                    ow_scaled = (ow_sc*86400)
                    res['ssquares'] = sum_squares(orig_scaled,ow_scaled)
                    res['sum-ow'] = ow_scaled.sum()
                    res['sum-orig'] = orig_scaled.sum()
                errors.append(res)
        #    break
        return pd.DataFrame(errors)

