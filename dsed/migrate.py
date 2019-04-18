'''
Functionality to migrate a Source Dynamic Sednet model to Openwater (with limitations)


'''
import os
import pandas as pd
import geopandas as gpd
import json
import openwater.nodes as node_types
from openwater.examples import from_source
from dsed.ow import DynamicSednetCatchment
from openwater import debugging
from openwater.config import Parameteriser, ParameterTableAssignment, DataframeInput, \
                             DataframeInputs, DefaultParameteriser

SQKM_TO_SQM = 1000*1000
M2_TO_HA = 1e-4
PER_SECOND_TO_PER_DAY=86400
PER_DAY_TO_PER_SECOND=1/PER_SECOND_TO_PER_DAY
G_TO_KG=1e-3

def nop(*args,**kwargs):
    pass

def levels_required(table,column,criteria):
    if len(set(table[column]))==1:
        return 0

    subsets = [table[table[criteria[0]]==v] for v in set(table[criteria[0]])]
        
    return max([1+levels_required(subset,column,criteria[1:]) for subset in subsets])

def simplify(table,column,criteria=['Constituent']):
    levels = levels_required(table,column,criteria)
    new_columns = criteria[:levels] + [column]
    return table.drop_duplicates(new_columns)[new_columns]

def build_ow_model(data_path,start='1986/07/01',end='2014/06/30',
                   link_renames={},
                   progress=print):
  def load_json(f):
    return json.load(open(os.path.join(data_path,f+'.json')))

  def load_csv(f):
    return pd.read_csv(os.path.join(data_path,f+'.csv'),index_col=0,parse_dates=True)

  network = gpd.read_file(os.path.join(data_path,'network.json'))

  time_period = pd.date_range(start,end)

  orig_climate = load_csv('climate')
  cropping = load_csv('cropping')
  cropping = cropping.reindex(time_period)
  constituents = load_json('constituents')
  fus = load_json('fus')
  pesticide_cgus = set([c.split('$')[-1] for c in cropping.columns if 'Dissolved_Load_g_per_Ha' in c])
  cg_models = load_csv('cgmodels')
  cg_models = simplify(cg_models,'model',['Constituent','Functional Unit','Catchment'])

  fine_sed_cg = cg_models[cg_models.Constituent=='Sediment - Fine']
  fine_sed_cg = dict(zip(fine_sed_cg['Functional Unit'],fine_sed_cg.model))
  erosion_cgus = [fu for fu,model in fine_sed_cg.items() if model == 'Dynamic_SedNet.Models.SedNet_Sediment_Generation']
  emc_cgus = [fu for fu,model in fine_sed_cg.items() if model == 'RiverSystem.Catchments.Models.ContaminantGenerationModels.EmcDwcCGModel']

  cr_models = load_csv('transportmodels')
  cr_models = simplify(cr_models,'model',['Constituent'])

  routing_models = load_csv('routing_models')
  routing_models.replace(link_renames,inplace=True)

  RR = node_types.Sacramento
  ROUTING= node_types.StorageRouting

  dissolved_nutrients = [c for c in constituents if '_D' in c or '_F' in c]
  particulate_nutrients = [c for c in constituents if '_Particulate' in c]
  sediments = [c for c in constituents if c.startswith('Sediment - ')]
  pesticides = [c for c in constituents if not c in dissolved_nutrients+particulate_nutrients+sediments]

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

  model._parameteriser = p

  return model